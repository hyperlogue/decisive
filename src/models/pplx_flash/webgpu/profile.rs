#[cfg(any(feature = "kernel-checks", feature = "pplx-webgpu"))]
use anyhow::Context;
use anyhow::{Result, ensure};
use std::cell::RefCell;

const QUERIES: u32 = 2048;
pub(super) struct Profile {
    queries: wgpu::QuerySet,
    entries: RefCell<Option<Vec<(String, [u32; 16])>>>,
}
impl Profile {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Pplx kernel timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERIES,
            }),
            entries: RefCell::new(None),
        }
    }
    #[cfg(any(feature = "kernel-checks", feature = "pplx-webgpu"))]
    pub fn begin(&self) {
        *self.entries.borrow_mut() = Some(Vec::new());
    }
    pub fn writes(
        &self,
        name: &str,
        params: &[u32; 16],
    ) -> Result<Option<wgpu::ComputePassTimestampWrites<'_>>> {
        let mut entries = self.entries.borrow_mut();
        let Some(entries) = entries.as_mut() else {
            return Ok(None);
        };
        let index = entries.len() as u32 * 2;
        ensure!(index + 1 < QUERIES, "kernel timestamp capacity exceeded");
        entries.push((name.to_owned(), *params));
        Ok(Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: Some(index + 1),
        }))
    }
    #[cfg(any(feature = "kernel-checks", feature = "pplx-webgpu"))]
    pub async fn finish(&self, device: &super::Device) -> Result<serde_json::Value> {
        let entries = self
            .entries
            .borrow_mut()
            .take()
            .context("profile not started")?;
        ensure!(!entries.is_empty(), "empty kernel profile");
        let count = entries.len() as u32 * 2;
        let size = count as u64 * 8;
        let resolve = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pplx timestamps resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pplx timestamps readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.encoder();
        encoder.resolve_query_set(&self.queries, 0..count, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, size);
        device.queue.submit([encoder.finish()]);
        let (send, receive) = futures_channel::oneshot::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
        #[cfg(not(target_arch = "wasm32"))]
        device
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("profile completion")?;
        receive.await.context("profile readback cancelled")??;
        let ticks = {
            let view = staging.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, u64>(&view).to_vec()
        };
        staging.unmap();
        staging.destroy();
        resolve.destroy();
        let period = device.queue.get_timestamp_period() as f64 / 1e6;
        let mut dispatches = Vec::new();
        for (i, (name, params)) in entries.iter().enumerate() {
            ensure!(
                ticks[i * 2 + 1] >= ticks[i * 2],
                "invalid kernel timestamp order"
            );
            dispatches.push(serde_json::json!({"kernel":name,"params":params,
                "start_ms":(ticks[i*2].saturating_sub(ticks[0])) as f64 * period,
                "end_ms":(ticks[i*2+1].saturating_sub(ticks[0])) as f64 * period,
                "gpu_ms":(ticks[i*2+1]-ticks[i*2]) as f64 * period}));
        }
        Ok(serde_json::json!({"dispatches":dispatches,
            "origin_ticks":ticks[0].to_string(),"tick_period_ns":device.queue.get_timestamp_period(),
            "span_ms":(ticks[ticks.len()-1]-ticks[0]) as f64 * period,
            "instrumented":true,"note":"Pass intervals may overlap; their sum is not decoder latency"}))
    }
}
