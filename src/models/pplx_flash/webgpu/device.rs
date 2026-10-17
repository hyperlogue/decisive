use anyhow::{Context, Result, ensure};
use wgpu::util::DeviceExt;

pub(super) struct Device {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
}
impl Device {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context("no WebGPU compute adapter")?;
        let info = adapter.get_info();
        let limits = wgpu::Limits::default();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Pplx Flash inference"),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .context("request WebGPU device")?;
        Ok(Self {
            device,
            queue,
            info,
        })
    }
    pub fn buffer(&self, values: &[f32], label: &str) -> Result<wgpu::Buffer> {
        self.bytes(bytemuck::cast_slice(values), label)
    }
    pub fn bytes(&self, bytes: &[u8], label: &str) -> Result<wgpu::Buffer> {
        self.check_size(bytes.len())?;
        ensure!(bytes.len() % 4 == 0, "unaligned GPU buffer");
        Ok(self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            }))
    }
    pub fn empty(&self, elements: usize, label: &str) -> Result<wgpu::Buffer> {
        let bytes = elements
            .checked_mul(4)
            .context("GPU buffer size overflow")?;
        self.check_size(bytes)?;
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }
    fn check_size(&self, bytes: usize) -> Result<()> {
        let limits = self.device.limits();
        ensure!(
            bytes > 0
                && bytes as u64 <= limits.max_buffer_size
                && bytes as u64 <= limits.max_storage_buffer_binding_size as u64,
            "GPU storage buffer of {bytes} bytes exceeds device limits"
        );
        Ok(())
    }
    pub fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Pplx Flash request"),
            })
    }
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
        params: &[u32; 16],
        groups: [u32; 3],
    ) -> Result<()> {
        let limit = self.device.limits().max_compute_workgroups_per_dimension;
        ensure!(
            groups.iter().all(|&n| n > 0 && n <= limit),
            "GPU dispatch exceeds device limits"
        );
        // Uniforms live until submission completes; reusing one would overwrite earlier dispatches.
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pplx Flash dispatch parameters"),
                contents: bytemuck::cast_slice(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: buffers.len() as u32,
            resource: uniform.as_entire_binding(),
        });
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pplx Flash dispatch"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Pplx Flash compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        Ok(())
    }
    pub async fn read(
        &self,
        mut encoder: wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        elements: usize,
    ) -> Result<Vec<f32>> {
        let bytes = elements.checked_mul(4).context("GPU read size overflow")? as u64;
        ensure!(
            bytes > 0 && bytes <= source.size(),
            "GPU read outside source buffer"
        );
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pplx Flash result"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let (send, receive) = futures_channel::oneshot::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("GPU completion")?;
        receive.await.context("GPU readback cancelled")??;
        let values = {
            let view = staging.slice(..).get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        staging.unmap();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite GPU result"
        );
        Ok(values)
    }
}
