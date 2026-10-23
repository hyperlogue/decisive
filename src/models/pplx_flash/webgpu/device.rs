use anyhow::{Context, Result, ensure};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
};
use wgpu::util::DeviceExt;

struct PendingDispatch {
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    groups: [u32; 3],
}

pub(super) struct Completion {
    receive: futures_channel::oneshot::Receiver<()>,
    #[cfg(not(target_arch = "wasm32"))]
    device: wgpu::Device,
}
impl Completion {
    pub async fn wait(self) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        self.receive.await.context("GPU completion cancelled")
    }
}

pub(super) struct Device {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub profile: Option<super::profile::Profile>,
    pub batch_passes: Cell<bool>,
    pending: RefCell<Vec<PendingDispatch>>,
    cached_uniforms: RefCell<HashMap<[u32; 16], wgpu::Buffer>>,
    cached_bindings: RefCell<HashMap<(wgpu::ComputePipeline, Vec<wgpu::Buffer>, [u32; 16]), wgpu::BindGroup>>,
}
impl Device {
    pub async fn with_features(features: wgpu::Features) -> Result<Self> {
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
        ensure!(
            adapter.features().contains(features),
            "required WebGPU precision is unavailable"
        );
        let mut limits = wgpu::Limits::default();
        limits.max_compute_workgroup_storage_size = adapter.limits().max_compute_workgroup_storage_size.min(32768);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Pplx Flash inference"),
                required_limits: limits,
                required_features: features,
                ..Default::default()
            })
            .await
            .context("request WebGPU device")?;
        Ok(Self {
            batch_passes: Cell::new(false),
            pending: RefCell::new(Vec::new()),
            cached_uniforms: RefCell::new(HashMap::new()),
            cached_bindings: RefCell::new(HashMap::new()),
            profile: features
                .contains(wgpu::Features::TIMESTAMP_QUERY)
                .then(|| super::profile::Profile::new(&device)),
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
    pub fn discard_pending(&self) {
        self.pending.borrow_mut().clear();
        self.cached_bindings.borrow_mut().clear();
        for (_, uniform) in self.cached_uniforms.borrow_mut().drain() {
            uniform.destroy();
        }
    }
    pub fn submit_commands(&self, mut encoder: wgpu::CommandEncoder) {
        self.flush(&mut encoder);
        self.queue.submit([encoder.finish()]);
    }
    pub fn submit_deferred(&self, encoder: wgpu::CommandEncoder) -> Completion {
        self.submit_commands(encoder);
        self.completion()
    }
    pub fn idle_completion(&self) -> Completion {
        self.pending.borrow_mut().clear();
        self.queue.submit([]);
        self.completion()
    }
    fn completion(&self) -> Completion {
        let (send, receive) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = send.send(());
        });
        Completion {
            receive,
            #[cfg(not(target_arch = "wasm32"))]
            device: self.device.clone(),
        }
    }
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
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
        let key = (pipeline.clone(), buffers.iter().map(|&b| b.clone()).collect(), *params);
        let cached = self.cached_bindings.borrow().get(&key).cloned();
        let bindings = match cached {
            Some(bindings) => bindings,
            None => {
                let uniform = self
                    .cached_uniforms
                    .borrow_mut()
                    .entry(*params)
                    .or_insert_with(|| {
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Pplx Flash dispatch parameters"),
                                contents: bytemuck::cast_slice(params),
                                usage: wgpu::BufferUsages::UNIFORM,
                            })
                    })
                    .clone();
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
                self.cached_bindings.borrow_mut().insert(key, bindings.clone());
                bindings
            }
        };
        if self.batch_passes.get() {
            ensure!(
                self.profile.is_none(),
                "batched passes do not support per-dispatch timestamps"
            );
            self.pending.borrow_mut().push(PendingDispatch {
                pipeline: pipeline.clone(),
                bindings,
                groups,
            });
            return Ok(());
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Pplx Flash compute"),
            timestamp_writes: self
                .profile
                .as_ref()
                .map(|profile| profile.writes(name, params))
                .transpose()?
                .flatten(),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        Ok(())
    }
    fn flush(&self, encoder: &mut wgpu::CommandEncoder) {
        let pending = self.pending.take();
        if pending.is_empty() {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Pplx Flash batched layer"),
            timestamp_writes: None,
        });
        for dispatch in &pending {
            pass.set_pipeline(&dispatch.pipeline);
            pass.set_bind_group(0, &dispatch.bindings, &[]);
            pass.dispatch_workgroups(dispatch.groups[0], dispatch.groups[1], dispatch.groups[2]);
        }
    }
    pub async fn read(
        &self,
        encoder: wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        elements: usize,
    ) -> Result<Vec<f32>> {
        let values: Vec<_> = self
            .read_words(encoder, source, elements)
            .await?
            .into_iter()
            .map(f32::from_bits)
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite GPU result"
        );
        Ok(values)
    }
    pub async fn read_words(
        &self,
        encoder: wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        elements: usize,
    ) -> Result<Vec<u32>> {
        self.read_words_submitted(encoder, source, elements, || {})
            .await
    }
    pub async fn read_submitted(
        &self,
        encoder: wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        elements: usize,
        submitted: impl FnOnce(),
    ) -> Result<Vec<f32>> {
        let values: Vec<_> = self
            .read_words_submitted(encoder, source, elements, submitted)
            .await?
            .into_iter()
            .map(f32::from_bits)
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite GPU result"
        );
        Ok(values)
    }
    async fn read_words_submitted(
        &self,
        mut encoder: wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        elements: usize,
        submitted: impl FnOnce(),
    ) -> Result<Vec<u32>> {
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
        self.flush(&mut encoder);
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        submitted();
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
            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
        };
        staging.unmap();
        staging.destroy();
        Ok(values)
    }
}
