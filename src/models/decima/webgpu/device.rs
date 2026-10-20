use anyhow::{Context, Result, ensure};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

pub(super) struct Device {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub align: usize,
}
// Every kernel binds `reads` read-only storage buffers, then `writes` writable ones, then one
// uniform slot addressed by a dynamic offset into the request's parameter ring.
pub(super) struct Kernel {
    pub pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    pub reads: usize,
    pub writes: usize,
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
        let mut limits = wgpu::Limits::default();
        limits.max_compute_workgroup_storage_size = adapter
            .limits()
            .max_compute_workgroup_storage_size
            .min(32768);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Decima inference"),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .context("request WebGPU device")?;
        let align = device.limits().min_uniform_buffer_offset_alignment as usize;
        Ok(Self {
            device,
            queue,
            info,
            align,
        })
    }
    pub fn kernel(&self, name: &str, source: String, reads: usize, writes: usize) -> Kernel {
        let entries: Vec<_> = (0..=reads + writes)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if i == reads + writes {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: i < reads,
                        }
                    },
                    has_dynamic_offset: i == reads + writes,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(name),
                entries: &entries,
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Kernel {
            pipeline,
            layout,
            reads,
            writes,
        }
    }
    pub fn buffer(&self, values: &[f32], label: &str) -> Result<wgpu::Buffer> {
        let bytes = bytemuck::cast_slice(values);
        self.check_size(bytes.len())?;
        Ok(self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }))
    }
    pub fn empty(
        &self,
        bytes: usize,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> Result<wgpu::Buffer> {
        self.check_size(bytes)?;
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.div_ceil(16) as u64 * 16,
            usage,
            mapped_at_creation: false,
        }))
    }
    pub fn storage(&self, elements: usize, label: &str) -> Result<wgpu::Buffer> {
        self.empty(
            elements
                .checked_mul(4)
                .context("GPU buffer size overflow")?,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            label,
        )
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
}

struct Dispatch {
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    offset: u32,
    groups: [u32; 3],
}
// Records one request's dispatches. Parameters go to a single uniform ring written once per
// submission, and bind groups over the persistent buffers are reused across requests.
pub(super) struct Frame {
    params: Vec<u8>,
    dispatches: Vec<Dispatch>,
}
pub(super) struct Bindings {
    ring: Option<wgpu::Buffer>,
    slots: usize,
    cache: HashMap<(wgpu::ComputePipeline, Vec<wgpu::Buffer>), wgpu::BindGroup>,
}
impl Bindings {
    pub fn new() -> Self {
        Self {
            ring: None,
            slots: 0,
            cache: HashMap::new(),
        }
    }
    // Grows the ring before recording; bind groups over the old ring are dropped.
    pub fn reserve(&mut self, device: &Device, slots: usize) {
        if slots <= self.slots {
            return;
        }
        let slots = slots.next_power_of_two().max(512);
        self.ring = Some(device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Decima parameter ring"),
            size: (slots * device.align) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.slots = slots;
        self.cache.clear();
    }
}
impl Frame {
    pub fn new() -> Self {
        Self {
            params: Vec::new(),
            dispatches: Vec::new(),
        }
    }
    pub fn dispatch(
        &mut self,
        device: &Device,
        bindings: &mut Bindings,
        kernel: &Kernel,
        buffers: &[&wgpu::Buffer],
        params: &[u32],
        groups: [u32; 3],
    ) -> Result<()> {
        ensure!(
            buffers.len() == kernel.reads + kernel.writes && params.len() <= 16,
            "GPU dispatch signature mismatch"
        );
        let limit = device.device.limits().max_compute_workgroups_per_dimension;
        ensure!(
            groups.iter().all(|&n| n > 0 && n <= limit),
            "GPU dispatch exceeds device limits"
        );
        let slot = self.params.len() / device.align;
        ensure!(slot < bindings.slots, "GPU parameter ring too small");
        let mut p = [0u32; 16];
        p[..params.len()].copy_from_slice(params);
        self.params.extend_from_slice(bytemuck::cast_slice(&p));
        self.params.resize((slot + 1) * device.align, 0);
        let bindings = Self::bind(device, bindings, kernel, buffers)?;
        self.dispatches.push(Dispatch {
            pipeline: kernel.pipeline.clone(),
            bindings,
            offset: (slot * device.align) as u32,
            groups,
        });
        Ok(())
    }
    fn bind(
        device: &Device,
        bindings: &mut Bindings,
        kernel: &Kernel,
        buffers: &[&wgpu::Buffer],
    ) -> Result<wgpu::BindGroup> {
        let pipeline = &kernel.pipeline;
        let key = (
            pipeline.clone(),
            buffers.iter().map(|b| (*b).clone()).collect::<Vec<_>>(),
        );
        if let Some(b) = bindings.cache.get(&key) {
            return Ok(b.clone());
        }
        let ring = bindings.ring.as_ref().context("GPU parameter ring")?;
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
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: ring,
                offset: 0,
                size: Some(wgpu::BufferSize::new(64).unwrap()),
            }),
        });
        let group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Decima dispatch"),
            layout: &kernel.layout,
            entries: &entries,
        });
        bindings.cache.insert(key, group.clone());
        Ok(group)
    }
    // Records every dispatch into one compute pass.
    pub fn encode(self, device: &Device, bindings: &Bindings, encoder: &mut wgpu::CommandEncoder) {
        if self.dispatches.is_empty() {
            return;
        }
        device
            .queue
            .write_buffer(bindings.ring.as_ref().unwrap(), 0, &self.params);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Decima request"),
            timestamp_writes: None,
        });
        for d in &self.dispatches {
            pass.set_pipeline(&d.pipeline);
            pass.set_bind_group(0, &d.bindings, &[d.offset]);
            pass.dispatch_workgroups(d.groups[0], d.groups[1], d.groups[2]);
        }
    }
}

pub(super) async fn read(
    device: &Device,
    mut encoder: wgpu::CommandEncoder,
    source: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    elements: usize,
) -> Result<Vec<f32>> {
    let bytes = elements.checked_mul(4).context("GPU read size overflow")? as u64;
    ensure!(
        bytes > 0 && bytes <= source.size() && bytes <= staging.size(),
        "GPU read outside source buffer"
    );
    encoder.copy_buffer_to_buffer(source, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    let (send, receive) = futures_channel::oneshot::channel();
    staging
        .slice(..bytes)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
    #[cfg(not(target_arch = "wasm32"))]
    device
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("GPU completion")?;
    receive.await.context("GPU readback cancelled")??;
    let values = {
        let view = staging.slice(..bytes).get_mapped_range();
        bytemuck::cast_slice::<u8, f32>(&view).to_vec()
    };
    staging.unmap();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "non-finite GPU result"
    );
    Ok(values)
}
