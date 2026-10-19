//! Two disjoint banks of raw checkpoint weights. The uploader is a separate
//! Wasm object so it can run while BrowserGpu::run_layer holds &mut self.
use super::{Gpu, Tensor};
use crate::pplx_flash::weights::{Index, Range};
use anyhow::{Result, ensure};
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    rc::Rc,
};
use wasm_bindgen::prelude::*;

fn error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[derive(Default)]
struct Bank {
    layer: Option<usize>,
    ready: bool,
    tensors: HashMap<String, Tensor>,
    spare: Vec<wgpu::Buffer>,
}
impl Bank {
    fn assign(&mut self, layer: usize, ranges: &[Range], packed: bool) -> Result<()> {
        // Retain only buffers needed by this layer's shape. In particular, a
        // linear-attention/full-attention transition cannot grow a bank's pool.
        let mut needed = HashMap::<u64, usize>::new();
        for r in ranges {
            *needed.entry(weight_bytes(r, packed)?).or_default() += 1;
        }
        self.spare.retain(|b| {
            if let Some(n) = needed.get_mut(&b.size()) {
                if *n > 0 {
                    *n -= 1;
                    return true;
                }
            }
            b.destroy();
            false
        });
        self.layer = Some(layer);
        self.ready = false;
        Ok(())
    }
    fn retire(&mut self) {
        for b in self.spare.drain(..) {
            b.destroy();
        }
        self.spare
            .extend(self.tensors.drain().map(|(_, t)| t.buffer));
        self.layer = None;
        self.ready = false;
    }
    fn clear(&mut self) {
        self.retire();
        for b in self.spare.drain(..) {
            b.destroy();
        }
    }
    fn bytes(&self) -> u64 {
        self.tensors.values().map(|t| t.buffer.size()).sum::<u64>()
            + self.spare.iter().map(wgpu::Buffer::size).sum::<u64>()
    }
}
impl Drop for Bank {
    fn drop(&mut self) {
        self.clear();
    }
}

#[derive(Default)]
struct State {
    banks: [Bank; 2],
    active: bool,
    closed: bool,
    next: usize,
    computing: Option<usize>,
    in_flight: VecDeque<usize>,
    peak_in_flight: usize,
    submitted: bool,
    waiter: Option<futures_channel::oneshot::Sender<Result<(), String>>>,
    uploads: usize,
    reused: usize,
    peak_bytes: u64,
}
impl State {
    fn reject_waiter(&mut self) {
        if let Some(send) = self.waiter.take() {
            let _ = send.send(Err("layer submission cancelled".into()));
        }
    }
    fn clear(&mut self) {
        self.reject_waiter();
        for bank in &mut self.banks {
            bank.clear();
        }
        self.active = false;
        self.computing = None;
        self.in_flight.clear();
        self.submitted = false;
        self.next = 0;
    }
}

#[derive(Clone)]
#[wasm_bindgen(js_name=PplxFlashWeightBuffers)]
pub struct WeightBuffers {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub(super) index: Rc<Index>,
    state: Rc<RefCell<State>>,
    packed: bool,
}

fn weight_bytes(range: &Range, packed: bool) -> Result<u64> {
    if packed && range.shape.len() == 2 {
        Ok(crate::pplx_flash::packed_q4::size(range.shape[0], range.shape[1])? as u64)
    } else {
        Ok(range.end - range.start)
    }
}

#[wasm_bindgen(js_class=PplxFlashWeightBuffers)]
impl WeightBuffers {
    #[wasm_bindgen(js_name=stageTensor)]
    pub fn stage_tensor(&self, layer: usize, name: &str, bytes: &[u8]) -> Result<(), JsValue> {
        self.stage(layer, name, bytes).map_err(error)
    }

    #[wasm_bindgen(js_name=finishUpload)]
    pub fn finish_upload(&self, layer: usize) -> Result<(), JsValue> {
        let ranges = self.index.layer(layer).map_err(error)?;
        let mut s = self.state.borrow_mut();
        if s.closed || !s.active || s.computing == Some(layer) {
            return Err(error("bank is unavailable for upload"));
        }
        let bank = &mut s.banks[layer % 2];
        if bank.layer != Some(layer)
            || bank.ready
            || bank.tensors.len() != ranges.len()
            || ranges.iter().any(|r| !bank.tensors.contains_key(&r.name))
        {
            return Err(error("incomplete or already sealed weight bank"));
        }
        // wgpu's native write_buffer defers transfers until submit. On WebGPU
        // this also explicitly enqueues uploads before the next layer dispatch.
        self.queue.submit([]);
        bank.ready = true;
        Ok(())
    }

    #[wasm_bindgen(js_name=whenSubmitted)]
    pub fn when_submitted(&self, layer: usize) -> Result<js_sys::Promise, JsValue> {
        let mut s = self.state.borrow_mut();
        if s.closed || !s.active || s.next != layer || s.waiter.is_some() {
            return Err(error("invalid or duplicate layer submission wait"));
        }
        if s.submitted {
            return Ok(js_sys::Promise::resolve(&JsValue::UNDEFINED));
        }
        let (send, receive) = futures_channel::oneshot::channel();
        s.waiter = Some(send);
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            receive.await.map_err(error)?.map_err(error)?;
            Ok(JsValue::UNDEFINED)
        }))
    }

    pub fn reset(&self) -> Result<(), JsValue> {
        let mut s = self.state.borrow_mut();
        if s.computing.is_some() || !s.in_flight.is_empty() {
            return Err(error("cannot reset an in-flight weight bank"));
        }
        s.clear();
        s.uploads = 0;
        s.reused = 0;
        s.peak_bytes = 0;
        s.peak_in_flight = 0;
        Ok(())
    }

    pub fn diagnostics(&self) -> String {
        let s = self.state.borrow();
        serde_json::json!({"banks":2,"raw_weight_bytes":s.banks.iter().map(Bank::bytes).sum::<u64>(),
            "peak_raw_weight_bytes":s.peak_bytes,"tensor_uploads":s.uploads,"reused_buffers":s.reused,
            "bank_layers":s.banks.iter().map(|b| b.layer).collect::<Vec<_>>(),
            "computing_layer":s.computing,"packed_q4":self.packed,"weight_bytes":s.banks.iter().map(Bank::bytes).sum::<u64>(),
            "peak_weight_bytes":s.peak_bytes,"scope":if self.packed {"prepacked Q4 group-64 weight banks; excludes activations"} else {"raw BF16 weight banks; excludes active converted/quantized weights and activations"},
            "in_flight_layers":s.in_flight,"peak_in_flight_layers":s.peak_in_flight,
            "overlap":"uploads enqueued after compute submission and before its completion wait; hardware overlap is implementation-dependent"}).to_string()
    }
}

impl WeightBuffers {
    fn stage(&self, layer: usize, name: &str, bytes: &[u8]) -> Result<()> {
        let range = self.index.tensor(name)?;
        ensure!(
            layer < 32 && name.starts_with(&format!("language_model.layers.{layer}.")),
            "tensor belongs to another layer"
        );
        ensure!(
            bytes.len() as u64 == weight_bytes(&range, self.packed)? && bytes.len() % 4 == 0,
            "invalid tensor byte length"
        );
        let limits = self.device.limits();
        ensure!(
            !bytes.is_empty()
                && bytes.len() as u64 <= limits.max_buffer_size
                && bytes.len() as u64 <= limits.max_storage_buffer_binding_size as u64,
            "weight exceeds GPU buffer limits"
        );
        let mut s = self.state.borrow_mut();
        ensure!(!s.closed && s.active, "no active buffered chunk");
        ensure!(
            layer == s.next || (s.computing == Some(s.next) && layer == s.next + 1),
            "only the current and next layer may be uploaded"
        );
        ensure!(
            s.computing != Some(layer),
            "cannot overwrite an in-flight weight bank"
        );
        let bank = &mut s.banks[layer % 2];
        if bank.layer.is_none() {
            bank.assign(layer, &self.index.layer(layer)?, self.packed)?;
        }
        ensure!(
            bank.layer == Some(layer) && !bank.ready,
            "weight bank is occupied or sealed"
        );
        ensure!(!bank.tensors.contains_key(name), "duplicate tensor upload");
        let reused = bank
            .spare
            .iter()
            .position(|b| b.size() == bytes.len() as u64);
        let buffer = match reused {
            Some(i) => bank.spare.swap_remove(i),
            None => self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(if layer % 2 == 0 {
                    "Pplx weight bank A"
                } else {
                    "Pplx weight bank B"
                }),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        };
        self.queue.write_buffer(&buffer, 0, bytes);
        bank.tensors.insert(
            name.into(),
            Tensor {
                buffer,
                shape: range.shape,
            },
        );
        s.uploads += 1;
        s.reused += usize::from(reused.is_some());
        s.peak_bytes = s.peak_bytes.max(s.banks.iter().map(Bank::bytes).sum());
        Ok(())
    }
    pub(crate) fn begin_chunk(&self) -> Result<()> {
        let mut s = self.state.borrow_mut();
        ensure!(
            !s.closed && !s.active && s.computing.is_none() && s.in_flight.is_empty(),
            "buffered chunk is already active"
        );
        s.active = true;
        s.next = 0;
        s.submitted = false;
        Ok(())
    }
    pub(in crate::pplx_flash) fn start_layer(
        &self,
        layer: usize,
    ) -> Result<HashMap<String, Tensor>> {
        let mut s = self.state.borrow_mut();
        ensure!(
            s.active
                && !s.closed
                && s.next == layer
                && s.computing.is_none()
                && s.in_flight.len() < 2,
            "invalid buffered layer order"
        );
        let bank = &s.banks[layer % 2];
        ensure!(
            bank.layer == Some(layer) && bank.ready,
            "layer weights have not finished uploading"
        );
        let tensors = bank.tensors.clone();
        s.computing = Some(layer);
        Ok(tensors)
    }
    pub(crate) fn submitted(&self) {
        let mut s = self.state.borrow_mut();
        s.submitted = true;
        if let Some(send) = s.waiter.take() {
            let _ = send.send(Ok(()));
        }
    }
    pub(crate) fn finish_layer(&self, layer: usize) {
        let mut s = self.state.borrow_mut();
        assert_eq!(s.computing, Some(layer));
        s.banks[layer % 2].retire();
        s.computing = None;
        s.submitted = false;
        s.next += 1;
    }
    pub(crate) fn queued(&self, layer: usize) {
        let mut s = self.state.borrow_mut();
        assert_eq!(s.computing, Some(layer));
        s.computing = None;
        s.in_flight.push_back(layer);
        s.peak_in_flight = s.peak_in_flight.max(s.in_flight.len());
        s.next += 1;
        if let Some(send) = s.waiter.take() {
            let _ = send.send(Ok(()));
        }
    }
    pub(crate) fn collected(&self, layer: usize) {
        let mut s = self.state.borrow_mut();
        assert_eq!(s.in_flight.pop_front(), Some(layer));
        s.banks[layer % 2].retire();
    }
    pub(crate) fn end_chunk(&self) -> Result<()> {
        let mut s = self.state.borrow_mut();
        ensure!(
            s.active && s.next == 32 && s.computing.is_none() && s.in_flight.is_empty(),
            "unfinished buffered chunk"
        );
        s.active = false;
        Ok(())
    }
    pub(crate) fn owns(&self, buffer: &wgpu::Buffer) -> bool {
        self.state
            .borrow()
            .banks
            .iter()
            .any(|b| b.tensors.values().any(|t| &t.buffer == buffer))
    }
    pub(crate) fn fail(&self) {
        self.state.borrow_mut().clear();
    }
    pub(crate) fn close(&self) {
        let mut s = self.state.borrow_mut();
        s.clear();
        s.closed = true;
    }
}

impl Gpu {
    pub(crate) fn weight_uploader(&self, index: Rc<Index>) -> WeightBuffers {
        WeightBuffers {
            device: self.device.device.clone(),
            queue: self.device.queue.clone(),
            index,
            state: Rc::new(RefCell::new(State::default())),
            packed: self.grouped_q4(),
        }
    }
    pub(crate) fn prepare_uploaded(&self, tensor: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "kernel-checks")]
        if self.fp16_projections && tensor.shape.len() == 2 && !self.grouped_q4() {
            return Ok(Tensor {
                shape: tensor.shape.clone(),
                buffer: self.checkpoint_fp16(&tensor.buffer, tensor.buffer.size() as usize)?,
            });
        }
        Ok(tensor.clone())
    }
}
