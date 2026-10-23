use super::{cpu, int8::Matrix};

#[derive(Default)]
pub(super) struct Scratch {
    expanded: Vec<f32>,
    projected: Vec<f32>,
    packed: Vec<f32>,
}

pub(super) fn apply(
    hidden: &mut [f32],
    first: &Matrix,
    first_bias: &[f32],
    second: &Matrix,
    second_bias: &[f32],
    norm: &cpu::Norm,
    tile: usize,
    scratch: &mut Scratch,
) {
    let d = first.cols;
    #[cfg(feature = "decima-wasm-threads")]
    let count = hidden.len() / d;
    let run = |hidden: &mut [f32], scratch: &mut Scratch| {
        first.apply_serial_wasm(
            hidden,
            &mut scratch.expanded,
            first_bias,
            &mut scratch.packed,
            tile,
        );
        super::activation_wasm::apply(&mut scratch.expanded);
        second.apply_serial_wasm(
            &scratch.expanded,
            &mut scratch.projected,
            second_bias,
            &mut scratch.packed,
            tile,
        );
        cpu::add(hidden, &scratch.projected);
        norm.in_place(hidden);
    };
    #[cfg(feature = "decima-wasm-threads")]
    {
        use rayon::prelude::*;
        use std::cell::RefCell;
        thread_local! {
            static FFN: RefCell<Scratch> = RefCell::new(Scratch::default());
        }
        let workers = super::wasm_threads::count();
        if workers > 1 {
            let job = count.div_ceil(workers).div_ceil(tile) * tile;
            let block = 64 / tile * tile;
            hidden.par_chunks_mut(job * d).for_each(|hidden| {
                FFN.with(|scratch| {
                    let mut scratch = scratch.borrow_mut();
                    for block in hidden.chunks_mut(block * d) {
                        run(block, &mut scratch);
                    }
                });
            });
            return;
        }
    }
    let block = 64 / tile * tile;
    for hidden in hidden.chunks_mut(block * d) {
        run(hidden, scratch);
    }
}
