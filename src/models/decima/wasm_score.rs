use super::{D, Kernel, Model, Scratch, cpu, specialization};
use rayon::prelude::*;

pub(super) struct Choice {
    pub scratch: Scratch,
    pub plan: specialization::CpuPlan,
    pub similarity: f32,
    pub logit: f32,
    pub mean: Vec<f32>,
}
impl Default for Choice {
    fn default() -> Self {
        Self {
            scratch: Scratch::default(),
            plan: specialization::default_plan(Kernel::WasmSimd),
            similarity: 0.,
            logit: 0.,
            mean: vec![],
        }
    }
}

pub(super) fn prepare(
    model: &Model,
    choices: &mut [Choice],
    hidden: &[std::sync::Arc<[f32]>],
    plans: &[specialization::CpuPlan],
) {
    assert_eq!(choices.len(), hidden.len());
    assert_eq!(choices.len(), plans.len());
    choices
        .par_iter_mut()
        .zip(hidden)
        .zip(plans)
        .for_each(|((choice, hidden), &plan)| {
            choice.plan = plan;
            choice.scratch.hidden.clear();
            choice.scratch.hidden.extend_from_slice(hidden);
            model.scorer[0].prepare_query(&mut choice.scratch, plan, model.avx2);
        });
}

pub(super) fn finish(
    model: &Model,
    choices: &mut [Choice],
    memory: &[(Vec<f32>, Vec<f32>)],
    attention_tile: usize,
) {
    choices.par_iter_mut().for_each(|choice| {
        let x = &mut choice.scratch;
        let plan = choice.plan;
        for (index, (layer, (mk, mv))) in model.scorer.iter().zip(memory).enumerate() {
            if index > 0 {
                layer.prepare_query(x, plan, model.avx2);
            }
            cpu::attention_with_tile(
                &x.q,
                mk,
                mv,
                D,
                6,
                &mut x.ctx,
                &mut x.scores,
                &mut x.heads,
                attention_tile,
                model.avx2,
            );
            layer
                .cross_attn
                .o
                .apply_with_plan(&x.ctx, &mut x.tmp, &mut x.work, plan);
            cpu::add(&mut x.hidden, &x.tmp);
            layer.n3.apply(&x.hidden, &mut x.norm);
            layer
                .ff1
                .apply_with_plan(&x.norm, &mut x.ff, &mut x.work, plan);
            cpu::gelu(&mut x.ff);
            layer
                .ff2
                .apply_with_plan(&x.ff, &mut x.tmp, &mut x.work, plan);
            cpu::add(&mut x.hidden, &x.tmp);
        }
        model.out_norm.in_place(&mut x.hidden);
        choice.mean = cpu::mean(&x.hidden, D);
        model.score.apply(&choice.mean, &mut x.tmp, &mut x.work);
        choice.logit = (x.tmp[0] + model.sim_scale * (choice.similarity - model.sim_center))
            / model.config.temperature;
    });
}
