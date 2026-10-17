fn main() -> anyhow::Result<()> {
    use decisive::laya::{Scratch, cpu::Kernel, weights::Storage};
    let directory = std::env::args()
        .filter(|arg| arg != "--bench")
        .nth(1)
        .unwrap_or_else(|| "models/laya".into());
    if directory == "decima" {
        let cpu = decisive::decima::Model::load(
            "models/decima-small",
            decisive::decima::Options::default(),
        )?;
        let request = serde_json::json!({"state":"The customer was charged twice.","questions":{"q":{"type":"choice","instructions":"Choose the correct department.","criteria":["billing","technical support","shipping"]}}});
        let expected = cpu.predict_json(&request, &mut decisive::decima::Session::default())?;
        let gpu = pollster::block_on(decisive::decima::webgpu::Model::from_cpu(cpu))?;
        eprintln!("adapter: {:?}", gpu.adapter_info());
        let mut session = decisive::decima::webgpu::Session::default();
        let actual = pollster::block_on(gpu.predict_json(&request, &mut session))?;
        let cached = pollster::block_on(gpu.predict_json(&request, &mut session))?;
        anyhow::ensure!(actual == cached, "GPU choice cache changed outputs");
        println!(
            "{}",
            serde_json::json!({"reference":expected,"output":actual,"cache_hits":session.choice_cache_hits})
        );
        return Ok(());
    }
    let storage = if cfg!(feature = "fp16") {
        Storage::F16
    } else {
        Storage::F32
    };
    let cpu = decisive::laya::Model::load_with_kernel(directory, storage, Kernel::detect())?;
    let request = serde_json::json!({"state":"The customer was charged twice.","questions":{"q":{"type":"choice","instructions":"Choose the correct department.","criteria":["billing","technical support","shipping"]}}});
    let expected = cpu.predict_json(request.clone(), &mut Scratch::default())?;
    let gpu = pollster::block_on(decisive::laya::webgpu::Model::from_cpu(cpu))?;
    eprintln!("adapter: {:?}", gpu.adapter_info());
    let actual = pollster::block_on(gpu.predict_json(request))?;
    println!(
        "{}",
        serde_json::json!({"reference":expected,"output":actual})
    );
    Ok(())
}
