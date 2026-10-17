use anyhow::{Context, Result, ensure};
use clap::Parser;
use decisive::decima::{Model, Options, Question, Session};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models/decima-small")]
    model: PathBuf,
    #[arg(help = "Official XNLI-1.0.zip; download URL and checksum are in the protocol")]
    #[arg(long)]
    archive: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value="test", value_parser=["dev","test"])]
    split: String,
    #[arg(long, default_value_t = 100)]
    pairs: usize,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "en,fr,es,de,el,bg,ru,tr,ar,vi,th,zh,hi,sw,ur"
    )]
    languages: Vec<String>,
}
fn main() -> Result<()> {
    let args = Args::parse_from(std::env::args().filter(|arg| arg != "--bench"));
    ensure!(
        args.pairs > 0 && !args.languages.is_empty(),
        "positive pair count and languages required"
    );
    ensure!(!args.output.exists(), "choose a new output path");
    let source_sha256 = format!("{:x}", Sha256::digest(fs::read(&args.archive)?));
    ensure!(
        source_sha256 == "4ba1d5e1afdb7161f0f23c66dc787802ccfa8a25a3ddd3b165a35e50df346ab1",
        "XNLI archive checksum mismatch"
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let mut archive = zip::ZipArchive::new(fs::File::open(&args.archive)?)?;
    let file = archive.by_name(&format!("XNLI-1.0/xnli.{}.jsonl", args.split))?;
    let records: Vec<Value> = BufReader::new(file)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect::<Result<_>>()?;
    let ids: Vec<_> = records
        .iter()
        .map(|r| r["pairID"].as_str().context("pairID"))
        .collect::<Result<BTreeSet<_>>>()?
        .into_iter()
        .collect();
    ensure!(
        args.pairs <= ids.len(),
        "more pairs requested than available"
    );
    let selected: BTreeSet<_> = (0..args.pairs)
        .map(|i| ids[(i * ids.len() + ids.len() / 2) / args.pairs])
        .collect();
    let model = Model::load(&args.model, Options::default())?;
    let mut session = Session::default();
    let mut output = fs::File::create_new(&args.output)?;
    let mut accepted = 0;
    let mut skipped = Vec::new();
    for r in &records {
        let pair = r["pairID"].as_str().context("pairID")?;
        let lang = r["language"].as_str().context("language")?;
        if !selected.contains(pair) || !args.languages.iter().any(|l| l == lang) {
            continue;
        }
        let state = r["sentence1"].as_str().context("premise")?;
        let hypothesis = r["sentence2"].as_str().context("hypothesis")?;
        let question = Question {
            text: format!(
                "Given the premise, how is this hypothesis related to it? Hypothesis: {hypothesis}"
            ),
            choices: vec![
                "The hypothesis follows from the premise.".into(),
                "The premise does not determine whether the hypothesis is true.".into(),
                "The hypothesis contradicts the premise.".into(),
            ],
            lang: lang.into(),
            kind: decisive::decima::Kind::Choice,
        };
        let before = session.stats.state_tokens;
        let expected = match model.decide(state, &question, &mut session) {
            Ok(p) => p,
            Err(e) if e.to_string().starts_with("STATE_TRUNCATED:") => {
                skipped.push(json!({"pair":pair,"language":lang,"reason":e.to_string()}));
                continue;
            }
            Err(e) => return Err(e),
        };
        let gold = match r["gold_label"].as_str() {
            Some("entailment") => 0,
            Some("neutral") => 1,
            Some("contradiction") => 2,
            _ => anyhow::bail!("unknown XNLI label"),
        };
        writeln!(
            output,
            "{}",
            json!({"id":format!("xnli-{}-{lang}-{pair}",args.split),"pair_id":pair,"language":lang,"state":state,"question":question,"gold":gold,"expected":expected,"state_tokens":session.stats.state_tokens-before})
        )?;
        accepted += 1;
    }
    let manifest = json!({"source":"https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip","source_sha256":"4ba1d5e1afdb7161f0f23c66dc787802ccfa8a25a3ddd3b165a35e50df346ab1","split":args.split,"pairs":args.pairs,"languages":args.languages,"accepted":accepted,"skipped":skipped,"selection":"Sorted unique pair IDs, midpoint of each equal-width index interval; same pairs across languages.","reference":"Decisive native FP32; implementation fidelity audited separately against SDK fixtures.","checkpoint_revision":decisive::decima::REVISION,"scope":"Quantization comparison, not a model leaderboard. Translations are correlated. Decima trained on XNLI/MNLI training data."});
    fs::write(
        args.output.with_extension("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    eprintln!("wrote {accepted} labeled decisions");
    Ok(())
}
