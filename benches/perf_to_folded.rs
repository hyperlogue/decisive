use std::{collections::BTreeMap, fs};
fn symbol(line: &str) -> Option<String> {
    let line = line.trim();
    let (_, rest) = line.split_once(char::is_whitespace)?;
    let (name, _) = rest.trim().rsplit_once(" (")?;
    let name = name.rsplit_once("+0x").map_or(name, |(n, _)| n);
    Some(name.replace(';', ":"))
}
fn compact(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0usize;
    for c in s.chars() {
        match c {
            '<' => {
                if depth == 0 {
                    out.push('<');
                }
                depth += 1;
            }
            '>' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    out.push('>');
                }
            }
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    if out.starts_with("<>") {
        s.to_owned()
    } else {
        out
    }
}
fn main() {
    let args: Vec<_> = std::env::args().filter(|arg| arg != "--bench").collect();
    assert_eq!(args.len(), 4, "LEAVES STACKS OUTPUT_PREFIX");
    let leaves = fs::read_to_string(&args[1]).unwrap();
    let stacks = fs::read_to_string(&args[2]).unwrap();
    let leaves: Vec<_> = leaves
        .lines()
        .filter_map(|l| l.split_once("cpu-clock:u:").and_then(|(_, s)| symbol(s)))
        .collect();
    let blocks: Vec<_> = stacks
        .trim()
        .split("\n\n")
        .filter(|s| !s.trim().is_empty())
        .collect();
    assert_eq!(leaves.len(), blocks.len(), "sample alignment");
    let mut folded = BTreeMap::<String, u64>::new();
    let mut flat = BTreeMap::<String, u64>::new();
    let mut unavailable = 0;
    let mut max_depth = 0;
    for (leaf, block) in leaves.iter().zip(blocks) {
        *flat.entry(leaf.clone()).or_default() += 1;
        let mut frames: Vec<_> = block.lines().skip(1).filter_map(symbol).collect();
        if frames.is_empty() {
            unavailable += 1;
            frames = vec![leaf.clone(), "[caller stack unavailable]".into()];
        } else {
            assert_eq!(&frames[0], leaf, "sample leaf mismatch");
        }
        max_depth = max_depth.max(frames.len());
        frames.reverse();
        let key = frames
            .iter()
            .map(|s| compact(s))
            .collect::<Vec<_>>()
            .join(";");
        *folded.entry(key).or_default() += 1;
    }
    fs::write(
        format!("{}.folded", args[3]),
        folded
            .iter()
            .map(|(k, v)| format!("{k} {v}\n"))
            .collect::<String>(),
    )
    .unwrap();
    let mut rows: Vec<_> = flat.into_iter().collect();
    rows.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    fs::write(
        format!("{}-self.tsv", args[3]),
        format!(
            "samples\tpercent\tsymbol\n{}",
            rows.iter()
                .map(|(s, n)| format!("{n}\t{:.5}\t{s}\n", *n as f64 * 100. / leaves.len() as f64))
                .collect::<String>()
        ),
    )
    .unwrap();
    println!(
        "{{\"samples\":{},\"unavailable_caller_stacks\":{unavailable},\"maximum_resolved_depth\":{max_depth},\"scope\":\"Software CPU-clock samples; folded stacks omit generic arguments where possible. Missing caller chains retain the sampled leaf under an explicit placeholder. No instruction-address, PID, DSO-path or hostname data is published.\"}}",
        leaves.len()
    );
}
