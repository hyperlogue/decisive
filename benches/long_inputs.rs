use anyhow::{Result, ensure};
use decisive::laya::input::Input;
use serde_json::{Value, json};

fn main() -> Result<()> {
    let followups = std::env::args().skip(1).any(|s| s == "--followups");
    let input = Input::new(
        &std::fs::read("models/laya/tokenizer/tokenizer.json")?,
        &std::fs::read("models/laya/tokenizer/tokenizer_config.json")?,
    )?;
    let smoke: Value = serde_json::from_str(include_str!("../benches/smoke.jsonl"))?;
    let questions = json!({"intent":smoke["questions"]["intent"]});
    let filler = "The archive contains older account notes. ";
    for length in [512, 1024, 2048] {
        let mut state = smoke["state"].as_str().unwrap().to_owned();
        let count = |state: &str| -> Result<usize> {
            Ok(
                input.encode(&json!(state), questions.as_object().unwrap(), 8192, 192)?[0]
                    .ids
                    .len(),
            )
        };
        loop {
            let next = format!("{filler}{state}");
            if count(&next)? > length {
                break;
            }
            state = next;
        }
        while count(&state)? < length {
            state = format!("account {state}");
        }
        ensure!(count(&state)? == length, "filler must reach exact length");
        let encoded = input.encode(&json!(state), questions.as_object().unwrap(), length, 192)?;
        ensure!(
            encoded[0].state_tokens == encoded[0].state_tokens_used,
            "unexpected truncation"
        );
        if !followups {
            println!(
                "{}",
                json!({"state":state,"questions":questions,"max_len":length})
            );
        } else {
            for question in ["intent", "urgent", "priority"] {
                let mut q = serde_json::Map::new();
                q.insert(question.into(), smoke["questions"][question].clone());
                println!(
                    "{}",
                    json!({"state":state,"questions":q,"max_len":length+128})
                );
            }
            let mut reordered = questions.clone();
            reordered["intent"]["option_order"] = json!([2, 0, 1]);
            println!(
                "{}",
                json!({"state":state,"questions":reordered,"max_len":length+128})
            );
            let updated = format!(
                "{state} The customer has also reported being unable to sign in after a password reset."
            );
            println!(
                "{}",
                json!({"state":updated,"questions":questions,"max_len":length+128})
            );
            println!(
                "{}",
                json!({"state":state,"questions":questions,"max_len":length+128})
            );
        }
    }
    Ok(())
}
