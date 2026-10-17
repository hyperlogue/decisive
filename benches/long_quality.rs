use anyhow::{Result, ensure};
use decisive::laya::input::Input;
use serde_json::json;
fn main() -> Result<()> {
    let input = Input::new(
        &std::fs::read("models/laya/tokenizer/tokenizer.json")?,
        &std::fs::read("models/laya/tokenizer/tokenizer_config.json")?,
    )?;
    let questions = json!({"action":{"type":"choice","instructions":"For CURRENT REQUEST only, apply this rule: denied approval means stop; otherwise an amount greater than the budget or missing approval means review; otherwise proceed. Ignore all archived requests.","criteria":{"proceed":"Proceed with the current request.","review":"Send the current request for review.","stop":"Stop the current request."}}});
    let cases = [
        (
            "approved_within_budget",
            "Approval is granted. The requested amount is 40 units and the budget is 90 units.",
            "proceed",
        ),
        (
            "over_budget",
            "Approval is granted. The requested amount is 120 units and the budget is 90 units.",
            "review",
        ),
        (
            "missing_approval",
            "Approval is missing. The requested amount is 40 units and the budget is 90 units.",
            "review",
        ),
        (
            "denied_approval",
            "Approval is denied. The requested amount is 40 units and the budget is 90 units.",
            "stop",
        ),
    ];
    let archives = [
        "An older request concerned replacing a desk lamp; its invoice was filed in the blue folder.",
        "A previous courier delivery used the east entrance, with the parcel collected before lunch.",
        "Last quarter's project notes list a completed inventory count and an updated storage map.",
        "The former office schedule reserved Tuesday mornings for checking the shared equipment.",
        "An archived order for paper was approved, shipped in two boxes, and recorded as complete.",
        "An unrelated historic request was denied after its proposed amount exceeded that team's budget.",
        "The museum archive indexed a photograph of a stone bridge taken during a winter survey.",
        "A previous maintenance visit replaced a worn cable and entered the serial number in a register.",
        "The library record describes a revised catalog card, a shelf location and a returned reference book.",
        "An old training session covered document retention, meeting notes and the visitor sign-in procedure.",
        "A historical travel request required review because approval was missing when it was submitted.",
        "The warehouse summary records reusable crates, a clean aisle and a scheduled inspection of the lights.",
    ];
    for length in [512, 1024, 2048] {
        for (case, record, label) in cases {
            for placement in ["start", "middle", "end"] {
                let current = format!("CURRENT REQUEST. {record} END CURRENT REQUEST.");
                let state = |pieces: &[String], padding: usize| {
                    let split = match placement {
                        "start" => 0,
                        "middle" => pieces.len() / 2,
                        _ => pieces.len(),
                    };
                    let before = pieces[..split].join(" ");
                    let after = pieces[split..].join(" ");
                    let pad = " account".repeat(padding);
                    if placement == "end" {
                        format!("ARCHIVED RECORDS. {before}{pad} {current} {after}")
                    } else {
                        format!("{before} {current} ARCHIVED RECORDS. {after}{pad}")
                    }
                };
                let count = |text: &str| -> Result<usize> {
                    Ok(
                        input.encode(&json!(text), questions.as_object().unwrap(), 8192, 192)?[0]
                            .ids
                            .len(),
                    )
                };
                let mut pieces = Vec::new();
                loop {
                    let mut next = pieces.clone();
                    next.push(archives[pieces.len() % archives.len()].to_owned());
                    if count(&state(&next, 0))? > length {
                        break;
                    }
                    pieces = next;
                }
                let mut padding = 0;
                while count(&state(&pieces, padding))? < length {
                    padding += 1;
                    ensure!(padding < 128, "padding failed");
                }
                let state = state(&pieces, padding);
                ensure!(count(&state)? == length, "exact target length");
                let encoded =
                    input.encode(&json!(state), questions.as_object().unwrap(), length, 192)?;
                ensure!(
                    encoded[0].state_tokens == encoded[0].state_tokens_used,
                    "no truncation"
                );
                println!(
                    "{}",
                    json!({"id":format!("long-rule-{length}-{case}-{placement}"),"state":state,"questions":questions,"max_len":length,"gold":{"action":{"label":label}},"metadata":{"target_tokens":length,"evidence_placement":placement,"label_source":"Deterministic stated approval/budget rule; synthetic evaluation, not human production data."}})
                );
            }
        }
    }
    Ok(())
}
