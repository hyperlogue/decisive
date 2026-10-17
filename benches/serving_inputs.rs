use anyhow::{Result, ensure};
use clap::Parser;
use decisive::laya::input::Input;
use serde_json::json;
use std::{fs, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models")]
    models: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "128,512,1024,2048")]
    lengths: Vec<usize>,
}
fn main() -> Result<()> {
    let args = Args::parse_from(std::env::args().filter(|arg| arg != "--bench"));
    ensure!(!args.out.exists(), "output already exists");
    let languages = [
        (
            "en",
            "I was charged twice and want a refund.",
            "The archive contains older account notes. ",
            "note ",
        ),
        (
            "es",
            "Me cobraron dos veces y quiero un reembolso.",
            "El archivo contiene notas antiguas de la cuenta. ",
            "nota ",
        ),
        (
            "de",
            "Ich wurde zweimal belastet und möchte eine Rückerstattung.",
            "Das Archiv enthält ältere Notizen zum Konto. ",
            "alt ",
        ),
        (
            "zh",
            "我被重复收费了，希望退款。",
            "档案中保存着以前的账户记录。",
            "的 ",
        ),
        (
            "ja",
            "二重に請求されたので、返金してほしいです。",
            "アーカイブには以前のアカウント記録が保存されています。",
            "の ",
        ),
        (
            "ar",
            "تم تحصيل المبلغ مني مرتين وأريد استرداد أموالي.",
            "يحتوي الأرشيف على ملاحظات قديمة عن الحساب. ",
            "و ",
        ),
        (
            "hi",
            "मुझसे दो बार पैसे लिए गए और मुझे धनवापसी चाहिए।",
            "संग्रह में खाते के पुराने विवरण रखे गए हैं। ",
            "है ",
        ),
    ];
    let questions = json!({"intent":{"type":"choice","instructions":"What does the customer need?","criteria":{"refund":null,"technical support":null,"new purchase":null}}});
    let technical = [
        "I cannot sign in after resetting my password and need technical support.",
        "No puedo iniciar sesión después de restablecer mi contraseña y necesito ayuda técnica.",
        "Nach dem Zurücksetzen meines Passworts kann ich mich nicht anmelden und brauche technische Hilfe.",
        "重置密码后我无法登录，需要技术支持。",
        "パスワードをリセットした後にログインできなくなったので、技術サポートが必要です。",
        "لا أستطيع تسجيل الدخول بعد إعادة تعيين كلمة المرور وأحتاج إلى دعم فني.",
        "पासवर्ड रीसेट करने के बाद मैं लॉग इन नहीं कर पा रहा हूँ और मुझे तकनीकी सहायता चाहिए।",
    ];
    let mut output = fs::File::create_new(&args.out)?;
    for (checkpoint, directory, head_limit) in [
        ("english", "laya", 192),
        ("multilingual", "laya-multilingual", 256),
    ] {
        let model = args.models.join(directory);
        let input = Input::new(
            &fs::read(model.join("tokenizer/tokenizer.json"))?,
            &fs::read(model.join("tokenizer/tokenizer_config.json"))?,
        )?;
        for (language_index, &(language, refund, filler, pad)) in languages.iter().enumerate() {
            if checkpoint == "english" && language != "en" {
                continue;
            }
            for &tokens in &args.lengths {
                for (scenario, relevant) in [
                    ("refund", refund),
                    ("technical support", technical[language_index]),
                ] {
                    ensure!((64..=8192).contains(&tokens), "invalid length");
                    let count = |state: &str| -> Result<usize> {
                        Ok(input.encode(
                            &json!(state),
                            questions.as_object().unwrap(),
                            16384,
                            head_limit,
                        )?[0]
                            .ids
                            .len())
                    };
                    let mut state = relevant.to_owned();
                    ensure!(count(&state)? <= tokens, "base request exceeds target");
                    loop {
                        let next = format!("{filler}{state}");
                        if count(&next)? > tokens {
                            break;
                        }
                        state = next;
                    }
                    let mut padding = Vec::new();
                    while count(&state)? < tokens {
                        let previous = count(&state)?;
                        let mut advanced = false;
                        for candidate in [pad, ". ", "! ", "a ", "1 "] {
                            let next = format!("{candidate}{state}");
                            let length = count(&next)?;
                            if length > previous && length <= tokens {
                                state = next;
                                padding.push(candidate);
                                advanced = true;
                                break;
                            }
                        }
                        ensure!(advanced, "cannot fill {checkpoint}/{language}/{tokens}");
                    }
                    let encoded = input.encode(
                        &json!(state),
                        questions.as_object().unwrap(),
                        tokens,
                        head_limit,
                    )?;
                    ensure!(
                        encoded[0].ids.len() == tokens
                            && encoded[0].state_tokens == encoded[0].state_tokens_used,
                        "length/truncation mismatch"
                    );
                    writeln!(
                        output,
                        "{}",
                        json!({"checkpoint":checkpoint,"language":language,"tokens":tokens,"scenario":scenario,"scope":"Synthetic length-controlled translated state; shared English question and labels. Not a representative language-quality corpus.","padding_fragments":padding,"request":{"state":state,"questions":questions,"max_len":tokens,"head_max_len":head_limit}})
                    )?;
                }
            }
        }
    }
    Ok(())
}
