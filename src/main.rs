use anyhow::Result;
use clap::{Parser, Subcommand};
#[cfg(feature = "pplx-flash")]
#[path = "models/pplx_flash/cli.rs"]
mod pplx_flash;

mod capabilities;
#[cfg(feature = "decima")]
#[path = "models/decima/cli.rs"]
mod decima;
#[cfg(feature = "jev-style")]
#[path = "models/jev_style/cli.rs"]
mod jev_style;
#[cfg(feature = "kev")]
#[path = "models/kev/cli.rs"]
mod kev;
#[cfg(feature = "laya")]
#[path = "models/laya/cli.rs"]
mod laya;

#[derive(Parser)]
#[command(
    version,
    about = "A fast decision-model inference engine for browsers and edge devices"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    #[command(about = "Show compiled models, backends and precision support")]
    Capabilities(capabilities::Args),
    #[cfg(feature = "pplx-flash")]
    PplxFlash(pplx_flash::Args),
    #[cfg(feature = "laya")]
    Laya(laya::Args),
    #[cfg(feature = "decima")]
    Decima(decima::Args),
    #[cfg(feature = "jev-style")]
    JevStyle(jev_style::Args),
    #[cfg(feature = "kev")]
    Kev(kev::Args),
}
fn main() -> Result<()> {
    match Args::parse().command {
        Command::Capabilities(args) => capabilities::run(args),
        #[cfg(feature = "pplx-flash")]
        Command::PplxFlash(args) => pplx_flash::run(args),
        #[cfg(feature = "laya")]
        Command::Laya(args) => laya::run(args),
        #[cfg(feature = "decima")]
        Command::Decima(args) => decima::run(args),
        #[cfg(feature = "jev-style")]
        Command::JevStyle(args) => jev_style::run(args),
        #[cfg(feature = "kev")]
        Command::Kev(args) => kev::run(args),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    #[test]
    fn commands_match_compiled_models() {
        Args::command().debug_assert();
        let command = Args::command();
        for (name, enabled) in [
            ("pplx-flash", cfg!(feature = "pplx-flash")),
            ("laya", cfg!(feature = "laya")),
            ("decima", cfg!(feature = "decima")),
            ("jev-style", cfg!(feature = "jev-style")),
            ("kev", cfg!(feature = "kev")),
        ] {
            assert_eq!(command.find_subcommand(name).is_some(), enabled);
        }
        assert!(Args::try_parse_from(["decisive", "capabilities", "--json"]).is_ok());
        #[cfg(feature = "pplx-flash")]
        {
            assert!(
                Args::try_parse_from(["decisive", "pplx-flash", "--quantization", "bf16", "serve"])
                    .is_ok()
            );
            for precision in [
                "int8",
                "q8",
                "q4",
                "q4-e4",
                "q4-mse",
                "q4-e4-mse",
                "q4-e4-nf4",
                "q4-e4-affine",
                "q4-all",
            ] {
                assert_eq!(
                    Args::try_parse_from([
                        "decisive",
                        "pplx-flash",
                        "--quantization",
                        precision,
                        "serve"
                    ])
                    .is_ok(),
                    cfg!(feature = "int8")
                );
            }
        }
        #[cfg(feature = "decima")]
        {
            assert!(Args::try_parse_from(["decisive", "decima", "serve"]).is_ok());
            assert_eq!(
                Args::try_parse_from(["decisive", "decima", "--precision", "int8", "serve"])
                    .is_ok(),
                cfg!(feature = "int8")
            );
            assert_eq!(
                Args::try_parse_from([
                    "decisive",
                    "decima",
                    "pack-int8",
                    "--output",
                    "unused",
                    "--fp16-projections"
                ])
                .is_ok(),
                cfg!(all(feature = "int8", feature = "fp16"))
            );
        }
        #[cfg(feature = "laya")]
        {
            assert!(Args::try_parse_from(["decisive", "laya", "serve"]).is_ok());
            assert_eq!(
                Args::try_parse_from(["decisive", "laya", "--storage", "f16", "serve"]).is_ok(),
                cfg!(feature = "fp16")
            );
            assert_eq!(
                Args::try_parse_from(["decisive", "laya", "--quantization", "int8", "serve"])
                    .is_ok(),
                cfg!(feature = "int8")
            );
        }
    }
}
