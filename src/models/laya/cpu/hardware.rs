use serde::Serialize;

pub fn available_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

#[derive(Serialize)]
pub struct Hardware {
    pub architecture: &'static str,
    pub available_threads: usize,
    pub packed_avx2_supported: bool,
    pub features: Vec<&'static str>,
    pub visible_cache_profiles: Vec<Cache>,
    pub cache_source: &'static str,
}

#[derive(Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cache {
    level: usize,
    kind: String,
    bytes: usize,
    line_bytes: usize,
}

pub fn detect() -> Hardware {
    let mut features = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        macro_rules! feature {
            ($($f:tt),*) => { $(if std::arch::is_x86_feature_detected!($f) { features.push($f); })* };
        }
        feature!(
            "sse2",
            "avx",
            "avx2",
            "fma",
            "f16c",
            "avx512f",
            "avx512bw",
            "avx512vl",
            "avx512bf16",
            "avx512fp16"
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            features.push("neon");
        }
        if std::arch::is_aarch64_feature_detected!("fp16") {
            features.push("fp16");
        }
    }
    Hardware {
        architecture: std::env::consts::ARCH,
        available_threads: available_threads(),
        packed_avx2_supported: super::packed_supported(),
        features,
        visible_cache_profiles: caches(),
        cache_source: if cfg!(target_os = "linux") {
            "Linux sysfs: distinct profiles from visible CPUs; not an affinity or sharing map; VM values may be virtualized"
        } else {
            "unavailable; no inferred cache size"
        },
    }
}

fn caches() -> Vec<Cache> {
    let mut result = Vec::new();
    #[cfg(target_os = "linux")]
    if let Ok(cpus) = std::fs::read_dir("/sys/devices/system/cpu") {
        for cpu in cpus.flatten() {
            let name = cpu.file_name();
            let name = name.to_string_lossy();
            if !name
                .strip_prefix("cpu")
                .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            {
                continue;
            }
            let Ok(indices) = std::fs::read_dir(cpu.path().join("cache")) else {
                continue;
            };
            for index in indices.flatten() {
                let read = |name| {
                    std::fs::read_to_string(index.path().join(name))
                        .ok()
                        .map(|s| s.trim().to_owned())
                };
                let entry = (|| {
                    Some(Cache {
                        level: read("level")?.parse().ok()?,
                        kind: read("type")?,
                        bytes: parse_size(&read("size")?)?,
                        line_bytes: read("coherency_line_size")?.parse().ok()?,
                    })
                })();
                if let Some(c) = entry {
                    result.push(c);
                }
            }
        }
    }
    result.sort();
    result.dedup();
    result
}

#[cfg(any(target_os = "linux", test))]
fn parse_size(s: &str) -> Option<usize> {
    let (n, factor) = if let Some(n) = s.strip_suffix('K') {
        (n, 1024)
    } else if let Some(n) = s.strip_suffix('M') {
        (n, 1024 * 1024)
    } else {
        (s, 1)
    };
    n.parse::<usize>().ok()?.checked_mul(factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_size_units_and_overflow() {
        assert_eq!(parse_size("512K"), Some(524288));
        assert_eq!(parse_size("32M"), Some(33554432));
        assert_eq!(parse_size("64"), Some(64));
        assert_eq!(parse_size("unknown"), None);
        assert_eq!(parse_size(&format!("{}M", usize::MAX)), None);
    }
}
