use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::BTreeMap,
    io::{Cursor, Read},
};
#[derive(Clone, Debug)]
enum V {
    Mark,
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Global(String),
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict(BTreeMap<String, V>),
    Storage(String, usize),
    Tensor {
        key: String,
        count: usize,
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<usize>,
    },
}
impl V {
    fn string(&self) -> Result<&str> {
        if let Self::String(s) = self {
            Ok(s)
        } else {
            bail!("expected checkpoint string")
        }
    }
    fn int(&self) -> Result<usize> {
        if let Self::Int(n) = self {
            Ok((*n).try_into()?)
        } else {
            bail!("expected checkpoint integer")
        }
    }
    fn tuple(&self) -> Result<&[V]> {
        if let Self::Tuple(v) = self {
            Ok(v)
        } else {
            bail!("expected checkpoint tuple")
        }
    }
    fn dict(&self) -> Result<&BTreeMap<String, V>> {
        if let Self::Dict(v) = self {
            Ok(v)
        } else {
            bail!("expected checkpoint dictionary")
        }
    }
    fn number(&self) -> Result<f64> {
        match self {
            Self::Float(v) => Ok(*v),
            Self::Int(v) => Ok(*v as f64),
            _ => bail!("expected checkpoint number"),
        }
    }
    fn is_false(&self) -> bool {
        matches!(self, Self::Bool(false) | Self::Int(0))
    }
}
struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    stack: Vec<V>,
    memo: BTreeMap<u32, V>,
}
impl<'a> Parser<'a> {
    fn read(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("pickle offset overflow")?;
        let b = self
            .bytes
            .get(self.pos..end)
            .context("truncated checkpoint metadata")?;
        self.pos = end;
        Ok(b)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.read(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read(4)?.try_into()?))
    }
    fn line(&mut self) -> Result<String> {
        let len = self.bytes[self.pos..]
            .iter()
            .position(|x| *x == b'\n')
            .context("unterminated pickle global")?;
        let text = String::from_utf8(self.read(len)?.to_vec())?;
        self.read(1)?;
        Ok(text)
    }
    fn pop(&mut self) -> Result<V> {
        self.stack.pop().context("empty pickle stack")
    }
    fn marked(&mut self) -> Result<Vec<V>> {
        let mark = self
            .stack
            .iter()
            .rposition(|v| matches!(v, V::Mark))
            .context("missing pickle mark")?;
        let out = self.stack.split_off(mark + 1);
        self.stack.pop();
        Ok(out)
    }
    fn items(&mut self, items: Vec<V>) -> Result<()> {
        ensure!(items.len() % 2 == 0, "odd pickle dictionary");
        let Some(V::Dict(map)) = self.stack.last_mut() else {
            bail!("SETITEM target is not a dictionary")
        };
        for pair in items.chunks_exact(2) {
            map.insert(pair[0].string()?.to_owned(), pair[1].clone());
        }
        Ok(())
    }
    fn parse(mut self) -> Result<V> {
        ensure!(
            self.bytes.len() <= 1024 * 1024,
            "head metadata exceeds 1 MiB"
        );
        loop {
            let op = self.u8()?;
            match op {
                0x80 => ensure!(self.u8()? == 2, "only Kev protocol-2 metadata is supported"),
                b'.' => {
                    ensure!(
                        self.stack.len() == 1 && self.pos == self.bytes.len(),
                        "invalid pickle end"
                    );
                    return self.pop();
                }
                b'(' => self.stack.push(V::Mark),
                b'N' => self.stack.push(V::Null),
                0x88 => self.stack.push(V::Bool(true)),
                0x89 => self.stack.push(V::Bool(false)),
                b'}' => self.stack.push(V::Dict(BTreeMap::new())),
                b']' => self.stack.push(V::List(Vec::new())),
                b')' => self.stack.push(V::Tuple(Vec::new())),
                b'K' => {
                    let n = self.u8()?;
                    self.stack.push(V::Int(n.into()));
                }
                b'M' => {
                    let n = u16::from_le_bytes(self.read(2)?.try_into()?);
                    self.stack.push(V::Int(n.into()));
                }
                b'J' => {
                    let n = i32::from_le_bytes(self.read(4)?.try_into()?);
                    self.stack.push(V::Int(n.into()));
                }
                b'G' => {
                    let x = f64::from_be_bytes(self.read(8)?.try_into()?);
                    ensure!(x.is_finite(), "non-finite metadata");
                    self.stack.push(V::Float(x));
                }
                b'X' => {
                    let n = self.u32()? as usize;
                    let s = String::from_utf8(self.read(n)?.to_vec())?;
                    self.stack.push(V::String(s));
                }
                b'q' | b'r' => {
                    let index = if op == b'q' {
                        self.u8()? as u32
                    } else {
                        self.u32()?
                    };
                    let value = self.stack.last().context("memo on empty stack")?.clone();
                    self.memo.insert(index, value);
                }
                b'h' | b'j' => {
                    let index = if op == b'h' {
                        self.u8()? as u32
                    } else {
                        self.u32()?
                    };
                    let value = self
                        .memo
                        .get(&index)
                        .context("missing pickle memo")?
                        .clone();
                    ensure!(
                        !matches!(value, V::Dict(_) | V::List(_)),
                        "mutable pickle aliases are unsupported"
                    );
                    self.stack.push(value);
                }
                b't' => {
                    let values = self.marked()?;
                    self.stack.push(V::Tuple(values));
                }
                0x85..=0x87 => {
                    let n = (op - 0x84) as usize;
                    ensure!(self.stack.len() >= n, "short tuple stack");
                    let values = self.stack.split_off(self.stack.len() - n);
                    self.stack.push(V::Tuple(values));
                }
                b'e' => {
                    let values = self.marked()?;
                    let Some(V::List(list)) = self.stack.last_mut() else {
                        bail!("APPENDS target is not a list")
                    };
                    list.extend(values);
                }
                b'u' => {
                    let values = self.marked()?;
                    self.items(values)?;
                }
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    self.items(vec![key, value])?;
                }
                b'c' => {
                    let name = format!("{} {}", self.line()?, self.line()?);
                    ensure!(
                        matches!(
                            name.as_str(),
                            "collections OrderedDict"
                                | "torch FloatStorage"
                                | "torch._utils _rebuild_tensor_v2"
                        ),
                        "unsupported pickle global {name}"
                    );
                    self.stack.push(V::Global(name));
                }
                b'Q' => {
                    let value = self.pop()?;
                    let v = value.tuple()?;
                    ensure!(
                        v.len() == 5
                            && v[0].string()? == "storage"
                            && matches!(&v[1],V::Global(s)if s=="torch FloatStorage"),
                        "unsupported tensor storage"
                    );
                    let key = v[2].string()?.to_owned();
                    ensure!(
                        !key.is_empty() && key.bytes().all(|c| c.is_ascii_digit()),
                        "invalid storage key"
                    );
                    let count = v[4].int()?;
                    ensure!(count <= 4 * 1024 * 1024, "oversized head storage");
                    self.stack.push(V::Storage(key, count));
                }
                b'R' => {
                    let args = self.pop()?;
                    let global = self.pop()?;
                    let V::Global(name) = global else {
                        bail!("REDUCE target is not a recognized global")
                    };
                    let args = args.tuple()?;
                    if name == "collections OrderedDict" {
                        ensure!(args.is_empty(), "unsupported OrderedDict constructor");
                        self.stack.push(V::Dict(BTreeMap::new()));
                    } else {
                        ensure!(
                            name == "torch._utils _rebuild_tensor_v2" && args.len() == 6,
                            "unsupported reconstruction"
                        );
                        let V::Storage(key, count) = &args[0] else {
                            bail!("tensor has no storage")
                        };
                        let shape = args[2]
                            .tuple()?
                            .iter()
                            .map(V::int)
                            .collect::<Result<Vec<_>>>()?;
                        let stride = args[3]
                            .tuple()?
                            .iter()
                            .map(V::int)
                            .collect::<Result<Vec<_>>>()?;
                        self.stack.push(V::Tensor {
                            key: key.clone(),
                            count: *count,
                            offset: args[1].int()?,
                            shape,
                            stride,
                        });
                    }
                }
                b'b' => {
                    let attrs = self.pop()?;
                    let attrs = attrs.dict()?;
                    ensure!(
                        attrs.len() == 1
                            && attrs.contains_key("_metadata")
                            && matches!(self.stack.last(), Some(V::Dict(_))),
                        "unsupported pickle BUILD state"
                    );
                }
                _ => bail!("unsupported head pickle opcode {op:#x}"),
            }
            ensure!(
                self.stack.len() <= 10000 && self.memo.len() <= 10000,
                "oversized pickle stack/memo"
            );
        }
    }
}
pub(super) struct Head {
    pub q: Vec<f32>,
    pub qb: Vec<f32>,
    pub k: Vec<f32>,
    pub kb: Vec<f32>,
    pub temperature: f64,
}
impl Head {
    pub fn load(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() <= 32 * 1024 * 1024,
            "head archive exceeds 32 MiB"
        );
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes))?;
        fn member(
            archive: &mut zip::ZipArchive<Cursor<&[u8]>>,
            name: &str,
            limit: usize,
        ) -> Result<Vec<u8>> {
            let mut file = archive.by_name(name)?;
            ensure!(
                file.size() <= limit as u64,
                "oversized archive member {name}"
            );
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            Ok(bytes)
        }
        ensure!(
            member(&mut archive, "head/byteorder", 16)? == b"little",
            "unsupported head byte order"
        );
        let pickle = member(&mut archive, "head/data.pkl", 1024 * 1024)?;
        let metadata = Parser {
            bytes: &pickle,
            pos: 0,
            stack: Vec::new(),
            memo: BTreeMap::new(),
        }
        .parse()?;
        let meta = metadata.dict()?;
        let get = |key: &str| {
            meta.get(key)
                .with_context(|| format!("missing head metadata {key}"))
        };
        ensure!(
            get("base")?.string()? == "Qwen/Qwen3.5-0.8B-Base"
                && get("base_revision")?.string()? == super::BASE_REVISION,
            "unsupported Kev base checkpoint"
        );
        ensure!(
            get("head_dim")?.int()? == 256
                && get("lora")?.int()? == 16
                && get("option_isolation")?.is_false()
                && get("special_embeddings")?.is_false()
                && get("weights_dtype")?.string()? == "fp32",
            "unsupported Kev head architecture"
        );
        let temperature = get("temperature")?.number()?;
        ensure!(
            temperature > 0. && temperature.is_finite(),
            "invalid Kev temperature"
        );
        let tensors = get("head")?.dict()?;
        let mut tensor = |name: &str, expected: &[usize]| -> Result<Vec<f32>> {
            let V::Tensor {
                key,
                count,
                offset,
                shape,
                stride,
            } = tensors.get(name).context("missing head tensor")?
            else {
                bail!("head entry is not a tensor")
            };
            let mut expected_stride = vec![1; expected.len()];
            for i in (0..expected.len().saturating_sub(1)).rev() {
                expected_stride[i] = expected_stride[i + 1] * expected[i + 1];
            }
            ensure!(
                shape == expected
                    && stride == &expected_stride
                    && *offset == 0
                    && *count == expected.iter().product::<usize>(),
                "unsupported head tensor layout {name}"
            );
            let bytes = member(&mut archive, &format!("head/data/{key}"), count * 4)?;
            ensure!(bytes.len() == count * 4, "head storage size mismatch");
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "non-finite head tensor"
            );
            Ok(values)
        };
        Ok(Self {
            q: tensor("q.weight", &[256, 1024])?,
            qb: tensor("q.bias", &[256])?,
            k: tensor("k.weight", &[256, 1024])?,
            kb: tensor("k.bias", &[256])?,
            temperature,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_executable_or_truncated_pickle() {
        for bytes in [
            b"\x80\x02cos\nsystem\n.".as_slice(),
            b"\x80\x02X\xff\xff\xff\xff",
            b"\x80\x02h\x01.",
            b"\x80\x05N.",
        ] {
            assert!(
                Parser {
                    bytes,
                    pos: 0,
                    stack: Vec::new(),
                    memo: BTreeMap::new()
                }
                .parse()
                .is_err()
            );
        }
        assert!(Head::load(b"not a zip archive").is_err());
    }
}
