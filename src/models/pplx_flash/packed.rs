use anyhow::{Result, ensure};

pub(super) const GROUP: usize = 64;

/// Bytes of one group of 64 codes and its f32 scale: Q4 packs two codes per byte.
const fn group_bytes(bits: u32) -> usize {
    GROUP * bits as usize / 8 + 4
}

pub(super) fn size(rows: usize, columns: usize, bits: u32) -> Result<usize> {
    ensure!(bits == 4 || bits == 8, "packed weights are Q4 or Q8");
    ensure!(
        rows > 0 && columns > 0 && columns % GROUP == 0,
        "invalid group-64 matrix shape"
    );
    rows.checked_mul(columns / GROUP)
        .and_then(|n| n.checked_mul(group_bytes(bits)))
        .ok_or_else(|| anyhow::anyhow!("packed matrix size overflow"))
}

// Q4 codes are stored as code+8 nibbles, low nibble first; Q8 codes as two's-complement bytes.
fn code(group: &[u8], k: usize, bits: u32) -> i32 {
    if bits == 4 {
        ((group[k / 2] >> (4 * (k % 2))) & 15) as i32 - 8
    } else {
        group[k] as i8 as i32
    }
}

#[cfg(any(feature = "cpu", feature = "kernel-checks", feature = "pplx-webgpu", test))]
pub(super) fn pack(bytes: &[u8], rows: usize, columns: usize, bits: u32) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(size(rows, columns, bits)?);
    ensure!(
        bytes.len() == rows * columns * 2,
        "invalid BF16 matrix byte count"
    );
    let limit = if bits == 4 { 7. } else { 127. };
    for group in bytes.chunks_exact(GROUP * 2) {
        let values: [f32; GROUP] = std::array::from_fn(|i| {
            f32::from_bits((u16::from_le_bytes([group[2 * i], group[2 * i + 1]]) as u32) << 16)
        });
        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite packed weight");
        let scale = (values.iter().fold(0f32, |a, v| a.max(v.abs())) / limit).max(f32::MIN_POSITIVE);
        let inverse = scale.recip();
        let code = |v: f32| (v * inverse).round_ties_even().clamp(-limit, limit) as i32;
        if bits == 4 {
            for pair in values.chunks_exact(2) {
                output.push((code(pair[0]) + 8) as u8 | ((code(pair[1]) + 8) as u8) << 4);
            }
        } else {
            output.extend(values.iter().map(|&v| code(v) as u8));
        }
        output.extend(scale.to_le_bytes());
    }
    Ok(output)
}

#[cfg(any(feature = "webgpu", test))]
pub(super) fn unpack(bytes: &[u8], bits: u32, output: &mut Vec<f32>) -> Result<()> {
    let stride = size(1, GROUP, bits)?;
    ensure!(
        !bytes.is_empty() && bytes.len() % stride == 0,
        "invalid packed rows"
    );
    output.clear();
    output.reserve(bytes.len() / stride * GROUP);
    for group in bytes.chunks_exact(stride) {
        let scale = f32::from_le_bytes(group[stride - 4..].try_into().unwrap());
        ensure!(scale.is_finite() && scale > 0., "invalid packed scale");
        output.extend((0..GROUP).map(|k| code(group, k, bits) as f32 * scale));
    }
    Ok(())
}

pub(super) fn project(
    bytes: &[u8],
    bits: u32,
    columns: usize,
    x: &[f32],
    outputs: usize,
) -> Result<Vec<f32>> {
    let stride = size(1, columns, bits)?;
    let group_stride = size(1, GROUP, bits)?;
    ensure!(
        x.len() == columns && bytes.len() >= outputs * stride,
        "invalid packed readout shape"
    );
    ensure!(
        x.iter().all(|v| v.is_finite()),
        "nonfinite readout activation"
    );
    let groups = columns / GROUP;
    let mut codes = vec![0i32; columns];
    let mut scales = Vec::with_capacity(groups);
    for (values, q) in x.chunks_exact(GROUP).zip(codes.chunks_exact_mut(GROUP)) {
        let scale = (values.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.).max(f32::MIN_POSITIVE);
        let inverse = scale.recip();
        for (q, &v) in q.iter_mut().zip(values) {
            *q = (v * inverse).round_ties_even().clamp(-127., 127.) as i32;
        }
        scales.push(scale);
    }
    Ok(bytes
        .chunks_exact(stride)
        .take(outputs)
        .map(|row| {
            let mut sum = 0.;
            for (g, group) in row.chunks_exact(group_stride).enumerate() {
                let scale = f32::from_le_bytes(group[group_stride - 4..].try_into().unwrap());
                let mut dot = 0;
                for k in 0..GROUP {
                    dot += codes[g * GROUP + k] * code(group, k, bits);
                }
                sum += dot as f32 * (scales[g] * scale);
            }
            sum
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouped_codes_and_scale_match_native() -> Result<()> {
        let values: Vec<f32> = (0..128)
            .map(|i| {
                if i < 64 {
                    (i as f32 - 32.) / 16.
                } else {
                    (i as f32 - 96.) * 8.
                }
            })
            .collect();
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        for (bits, group) in [(4, 36), (8, 68)] {
            let packed = pack(&bytes, 1, 128, bits)?;
            assert_eq!(packed.len(), 2 * group);
            #[cfg(all(feature = "cpu", feature = "int8"))]
            for (g, v) in values.chunks_exact(64).enumerate() {
                let scale = crate::pplx_flash::int8::group_scale_bits(v, false, bits as u8);
                let mut codes = [0i8; 64];
                crate::pplx_flash::int8::pack_bits(v, &mut codes, scale, false, bits as u8);
                let packed = &packed[g * group..(g + 1) * group];
                assert_eq!(&packed[group - 4..], &scale.to_le_bytes());
                for k in 0..64 {
                    assert_eq!(code(packed, k, bits), codes[k] as i32);
                }
            }
            let mut decoded = Vec::new();
            unpack(&packed, bits, &mut decoded)?;
            assert_eq!(decoded.len(), 128);
            assert!(project(&packed, bits, 128, &values, 1)?[0].is_finite());
        }
        assert!(size(1, 63, 4).is_err());
        assert!(size(1, 64, 2).is_err());
        Ok(())
    }
    #[cfg(all(feature = "cpu", feature = "int8"))]
    #[test]
    fn packed_projection_and_embeddings_match_native() -> Result<()> {
        let (n, k) = (67, 128);
        let values: Vec<f32> = (0..n * k)
            .map(|i| {
                if i % 128 >= 64 {
                    0.
                } else {
                    ((i * 23 % 127) as i32 - 63) as f32 / 32.
                }
            })
            .collect();
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..k)
            .map(|i| {
                if i == 0 {
                    127.
                } else {
                    ((i % 9) as f32 - 4.) * 0.5
                }
            })
            .collect();
        for bits in [4, 8] {
            let packed = pack(&bytes, n, k, bits)?;
            let native = crate::pplx_flash::int8::Matrix::from_bf16_quantized(
                &bytes, k, n, false, bits as u8, 8,
            )?;
            let mut reference = Vec::new();
            native.apply(&input, &mut reference, &mut Default::default());
            assert_eq!(project(&packed, bits, k, &input, 3)?, reference[..3]);
            assert_eq!(project(&packed, bits, k, &input, n)?, reference);
            let embedding = crate::pplx_flash::quantized_rows::Embedding::from_bf16(
                &bytes, k, bits as u8, false,
            )?;
            let mut reference = Vec::new();
            embedding.append(5, &mut reference);
            let row = size(1, k, bits)?;
            let mut decoded = Vec::new();
            unpack(&packed[5 * row..6 * row], bits, &mut decoded)?;
            assert_eq!(decoded, reference);
        }
        Ok(())
    }
}
