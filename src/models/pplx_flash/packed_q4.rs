use anyhow::{Result, ensure};

pub(super) const GROUP: usize = 64;
pub(super) const GROUP_BYTES: usize = 36;

pub(super) fn size(rows: usize, columns: usize) -> Result<usize> {
    ensure!(
        rows > 0 && columns > 0 && columns % GROUP == 0,
        "invalid group-64 matrix shape"
    );
    rows.checked_mul(columns / GROUP)
        .and_then(|n| n.checked_mul(GROUP_BYTES))
        .ok_or_else(|| anyhow::anyhow!("packed matrix size overflow"))
}

#[cfg(any(feature = "cpu", feature = "kernel-checks", feature = "pplx-webgpu", test))]
pub(super) fn pack(bytes: &[u8], rows: usize, columns: usize) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(size(rows, columns)?);
    ensure!(
        bytes.len() == rows * columns * 2,
        "invalid BF16 matrix byte count"
    );
    for group in bytes.chunks_exact(GROUP * 2) {
        let values: [f32; GROUP] = std::array::from_fn(|i| {
            f32::from_bits((u16::from_le_bytes([group[2 * i], group[2 * i + 1]]) as u32) << 16)
        });
        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite Q4 weight");
        let scale = (values.iter().fold(0f32, |a, v| a.max(v.abs())) / 7.).max(f32::MIN_POSITIVE);
        let inverse = scale.recip();
        let code = |v: f32| ((v * inverse).round_ties_even().clamp(-7., 7.) as i32 + 8) as u8;
        for pair in values.chunks_exact(2) {
            output.push(code(pair[0]) | (code(pair[1]) << 4));
        }
        output.extend(scale.to_le_bytes());
    }
    Ok(output)
}

#[cfg(any(feature = "webgpu", test))]
pub(super) fn unpack(bytes: &[u8], output: &mut Vec<f32>) -> Result<()> {
    ensure!(
        !bytes.is_empty() && bytes.len() % GROUP_BYTES == 0,
        "invalid packed Q4 rows"
    );
    output.clear();
    output.reserve(bytes.len() / GROUP_BYTES * GROUP);
    for group in bytes.chunks_exact(GROUP_BYTES) {
        let scale = f32::from_le_bytes(group[32..36].try_into().unwrap());
        ensure!(scale.is_finite() && scale > 0., "invalid Q4 scale");
        for &pair in &group[..32] {
            output.push(((pair & 15) as i32 - 8) as f32 * scale);
            output.push(((pair >> 4) as i32 - 8) as f32 * scale);
        }
    }
    Ok(())
}

pub(super) fn project(bytes: &[u8], columns: usize, x: &[f32], outputs: usize) -> Result<Vec<f32>> {
    let stride = size(1, columns)?;
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
            for (g, group) in row.chunks_exact(GROUP_BYTES).enumerate() {
                let scale = f32::from_le_bytes(group[32..36].try_into().unwrap());
                let mut dot = 0;
                for k in 0..GROUP {
                    dot +=
                        codes[g * GROUP + k] * (((group[k / 2] >> (4 * (k % 2))) & 15) as i32 - 8);
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
        let packed = pack(&bytes, 1, 128)?;
        assert_eq!(packed.len(), 72);
        #[cfg(all(feature = "cpu", feature = "int8"))]
        for (g, v) in values.chunks_exact(64).enumerate() {
            let scale = crate::pplx_flash::int8::group_scale_bits(v, false, 4);
            let mut codes = [0i8; 64];
            crate::pplx_flash::int8::pack_bits(v, &mut codes, scale, false, 4);
            assert_eq!(&packed[g * 36 + 32..g * 36 + 36], &scale.to_le_bytes());
            for k in 0..64 {
                assert_eq!(
                    ((packed[g * 36 + k / 2] >> (4 * (k % 2))) & 15) as i8 - 8,
                    codes[k]
                );
            }
        }
        let mut decoded = Vec::new();
        unpack(&packed, &mut decoded)?;
        assert_eq!(decoded.len(), 128);
        assert!(project(&packed, 128, &values, 1)?[0].is_finite());
        assert!(size(1, 63).is_err());
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
        let packed = pack(&bytes, n, k)?;
        let native =
            crate::pplx_flash::int8::Matrix::from_bf16_quantized(&bytes, k, n, false, 4, 8)?;
        let input: Vec<f32> = (0..k)
            .map(|i| {
                if i == 0 {
                    127.
                } else {
                    ((i % 9) as f32 - 4.) * 0.5
                }
            })
            .collect();
        let mut reference = Vec::new();
        native.apply(&input, &mut reference, &mut Default::default());
        assert_eq!(project(&packed, k, &input, 3)?, reference[..3]);
        assert_eq!(project(&packed, k, &input, n)?, reference);
        let embedding =
            crate::pplx_flash::quantized_rows::Embedding::from_bf16(&bytes, k, 4, false)?;
        let mut reference = Vec::new();
        embedding.append(5, &mut reference);
        let mut decoded = Vec::new();
        unpack(&packed[5 * 72..6 * 72], &mut decoded)?;
        assert_eq!(decoded, reference);
        Ok(())
    }
}
