#![forbid(unsafe_code)]

pub fn vbyte_encode_deltas(values: &[u32]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut prev = 0_u32;
    for &value in values {
        let delta = value.saturating_sub(prev);
        encode_vbyte(delta, &mut out);
        prev = value;
    }
    out
}

pub fn vbyte_decode_deltas(bytes: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut shift = 0_u32;
    let mut acc = 0_u32;
    let mut prev = 0_u32;

    for &byte in bytes {
        acc |= u32::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            let value = prev.saturating_add(acc);
            out.push(value);
            prev = value;
            acc = 0;
            shift = 0;
        } else {
            shift += 7;
        }
    }

    out
}

fn encode_vbyte(mut value: u32, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push(((value & 0x7F) as u8) | 0x80);
        value >>= 7;
    }
    out.push((value & 0x7F) as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_vbyte_deltas() {
        let values = vec![2, 4, 32, 128, 2048];
        let encoded = vbyte_encode_deltas(&values);
        let decoded = vbyte_decode_deltas(&encoded);
        assert_eq!(values, decoded);
    }
}
