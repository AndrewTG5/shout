pub const SAMPLE_RATE: u32 = 48_000;
/// 10ms frame at 48 kHz (matches `nnnoiseless::DenoiseState::FRAME_SIZE`).
pub const FRAME_SAMPLES: usize = 480;
/// 480 little-endian `i16` samples = 960 bytes per UDP packet (well within 1500B Ethernet MTU).
pub const PACKET_BYTES: usize = FRAME_SAMPLES * std::mem::size_of::<i16>();

pub const SERVER_BIND_ADDR: &str = "192.168.1.1:4321";

/// Encode 480 `i16` samples into a 960-byte little-endian buffer.
pub fn encode_frame(samples: &[i16; FRAME_SAMPLES], out: &mut [u8; PACKET_BYTES]) {
    for (i, &sample) in samples.iter().enumerate() {
        let bytes = sample.to_le_bytes();
        out[i * 2] = bytes[0];
        out[i * 2 + 1] = bytes[1];
    }
}

/// Decode a 960-byte little-endian buffer into 480 `i16` samples.
pub fn decode_frame(bytes: &[u8; PACKET_BYTES], out: &mut [i16; FRAME_SAMPLES]) {
    for (dst, &chunk) in out.iter_mut().zip(bytes.as_chunks::<2>().0.iter()) {
        *dst = i16::from_le_bytes(chunk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_frame_encoding() {
        let mut orig = [0i16; FRAME_SAMPLES];
        for (i, s) in orig.iter_mut().enumerate() {
            *s = (i as i16).wrapping_mul(67).wrapping_sub(10_000);
        }
        let mut packet = [0u8; PACKET_BYTES];
        encode_frame(&orig, &mut packet);

        let mut decoded = [0i16; FRAME_SAMPLES];
        decode_frame(&packet, &mut decoded);
        assert_eq!(orig, decoded);
    }
}
