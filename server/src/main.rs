use anyhow::{Context, Result};
use nnnoiseless::DenoiseState;
use protocol::{
    FRAME_SAMPLES, PACKET_BYTES, SAMPLE_RATE, SERVER_BIND_ADDR, decode_frame, encode_frame,
};
use std::collections::{HashMap, VecDeque};
use std::io::ErrorKind;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

const CLIENT_TIMEOUT: Duration = Duration::from_secs(5);
const TICK_INTERVAL: Duration = Duration::from_millis(10);
const VAD_THRESHOLD: f32 = 0.6;
/// 30 frames * 10ms = 300ms hangover after voice activity drops below threshold.
const VAD_HANGOVER_FRAMES: u32 = 30;
/// Maximum queued 10ms frames per client for >2 client rooms (10ms max server-side queue).
const MAX_QUEUED_FRAMES: usize = 1;

struct ClientState {
    denoiser: Box<DenoiseState<'static>>,
    last_seen: Instant,
    hangover_remaining: u32,
    active_frames: VecDeque<[f32; FRAME_SAMPLES]>,
}

impl ClientState {
    fn new(now: Instant) -> Self {
        Self {
            denoiser: DenoiseState::new(),
            last_seen: now,
            hangover_remaining: 0,
            active_frames: VecDeque::with_capacity(MAX_QUEUED_FRAMES),
        }
    }

    /// Runs `nnnoiseless` on an incoming 10ms frame (scaled in `[-32768.0, 32767.0]`)
    /// and returns `Some(denoised)` if the VAD + hangover gate is open.
    fn denoise_and_gate(
        &mut self,
        pcm: &[i16; FRAME_SAMPLES],
        now: Instant,
    ) -> Option<[f32; FRAME_SAMPLES]> {
        self.last_seen = now;

        let mut input_f32 = [0.0f32; FRAME_SAMPLES];
        for (dst, &src) in input_f32.iter_mut().zip(pcm.iter()) {
            *dst = src as f32;
        }

        let mut denoised = [0.0f32; FRAME_SAMPLES];
        let vad_prob = self.denoiser.process_frame(&mut denoised, &input_f32);

        if self.update_vad_gate(vad_prob) {
            Some(denoised)
        } else {
            None
        }
    }

    fn update_vad_gate(&mut self, vad_prob: f32) -> bool {
        if vad_prob >= VAD_THRESHOLD {
            self.hangover_remaining = VAD_HANGOVER_FRAMES;
            true
        } else if self.hangover_remaining > 0 {
            self.hangover_remaining -= 1;
            true
        } else {
            false
        }
    }

    fn push_active_frame(&mut self, frame: [f32; FRAME_SAMPLES]) {
        if self.active_frames.len() >= MAX_QUEUED_FRAMES {
            self.active_frames.pop_front();
        }
        self.active_frames.push_back(frame);
    }
}

fn encode_f32_frame(frame: &[f32; FRAME_SAMPLES], out_packet: &mut [u8; PACKET_BYTES]) {
    let mut pcm_i16 = [0i16; FRAME_SAMPLES];
    for (dst, &s) in pcm_i16.iter_mut().zip(frame.iter()) {
        *dst = s.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    }
    encode_frame(&pcm_i16, out_packet);
}

/// Computes `N-1` mixed frames for all connected `recipients` given the set of `active_speakers`
/// for the current 10ms tick. Returns `(recipient_addr, encoded_packet)` only for recipients
/// that have at least one *other* active speaker.
fn compute_n_minus_1_packets(
    recipients: &[SocketAddr],
    active_speakers: &HashMap<SocketAddr, [f32; FRAME_SAMPLES]>,
) -> Vec<(SocketAddr, [u8; PACKET_BYTES])> {
    if active_speakers.is_empty() || recipients.len() <= 1 {
        return Vec::new();
    }

    let mut total_mix = [0.0f32; FRAME_SAMPLES];
    for frame in active_speakers.values() {
        for (acc, &s) in total_mix.iter_mut().zip(frame.iter()) {
            *acc += s;
        }
    }

    let total_active = active_speakers.len();
    let mut out_packets = Vec::with_capacity(recipients.len());

    for &recipient in recipients {
        let self_frame = active_speakers.get(&recipient);
        let other_active_count = if self_frame.is_some() {
            total_active - 1
        } else {
            total_active
        };

        if other_active_count == 0 {
            continue;
        }

        let mut packet = [0u8; PACKET_BYTES];
        if let Some(own) = self_frame {
            let mut mixed_i16 = [0i16; FRAME_SAMPLES];
            for ((dst, &tot), &self_s) in mixed_i16.iter_mut().zip(total_mix.iter()).zip(own.iter())
            {
                let sample = (tot - self_s).clamp(i16::MIN as f32, i16::MAX as f32);
                *dst = sample as i16;
            }
            encode_frame(&mixed_i16, &mut packet);
        } else {
            encode_f32_frame(&total_mix, &mut packet);
        }

        out_packets.push((recipient, packet));
    }

    out_packets
}

fn main() -> Result<()> {
    let socket = UdpSocket::bind(SERVER_BIND_ADDR)
        .with_context(|| format!("Failed to bind UDP socket on {SERVER_BIND_ADDR}"))?;

    println!(
        "=== Shout VOIP Server listening on {SERVER_BIND_ADDR} ({SAMPLE_RATE} Hz, {FRAME_SAMPLES} samples/frame) ==="
    );

    let mut clients: HashMap<SocketAddr, ClientState> = HashMap::new();
    let mut recv_buf = [0u8; 2048];
    let mut decoded_pcm = [0i16; FRAME_SAMPLES];
    let mut direct_packet = [0u8; PACKET_BYTES];
    let mut next_tick = Instant::now() + TICK_INTERVAL;

    loop {
        let now = Instant::now();
        if let Some(wait) = next_tick.checked_duration_since(now) {
            // Wait up to `wait` (at least 100us to avoid zero-timeout error on all OSes)
            let timeout = wait.max(Duration::from_micros(100));
            socket.set_read_timeout(Some(timeout))?;

            match socket.recv_from(&mut recv_buf) {
                Ok((len, peer)) => {
                    if len == PACKET_BYTES {
                        let pkt: &[u8; PACKET_BYTES] = recv_buf[..PACKET_BYTES].try_into().unwrap();
                        decode_frame(pkt, &mut decoded_pcm);

                        let rx_time = Instant::now();
                        clients.entry(peer).or_insert_with(|| {
                            println!("[+] Client connected: {peer}");
                            ClientState::new(rx_time)
                        });

                        let num_clients = clients.len();
                        let other_peer = (num_clients == 2)
                            .then(|| clients.keys().find(|&&addr| addr != peer).copied())
                            .flatten();

                        let client = clients.get_mut(&peer).unwrap();
                        if let Some(denoised) = client.denoise_and_gate(&decoded_pcm, rx_time) {
                            if let Some(_) = other_peer {
                                // Fast path for 2 clients: immediate zero-tick forwarding
                                encode_f32_frame(&denoised, &mut direct_packet);
                            } else if num_clients > 2 {
                                client.push_active_frame(denoised);
                            }
                        }
                    }
                }
                Err(err)
                    if err.kind() == ErrorKind::WouldBlock
                        || err.kind() == ErrorKind::TimedOut
                        || err.kind() == ErrorKind::Interrupted => {}
                Err(err) => {
                    eprintln!("UDP receive error: {err}");
                }
            }
        }

        let now = Instant::now();
        if now >= next_tick {
            // Advance next_tick; if we fell behind by multiple ticks, snap forward
            next_tick += TICK_INTERVAL;
            if next_tick <= now {
                next_tick = now + TICK_INTERVAL;
            }

            // 1. Prune clients inactive for > 5 seconds
            clients.retain(|addr, state| {
                let alive = now.duration_since(state.last_seen) <= CLIENT_TIMEOUT;
                if !alive {
                    println!("[-] Client timed out ({}s): {addr}", CLIENT_TIMEOUT.as_secs());
                }
                alive
            });

            // 2. Mix N-1 audio on the 10ms tick only when > 2 clients are connected
            if clients.len() > 2 {
                let mut active_speakers: HashMap<SocketAddr, [f32; FRAME_SAMPLES]> =
                    HashMap::with_capacity(clients.len());
                for (&addr, state) in clients.iter_mut() {
                    if let Some(frame) = state.active_frames.pop_front() {
                        active_speakers.insert(addr, frame);
                    }
                }

                if !active_speakers.is_empty() {
                    let recipients: Vec<SocketAddr> = clients.keys().copied().collect();
                    let packets = compute_n_minus_1_packets(&recipients, &active_speakers);
                    for (recipient, packet) in packets {
                        if let Err(err) = socket.send_to(&packet, recipient) {
                            eprintln!("Failed to send audio packet to {recipient}: {err}");
                        }
                    }
                }
            } else {
                for state in clients.values_mut() {
                    state.active_frames.clear();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_size_matches_nnnoiseless() {
        assert_eq!(
            FRAME_SAMPLES,
            DenoiseState::FRAME_SIZE,
            "protocol::FRAME_SAMPLES must match nnnoiseless::DenoiseState::FRAME_SIZE"
        );
    }

    #[test]
    fn vad_gate_and_hangover_behavior() {
        let mut client = ClientState::new(Instant::now());

        // Below threshold initially -> closed
        assert!(!client.update_vad_gate(0.1));
        assert_eq!(client.hangover_remaining, 0);

        // Above threshold -> opens and sets hangover to VAD_HANGOVER_FRAMES
        assert!(client.update_vad_gate(0.8));
        assert_eq!(client.hangover_remaining, VAD_HANGOVER_FRAMES);

        // Next 20 frames below threshold -> stays open due to hangover
        for expected_rem in (0..VAD_HANGOVER_FRAMES).rev() {
            assert!(client.update_vad_gate(0.2));
            assert_eq!(client.hangover_remaining, expected_rem);
        }

        // 21st frame below threshold -> closes
        assert!(!client.update_vad_gate(0.2));
    }

    #[test]
    fn bounded_queue_drops_oldest_frame() {
        let mut client = ClientState::new(Instant::now());
        for i in 1..=(MAX_QUEUED_FRAMES + 2) {
            client.push_active_frame([i as f32; FRAME_SAMPLES]);
        }
        assert_eq!(client.active_frames.len(), MAX_QUEUED_FRAMES);
        assert_eq!(
            client.active_frames.front().unwrap()[0],
            (MAX_QUEUED_FRAMES + 2) as f32
        );
    }

    #[test]
    fn n_minus_1_mixing_excludes_self_and_sums_others() {
        let addr_a: SocketAddr = "127.0.0.1:10001".parse().unwrap();
        let addr_b: SocketAddr = "127.0.0.1:10002".parse().unwrap();
        let addr_c: SocketAddr = "127.0.0.1:10003".parse().unwrap();
        let recipients = vec![addr_a, addr_b, addr_c];

        // Case 1: Only A is speaking -> A gets nothing, B and C get A's audio
        let mut active = HashMap::new();
        active.insert(addr_a, [1000.0f32; FRAME_SAMPLES]);

        let packets = compute_n_minus_1_packets(&recipients, &active);
        let pkt_map: HashMap<SocketAddr, [u8; PACKET_BYTES]> = packets.into_iter().collect();
        assert!(!pkt_map.contains_key(&addr_a));
        assert!(pkt_map.contains_key(&addr_b));
        assert!(pkt_map.contains_key(&addr_c));

        let mut decoded = [0i16; FRAME_SAMPLES];
        decode_frame(&pkt_map[&addr_b], &mut decoded);
        assert_eq!(decoded[0], 1000);

        // Case 2: Both A (1000) and B (2500) are speaking ->
        // A gets 2500, B gets 1000, C gets 3500
        active.insert(addr_b, [2500.0f32; FRAME_SAMPLES]);
        let packets = compute_n_minus_1_packets(&recipients, &active);
        let pkt_map: HashMap<SocketAddr, [u8; PACKET_BYTES]> = packets.into_iter().collect();

        decode_frame(&pkt_map[&addr_a], &mut decoded);
        assert_eq!(decoded[0], 2500);

        decode_frame(&pkt_map[&addr_b], &mut decoded);
        assert_eq!(decoded[0], 1000);

        decode_frame(&pkt_map[&addr_c], &mut decoded);
        assert_eq!(decoded[0], 3500);
    }
}
