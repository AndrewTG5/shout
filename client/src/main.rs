#![windows_subsystem = "windows"]

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    BufferSize, SampleFormat, StreamConfig, SupportedBufferSize, SupportedStreamConfigRange,
};
use protocol::{
    FRAME_SAMPLES, PACKET_BYTES, SAMPLE_RATE, SERVER_BIND_ADDR, decode_frame, encode_frame,
};
use rtrb::RingBuffer;
use std::net::UdpSocket;
use std::thread;

/// Target hardware callback size: 240 frames = 5ms at 48kHz.
const TARGET_BUFFER_FRAMES: u32 = 240;
/// Ring buffer capacity: 2 network frames = 960 samples (20ms).
const RING_BUFFER_SAMPLES: usize = FRAME_SAMPLES * 2;

fn main() -> Result<()> {
    let host = cpal::default_host();

    let input_device = host
        .default_input_device()
        .context("No default audio input device")?;
    let output_device = host
        .default_output_device()
        .context("No default audio output device")?;

    let in_cfg = select_f32_48k_config(
        input_device
            .supported_input_configs()
            .context("Failed to query input configs")?,
        1,
    )
    .context("Input device does not support 48kHz f32")?;

    let out_cfg = select_f32_48k_config(
        output_device
            .supported_output_configs()
            .context("Failed to query output configs")?,
        2,
    )
    .context("Output device does not support 48kHz f32")?;

    let in_channels = in_cfg.channels as usize;
    let out_channels = out_cfg.channels as usize;

    let (mut capture_producer, mut capture_consumer) = RingBuffer::<i16>::new(RING_BUFFER_SAMPLES);
    let (mut playback_producer, mut playback_consumer) =
        RingBuffer::<i16>::new(RING_BUFFER_SAMPLES);

    let socket = UdpSocket::bind("0.0.0.0:0").context("Failed to bind UDP socket")?;
    socket
        .connect(SERVER_BIND_ADDR)
        .context("Failed to connect UDP socket")?;
    let rx_socket = socket.try_clone().context("Failed to clone UDP socket")?;

    let tx_thread = thread::current();

    let _input_stream = input_device
        .build_input_stream(
            in_cfg,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let inv_channels = 1.0 / (in_channels as f32);
                for frame in data.chunks_exact(in_channels) {
                    let mono = frame.iter().copied().sum::<f32>() * inv_channels;
                    let sample_i16 = (mono.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    let _ = capture_producer.push(sample_i16);
                }
                if capture_producer.slots() <= RING_BUFFER_SAMPLES - FRAME_SAMPLES {
                    tx_thread.unpark();
                }
            },
            |_| {},
            None,
        )
        .context("Failed to build input stream")?;

    let _output_stream = output_device
        .build_output_stream(
            out_cfg,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for frame in data.chunks_exact_mut(out_channels) {
                    let sample_f32 = match playback_consumer.pop() {
                        Ok(s) => s as f32 * (1.0 / 32768.0),
                        Err(_) => 0.0,
                    };
                    frame.fill(sample_f32);
                }
            },
            |_| {},
            None,
        )
        .context("Failed to build output stream")?;

    _input_stream.play().context("Failed to start input stream")?;
    _output_stream.play().context("Failed to start output stream")?;

    thread::spawn(move || {
        let mut recv_buf = [0u8; PACKET_BYTES];
        let mut pcm_frame = [0i16; FRAME_SAMPLES];

        loop {
            if let Ok(PACKET_BYTES) = rx_socket.recv(&mut recv_buf) {
                decode_frame(&recv_buf, &mut pcm_frame);
                if let Ok(chunk) = playback_producer.write_chunk_uninit(FRAME_SAMPLES) {
                    chunk.fill_from_iter(pcm_frame);
                } else {
                    for &s in &pcm_frame {
                        let _ = playback_producer.push(s);
                    }
                }
            }
        }
    });

    let mut pcm_frame = [0i16; FRAME_SAMPLES];
    let mut packet = [0u8; PACKET_BYTES];

    loop {
        while let Ok(chunk) = capture_consumer.read_chunk(FRAME_SAMPLES) {
            let (first, second) = chunk.as_slices();
            pcm_frame[..first.len()].copy_from_slice(first);
            pcm_frame[first.len()..].copy_from_slice(second);
            chunk.commit_all();

            encode_frame(&pcm_frame, &mut packet);
            let _ = socket.send(&packet);
        }
        thread::park();
    }
}

fn select_f32_48k_config(
    ranges: impl Iterator<Item = SupportedStreamConfigRange>,
    preferred_channels: u16,
) -> Option<StreamConfig> {
    ranges
        .filter(|r| {
            r.sample_format() == SampleFormat::F32
                && r.min_sample_rate() <= SAMPLE_RATE
                && SAMPLE_RATE <= r.max_sample_rate()
        })
        .min_by_key(|r| r.channels().abs_diff(preferred_channels))
        .map(|r| {
            let buffer_size = match r.buffer_size() {
                SupportedBufferSize::Range { min, max } => {
                    BufferSize::Fixed(TARGET_BUFFER_FRAMES.clamp(*min, *max))
                }
                SupportedBufferSize::Unknown => BufferSize::Default,
            };
            let mut cfg: StreamConfig = r.with_sample_rate(SAMPLE_RATE).into();
            cfg.buffer_size = buffer_size;
            cfg
        })
}
