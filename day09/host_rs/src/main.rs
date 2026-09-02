// main.rs - Day 9: Rust host launching the same fused CUDA kernel as
// host_py.py, via a plain C FFI into libfused.so (see ../../shim.cu).
//
// Same two experiments, same synthetic rolling frame pattern, same
// RESULT_JSON schema as host_py.py -- see that file for the full
// experiment description. std only, no external crates (see
// Cargo.toml / build.rs). No unsafe beyond the four FFI calls.

use std::os::raw::{c_float, c_int, c_uchar};
use std::thread::sleep;
use std::time::{Duration, Instant};

extern "C" {
    fn fused_init(src_w: c_int, src_h: c_int, dst_w: c_int, dst_h: c_int) -> c_int;
    fn fused_upload(host_hwc: *const c_uchar) -> c_int;
    fn fused_run() -> c_int;
    #[allow(dead_code)] // only used by parity checks, not this benchmark
    fn fused_download(host_chw: *mut c_float) -> c_int;
    fn fused_free();
}

const SRC_W: i32 = 1280;
const SRC_H: i32 = 720;
const DST_W: i32 = 640;
const DST_H: i32 = 480;
const CH: i32 = 3;

const TIGHT_ITERS: usize = 1000;
const TIGHT_WARMUP: usize = 50;

const PACED_TICKS: usize = 600;
const PACED_HZ: f64 = 30.0;
const LATE_THRESHOLD_US: f64 = 2000.0; // 2 ms
const SPIN_TAIL_S: f64 = 0.0005; // sleep to within 0.5 ms, then spin

/// Frame-0 term of the rolling synthetic pattern:
///   value(frame, y, x, c) = (x*7 + y*13 + c*29 + frame) mod 256
/// Precomputed once; each frame adds its index and re-mods (make_frame).
fn build_base_pattern() -> Vec<i32> {
    let mut base = vec![0i32; (SRC_H * SRC_W * CH) as usize];
    for y in 0..SRC_H {
        for x in 0..SRC_W {
            for c in 0..CH {
                let idx = ((y * SRC_W + x) * CH + c) as usize;
                base[idx] = (x * 7 + y * 13 + c * 29) % 256;
            }
        }
    }
    base
}

/// Rolling frame, NOT timed (built before the timed upload+run region).
fn make_frame(base: &[i32], frame_idx: i32) -> Vec<u8> {
    base.iter().map(|&v| ((v + frame_idx) % 256) as u8).collect()
}

fn percentile(sorted_vals: &[f64], p: f64) -> f64 {
    if sorted_vals.is_empty() {
        return 0.0;
    }
    let idx = (p * ((sorted_vals.len() - 1) as f64)) as usize;
    sorted_vals[idx]
}

struct Stats {
    mean_us: f64,
    p50_us: f64,
    p99_us: f64,
    max_us: f64,
}

fn compute_stats(samples: &[f64]) -> Stats {
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = if sorted.is_empty() {
        0.0
    } else {
        sorted.iter().sum::<f64>() / sorted.len() as f64
    };
    Stats {
        mean_us: mean,
        p50_us: percentile(&sorted, 0.50),
        p99_us: percentile(&sorted, 0.99),
        max_us: *sorted.last().unwrap_or(&0.0),
    }
}

fn main() {
    let t0 = Instant::now();

    let rc = unsafe { fused_init(SRC_W, SRC_H, DST_W, DST_H) };
    if rc != 0 {
        eprintln!("host_rs: fused_init failed, rc={}", rc);
        std::process::exit(1);
    }

    let base = build_base_pattern();

    // ---- Experiment (i): tight loop ----
    let mut tight_samples_us: Vec<f64> = Vec::with_capacity(TIGHT_ITERS);
    let mut startup_ms: Option<f64> = None;
    for i in 0..TIGHT_ITERS {
        let frame = make_frame(&base, i as i32);

        // Error checks happen AFTER t_end, not inline -- keeping the branch
        // out of the timed region matches host_py.py, which stashes its
        // up_rc/run_rc the same way so both hosts time an identical region
        // (just the two FFI calls), with no built-in bias either direction.
        let t_start = Instant::now();
        let up_rc = unsafe { fused_upload(frame.as_ptr()) };
        let run_rc = unsafe { fused_run() };
        let t_end = Instant::now();

        if up_rc != 0 || run_rc != 0 {
            eprintln!("host_rs: upload/run failed at iter {}", i);
            std::process::exit(1);
        }
        if startup_ms.is_none() {
            startup_ms = Some(t_end.duration_since(t0).as_secs_f64() * 1000.0);
        }
        tight_samples_us.push(t_end.duration_since(t_start).as_secs_f64() * 1e6);
    }
    let tight_stats = compute_stats(&tight_samples_us[TIGHT_WARMUP..]);

    // ---- Experiment (ii): paced 30 Hz loop ----
    let mut jitter_samples_us: Vec<f64> = Vec::with_capacity(PACED_TICKS);
    let mut work_samples_us: Vec<f64> = Vec::with_capacity(PACED_TICKS);
    let mut late_ticks: u32 = 0;
    let paced_start = Instant::now();
    for i in 0..PACED_TICKS {
        let target_s = i as f64 / PACED_HZ;

        loop {
            let now_s = paced_start.elapsed().as_secs_f64();
            let remaining = target_s - now_s - SPIN_TAIL_S;
            if remaining > 0.0 {
                sleep(Duration::from_secs_f64(remaining));
            } else {
                break;
            }
        }
        while paced_start.elapsed().as_secs_f64() < target_s {} // spin the tail

        let tick_start_s = paced_start.elapsed().as_secs_f64();
        let jitter_us = (tick_start_s - target_s) * 1e6;
        jitter_samples_us.push(jitter_us);
        if jitter_us > LATE_THRESHOLD_US {
            late_ticks += 1;
        }

        let frame = make_frame(&base, TIGHT_ITERS as i32 + i as i32);
        let w0 = Instant::now();
        let up_rc = unsafe { fused_upload(frame.as_ptr()) };
        let run_rc = unsafe { fused_run() };
        let w1 = Instant::now();
        if up_rc != 0 || run_rc != 0 {
            eprintln!("host_rs: paced-loop upload/run failed at tick {}", i);
            std::process::exit(1);
        }
        work_samples_us.push(w1.duration_since(w0).as_secs_f64() * 1e6);
    }
    let jitter_stats = compute_stats(&jitter_samples_us);
    let work_mean_us = if work_samples_us.is_empty() {
        0.0
    } else {
        work_samples_us.iter().sum::<f64>() / work_samples_us.len() as f64
    };

    unsafe { fused_free() };

    let startup_ms_val = startup_ms.unwrap_or(0.0);
    println!(
        "RESULT_JSON: {{\"host\": \"rust\", \"startup_ms\": {:.3}, \
\"tight_loop\": {{\"iters\": {}, \"warmup_discarded\": {}, \"mean_us\": {:.3}, \
\"p50_us\": {:.3}, \"p99_us\": {:.3}, \"max_us\": {:.3}}}, \
\"paced_loop\": {{\"ticks\": {}, \"hz\": {:.1}, \"jitter_p50_us\": {:.3}, \
\"jitter_p99_us\": {:.3}, \"jitter_max_us\": {:.3}, \"late_ticks_over_2ms\": {}, \
\"work_mean_us\": {:.3}}}}}",
        startup_ms_val,
        TIGHT_ITERS,
        TIGHT_WARMUP,
        tight_stats.mean_us,
        tight_stats.p50_us,
        tight_stats.p99_us,
        tight_stats.max_us,
        PACED_TICKS,
        PACED_HZ,
        jitter_stats.p50_us,
        jitter_stats.p99_us,
        jitter_stats.max_us,
        late_ticks,
        work_mean_us
    );
}
