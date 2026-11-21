//! Batched inference worker (spec §3): game threads push leaf requests on an
//! mpsc channel; one worker thread owns the model and accumulates requests
//! into bucket-sized batches — {16, 64, 256} rows, or a multiple of 256
//! beyond — flushing when a bucket fills or a 2 ms deadline expires,
//! whichever comes first. Zero rows pad each batch to its bucket size; one
//! padded AzNet forward runs and the outputs are split back per request
//! over a rendezvous reply channel.
//!
//! CONTRACT: replies carry RAW LOGITS — no softmax in the worker; search
//! applies softmax (mirror: Python applies softmax inside mcts_search after
//! the forward, not in the network).

use crate::net::AzNet;
use crate::search;
use crate::shape;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Bucket sizes for padded batched inference (spec §3).
pub const BUCKETS: [usize; 3] = [16, 64, 256];

/// Largest nominal bucket; beyond it batches round up to multiples of 256.
const MAX_BUCKET: usize = BUCKETS[2];

/// Longest a partially-filled bucket is held before flushing (spec §3).
const MAX_WAIT: Duration = Duration::from_millis(2);

/// Batch counters for the exe's batch_stats events.
pub struct WorkerStats {
    pub batches: AtomicU64,
    pub real_rows: AtomicU64,
}

struct Request {
    planes: [f32; 324],
    reply: SyncSender<(Vec<f32>, f32)>,
}

/// Cloneable per-game-thread handle: sends one request to the worker and
/// waits on a per-request rendezvous channel for the reply.
#[derive(Clone)]
pub struct WorkerHandle {
    tx: Arc<Mutex<Sender<Request>>>,
}

impl search::Evaluator for WorkerHandle {
    fn evaluate(&mut self, planes: [f32; 324]) -> Result<(Vec<f32>, f32), String> {
        let (rtx, rrx) = sync_channel(0); // rendezvous: the reply send IS the hand-off
        self.tx
            .lock()
            .map_err(|_| "worker tx poisoned")?
            .send(Request { planes, reply: rtx })
            .map_err(|_| "worker gone")?;
        // Contract: the reply row is the network's RAW logit row (no softmax).
        rrx.recv().map_err(|_| String::from("worker dropped reply"))
    }
}

/// Spawn the worker thread owning `model`. Returns a cloneable handle for
/// the game threads plus the shared batch counters. Dropping every handle
/// shuts the worker down cleanly.
pub fn spawn<B: Backend>(model: AzNet<B>, device: B::Device) -> (WorkerHandle, Arc<WorkerStats>) {
    let (tx, rx) = channel::<Request>();
    let stats = Arc::new(WorkerStats {
        batches: AtomicU64::new(0),
        real_rows: AtomicU64::new(0),
    });
    let st2 = Arc::clone(&stats);
    std::thread::Builder::new()
        .name("sba-inference".into())
        .spawn(move || worker_loop(model, device, rx, st2))
        .expect("spawn inference worker");
    (
        WorkerHandle {
            tx: Arc::new(Mutex::new(tx)),
        },
        stats,
    )
}

/// Smallest bucket >= n; beyond the buckets, round up to a multiple of 256.
fn next_bucket(n: usize) -> usize {
    for &b in &BUCKETS {
        if n <= b {
            return b;
        }
    }
    n.div_ceil(256) * 256
}

fn worker_loop<B: Backend>(
    model: AzNet<B>,
    device: B::Device,
    rx: Receiver<Request>,
    stats: Arc<WorkerStats>,
) {
    let mut pending: Vec<Request> = Vec::new();
    loop {
        // Idle: block for the first row of the next batch — no busy-spin.
        // A disconnect with an empty queue means every handle is dropped and
        // nothing is owed a reply: shut down.
        while pending.is_empty() {
            match rx.recv() {
                Ok(r) => pending.push(r),
                Err(_) => return,
            }
        }
        // Accumulate toward the next bucket, but never wait past the 2 ms
        // deadline once we hold at least one row (latency bound). A disconnect
        // only surfaces once the queue is drained — requests sent before the
        // senders dropped stay receivable — so the flush below still answers
        // every accepted request.
        let deadline = Instant::now() + MAX_WAIT;
        while pending.len() < MAX_BUCKET {
            if pending.len() >= next_bucket(pending.len()) {
                break; // bucket filled exactly — run it now
            }
            let budget = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(budget) {
                Ok(r) => pending.push(r),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        run_batch(&model, &device, &mut pending, &stats);
    }
}

/// Pad `pending` to its bucket, run one forward, reply per row, count stats.
fn run_batch<B: Backend>(
    model: &AzNet<B>,
    device: &B::Device,
    pending: &mut Vec<Request>,
    stats: &WorkerStats,
) {
    let n_real = pending.len(); // count BEFORE the drain — these are the real rows
    let b = next_bucket(n_real);
    let mut flat = vec![0.0f32; b * 324];
    for (i, r) in pending.iter().enumerate() {
        flat[i * 324..(i + 1) * 324].copy_from_slice(&r.planes);
    }
    let x = Tensor::<B, 4>::from_data(TensorData::new(flat, shape(&[b, 4, 9, 9])), device);
    // RAW LOGITS contract: no softmax here — search applies softmax to the
    // returned row (mirror: Python applies softmax inside mcts_search after
    // the forward, not in the network).
    let (logits, values) = model.forward(x);
    let logits = logits
        .into_data()
        .to_vec::<f32>()
        .expect("f32 policy logits");
    let values = values.into_data().to_vec::<f32>().expect("f32 values");
    for (i, r) in pending.drain(..).enumerate() {
        // A requester that died before recv makes this send fail — nothing to
        // do; the worker must keep serving the rest regardless.
        let _ = r.reply.send((logits[i * 81..(i + 1) * 81].to_vec(), values[i]));
    }
    stats.batches.fetch_add(1, Ordering::Relaxed);
    stats.real_rows.fetch_add(n_real as u64, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::AzNet;
    use crate::{default_device, DefaultBackend};

    #[test]
    fn replies_match_direct_forward() {
        type B = DefaultBackend;
        let device = default_device();
        let model = AzNet::<B>::new(&device, 8, 1); // small random net
        let (mut handle, _stats) = spawn(model.clone(), device.clone());
        // Direct single-row forward for comparison. RAW LOGITS on both sides:
        // the worker replies with the network's raw logit row (no softmax) —
        // search applies softmax81 to the evaluator's return itself.
        let direct = |planes: [f32; 324]| -> (Vec<f32>, f32) {
            let x = Tensor::<B, 4>::from_data(
                TensorData::new(planes.to_vec(), shape(&[1, 4, 9, 9])),
                &device,
            );
            let (p, v) = model.forward(x);
            (
                p.into_data().to_vec::<f32>().unwrap(),
                v.into_data().to_vec::<f32>().unwrap()[0],
            )
        };
        let mut planes = [0.0f32; 324];
        planes[0] = 1.0;
        planes[100] = 0.7;
        planes[323] = 0.2;
        let (wp, wv) = search::Evaluator::evaluate(&mut handle, planes).unwrap();
        let (dp, dv) = direct(planes);
        let err = wp
            .iter()
            .zip(&dp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 1e-5, "policy err {err}");
        assert!((wv - dv).abs() < 1e-5);
    }

    #[test]
    fn concurrent_requests_all_answered() {
        type B = DefaultBackend;
        let device = default_device();
        let model = AzNet::<B>::new(&device, 8, 1);
        let (handle, stats) = spawn(model, device);
        std::thread::scope(|s| {
            for t in 0..4u64 {
                let mut h = handle.clone();
                s.spawn(move || {
                    for i in 0..75u64 {
                        let seed = t * 1000 + i;
                        let planes =
                            std::array::from_fn(|j| ((seed * 31 + j as u64) % 97) as f32 / 97.0);
                        let (p, v) = search::Evaluator::evaluate(&mut h, planes).unwrap();
                        assert_eq!(p.len(), 81);
                        assert!((-1.0..=1.0).contains(&v));
                    }
                });
            }
        });
        assert!(
            stats
                .real_rows
                .load(std::sync::atomic::Ordering::Relaxed)
                >= 300
        );
        assert!(stats.batches.load(std::sync::atomic::Ordering::Relaxed) >= 2);
    }

    #[test]
    fn next_bucket_rounds_up() {
        // first bucket >= n; beyond 256, round up to a multiple of 256
        assert_eq!(next_bucket(0), 16);
        assert_eq!(next_bucket(1), 16);
        assert_eq!(next_bucket(15), 16);
        assert_eq!(next_bucket(16), 16);
        assert_eq!(next_bucket(17), 64);
        assert_eq!(next_bucket(64), 64);
        assert_eq!(next_bucket(65), 256);
        assert_eq!(next_bucket(256), 256);
        assert_eq!(next_bucket(257), 512);
        assert_eq!(next_bucket(300), 512);
    }
}
