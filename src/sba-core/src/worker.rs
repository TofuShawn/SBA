//! Batched inference worker: game threads send leaf requests over mpsc; one thread
//! owns the model, packs requests into buckets {16, 64, 256} (or a multiple of 256)
//! and flushes when a bucket fills or the 2 ms deadline expires. Padded rows are
//! split back to their requests over a reply channel. A request may carry many rows.
//! CONTRACT: replies are RAW LOGITS; softmax happens in search, never here.

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
    /// k leaf rows riding ONE channel message: `evaluate` sends k = 1; the
    /// wave search's `evaluate_batch` submits a whole wave so the worker's
    /// bucket fills in a single send (mirrors python mcts_search's one
    /// batched forward per round).
    planes: Vec<[f32; 324]>,
    reply: SyncSender<Vec<(Vec<f32>, f32)>>,
}

/// Cloneable per-game-thread handle: sends one request to the worker and
/// waits on a per-request rendezvous channel for the reply.
#[derive(Clone)]
pub struct WorkerHandle {
    tx: Arc<Mutex<Sender<Request>>>,
}

impl search::Evaluator for WorkerHandle {
    fn evaluate(&mut self, planes: [f32; 324]) -> Result<(Vec<f32>, f32), String> {
        let mut rows = self.evaluate_batch(vec![planes])?;
        if rows.len() != 1 {
            return Err(format!(
                "worker replied {} rows for a 1-row submit",
                rows.len()
            ));
        }
        // Contract: the reply row is the network's RAW logit row (no softmax).
        Ok(rows.remove(0))
    }

    /// Bulk path: ALL k rows ride one channel message into the worker's
    /// bucket (a full 256-row wave flushes immediately; a partial wave
    /// still answers within the 2 ms deadline), one padded forward, the
    /// reply split back per row in input order. RAW LOGITS contract as in
    /// evaluate — search applies softmax.
    fn evaluate_batch(&mut self, planes: Vec<[f32; 324]>) -> Result<Vec<(Vec<f32>, f32)>, String> {
        let (rtx, rrx) = sync_channel(0); // rendezvous: the reply send IS the hand-off
        self.tx
            .lock()
            .map_err(|_| "worker tx poisoned")?
            .send(Request { planes, reply: rtx })
            .map_err(|_| "worker gone")?;
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
    let mut held_rows = 0usize; // total REAL rows across pending (a bulk request carries many)
    loop {
        // Idle: block for the first request of the next batch — no busy-spin.
        // A disconnect with an empty queue means every handle is dropped and
        // nothing is owed a reply: shut down.
        while pending.is_empty() {
            match rx.recv() {
                Ok(r) => {
                    held_rows += r.planes.len();
                    pending.push(r);
                }
                Err(_) => return,
            }
        }
        // Accumulate toward the next bucket but never wait past the 2 ms deadline once
        // holding a row; a full bucket flushes at once, and queued requests survive a
        // disconnect until the queue drains.
        let deadline = Instant::now() + MAX_WAIT;
        while held_rows < MAX_BUCKET {
            if held_rows >= next_bucket(held_rows) {
                break; // bucket filled exactly — run it now
            }
            let budget = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(budget) {
                Ok(r) => {
                    held_rows += r.planes.len();
                    pending.push(r);
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        run_batch(&model, &device, &mut pending, &stats);
        held_rows = 0; // run_batch drained every request it was handed
    }
}

/// Pad `pending` to its bucket, run one forward, reply per row, count stats.
/// A request may carry many rows (the bulk wave path) — its reply carries
/// the same rows back, in order.
fn run_batch<B: Backend>(
    model: &AzNet<B>,
    device: &B::Device,
    pending: &mut Vec<Request>,
    stats: &WorkerStats,
) {
    let n_real: usize = pending.iter().map(|r| r.planes.len()).sum(); // BEFORE the drain
    let b = next_bucket(n_real);
    let mut flat = vec![0.0f32; b * 324];
    let mut offsets = Vec::with_capacity(pending.len());
    let mut row = 0usize;
    for r in pending.iter() {
        offsets.push(row);
        for p in &r.planes {
            flat[row * 324..(row + 1) * 324].copy_from_slice(p);
            row += 1;
        }
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
        let start = offsets[i];
        let rows: Vec<(Vec<f32>, f32)> = (0..r.planes.len())
            .map(|j| {
                (
                    logits[(start + j) * 81..(start + j + 1) * 81].to_vec(),
                    values[start + j],
                )
            })
            .collect();
        // A requester that died before recv makes this send fail — nothing to
        // do; the worker must keep serving the rest regardless.
        let _ = r.reply.send(rows);
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
        assert!(stats.real_rows.load(std::sync::atomic::Ordering::Relaxed) >= 300);
        assert!(stats.batches.load(std::sync::atomic::Ordering::Relaxed) >= 2);
    }

    #[test]
    fn bulk_wave_is_one_message_one_forward() {
        // The wave search's submit shape: a full-bucket bulk (256 rows) rides
        // ONE channel message and runs ONE padded forward — no bucket
        // splitting, no deadline wait — and the replies align per row with
        // the direct single-row forwards (RAW LOGITS on both sides).
        type B = DefaultBackend;
        let device = default_device();
        let model = AzNet::<B>::new(&device, 8, 1);
        let (mut handle, stats) = spawn(model.clone(), device.clone());
        let planes: Vec<[f32; 324]> = (0..256u32)
            .map(|i| std::array::from_fn(|j| ((i * 31 + j as u32) % 97) as f32 / 97.0))
            .collect();
        let rows = search::Evaluator::evaluate_batch(&mut handle, planes.clone()).unwrap();
        assert_eq!(rows.len(), 256);
        assert_eq!(stats.batches.load(Ordering::Relaxed), 1);
        assert_eq!(stats.real_rows.load(Ordering::Relaxed), 256);
        for i in [0usize, 1, 127, 255] {
            let x = Tensor::<B, 4>::from_data(
                TensorData::new(planes[i].to_vec(), shape(&[1, 4, 9, 9])),
                &device,
            );
            let (p, v) = model.forward(x);
            let (dp, dv) = (
                p.into_data().to_vec::<f32>().unwrap(),
                v.into_data().to_vec::<f32>().unwrap()[0],
            );
            let err = rows[i]
                .0
                .iter()
                .zip(&dp)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(err < 1e-5, "row {i} policy err {err}");
            assert!((rows[i].1 - dv).abs() < 1e-5);
        }
    }

    #[test]
    fn bulk_subbucket_stays_one_forward() {
        // A partial wave (100 rows < the 256 bucket) still rides ONE message
        // and lands in exactly ONE padded forward after the deadline — the
        // bucket logic counts ROWS now, not requests.
        type B = DefaultBackend;
        let device = default_device();
        let model = AzNet::<B>::new(&device, 8, 1);
        let (mut handle, stats) = spawn(model, device);
        let planes: Vec<[f32; 324]> = (0..100u32)
            .map(|i| std::array::from_fn(|j| ((i * 17 + j as u32) % 89) as f32 / 89.0))
            .collect();
        let rows = search::Evaluator::evaluate_batch(&mut handle, planes).unwrap();
        assert_eq!(rows.len(), 100);
        assert_eq!(stats.batches.load(Ordering::Relaxed), 1);
        assert_eq!(stats.real_rows.load(Ordering::Relaxed), 100);
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
