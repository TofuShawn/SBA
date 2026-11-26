//! Training-record IO: one safetensors file per game (spec §5.3).
//! Tensor names are part of the Python contract: states/policy/z, plus a
//! __metadata__ map the Python loader and the resume logic read.

use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Wraps an f32 slice as a `TensorView` of the given shape (safetensors 0.4:
/// `TensorView::new(dtype, shape, data)` — the name comes from the serialize
/// map key, not from the view).
fn f32_view(data: &[f32], dims: Vec<usize>) -> Result<TensorView<'_>, String> {
    TensorView::new(safetensors::Dtype::F32, dims, bytemuck::cast_slice(data))
        .map_err(|e| e.to_string())
}

pub struct RecordMeta {
    pub game_type: String,
    pub seed: u64,
    pub result: String,
    pub sims: u32,
    pub weights_sha: String,
}

pub struct GameRecord {
    pub meta: RecordMeta,
    pub states: Vec<f32>, // T * 324
    pub policy: Vec<f32>, // T * 81
    pub z: Vec<f32>,      // T
}

impl GameRecord {
    pub fn moves(&self) -> usize {
        self.z.len()
    }

    pub fn write(&self, dir: &Path, game_idx: usize) -> Result<PathBuf, String> {
        let t = self.moves();
        debug_assert_eq!(self.states.len(), t * 324);
        debug_assert_eq!(self.policy.len(), t * 81);
        let mut data = HashMap::new();
        data.insert(
            "states".to_string(),
            f32_view(&self.states, vec![t, 4, 9, 9])?,
        );
        data.insert("policy".to_string(), f32_view(&self.policy, vec![t, 81])?);
        data.insert("z".to_string(), f32_view(&self.z, vec![t])?);
        let meta: HashMap<String, String> = HashMap::from([
            ("format_version".to_string(), "1".to_string()),
            ("game_type".to_string(), self.meta.game_type.clone()),
            ("seed".to_string(), self.meta.seed.to_string()),
            ("result".to_string(), self.meta.result.clone()),
            ("sims".to_string(), self.meta.sims.to_string()),
            ("weights_sha".to_string(), self.meta.weights_sha.clone()),
        ]);
        // Byte determinism: serialize without the __metadata__ map (HashMap order is
        // randomized), then splice it back key-sorted so same-seed records match bitwise.
        let blob = safetensors::serialize(data, &None).map_err(|e| e.to_string())?;
        let (header_len, _) =
            safetensors::SafeTensors::read_metadata(&blob).map_err(|e| e.to_string())?;
        let mut header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&blob[8..8 + header_len]).map_err(|e| e.to_string())?;
        let mut meta_sorted: Vec<(String, String)> = meta.into_iter().collect();
        meta_sorted.sort();
        let mut md = serde_json::Map::new();
        for (k, v) in meta_sorted {
            md.insert(k, serde_json::Value::String(v));
        }
        header.insert("__metadata__".to_string(), serde_json::Value::Object(md));
        let mut header_bytes =
            serde_json::to_vec(&serde_json::Value::Object(header)).map_err(|e| e.to_string())?;
        // force the 8-byte header alignment safetensors requires
        let pad = (8 - header_bytes.len() % 8) % 8;
        header_bytes.resize(header_bytes.len() + pad, b' ');
        let mut out = (header_bytes.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header_bytes);
        // the tensor data section rides over unchanged (offsets are
        // relative to the end of the header)
        out.extend_from_slice(&blob[8 + header_len..]);
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("g{game_idx:05}.safetensors"));
        fs::write(&path, &out).map_err(|e| e.to_string())?;
        Ok(path)
    }
}

pub fn read_record(path: &Path) -> Result<GameRecord, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| e.to_string())?;
    // safetensors 0.4 has no SafeTensors::metadata(); the __metadata__ map is
    // reached through read_metadata + Metadata::metadata().
    let md: HashMap<String, String> = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| e.to_string())?
        .1
        .metadata()
        .clone()
        .unwrap_or_default();
    let f32s = |name: &str| -> Result<Vec<f32>, String> {
        let v = st.tensor(name).map_err(|e| e.to_string())?;
        assert_eq!(v.dtype(), safetensors::Dtype::F32, "{name}");
        Ok(bytemuck::cast_slice(v.data()).to_vec())
    };
    Ok(GameRecord {
        meta: RecordMeta {
            game_type: md.get("game_type").cloned().unwrap_or_default(),
            seed: md.get("seed").and_then(|s| s.parse().ok()).unwrap_or(0),
            result: md.get("result").cloned().unwrap_or_default(),
            sims: md.get("sims").and_then(|s| s.parse().ok()).unwrap_or(0),
            weights_sha: md.get("weights_sha").cloned().unwrap_or_default(),
        },
        states: f32s("states")?,
        policy: f32s("policy")?,
        z: f32s("z")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> GameRecord {
        GameRecord {
            meta: RecordMeta {
                game_type: "ultimate".into(),
                seed: 42,
                result: "X".into(),
                sims: 8,
                weights_sha: "abc".into(),
            },
            states: vec![0.5; 2 * 324],
            policy: vec![0.25; 2 * 81],
            z: vec![0.5, -0.5],
        }
    }

    #[test]
    fn write_read_round_trip() {
        let dir = std::env::temp_dir().join("sba_core_records_test");
        std::fs::create_dir_all(&dir).unwrap();
        let rec = sample();
        let path = rec.write(&dir, 7).unwrap();
        assert!(path.ends_with("g00007.safetensors"));
        let back = read_record(&path).unwrap();
        assert_eq!(back.states, rec.states);
        assert_eq!(back.policy, rec.policy);
        assert_eq!(back.z, rec.z);
        assert_eq!(back.meta.seed, 42);
        assert_eq!(back.meta.game_type, "ultimate");
    }

    #[test]
    fn format_version_is_1() {
        let dir = std::env::temp_dir().join("sba_core_records_test2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = sample().write(&dir, 0).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        // safetensors 0.4 has no SafeTensors::metadata(); use read_metadata.
        let md = safetensors::SafeTensors::read_metadata(&bytes)
            .unwrap()
            .1
            .metadata()
            .clone()
            .unwrap_or_default();
        assert_eq!(md.get("format_version").map(String::as_str), Some("1"));
        assert_eq!(st.tensor("states").unwrap().shape(), &[2, 4, 9, 9]);
        assert_eq!(st.tensor("policy").unwrap().shape(), &[2, 81]);
        assert_eq!(st.tensor("z").unwrap().shape(), &[2]);
    }
}
