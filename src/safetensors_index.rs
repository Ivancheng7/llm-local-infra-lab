use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Deserialize)]
struct RawIndex {
    weight_map: BTreeMap<String, String>,
    #[serde(default)]
    metadata: Option<RawIndexMetadata>,
}

#[derive(Deserialize)]
struct RawIndexMetadata {
    // HF writes total_size as a JSON number that some models emit as float.
    total_size: Option<serde_json::Number>,
}

impl RawIndexMetadata {
    fn total_size_i64(&self) -> i64 {
        match &self.total_size {
            Some(n) => n
                .as_i64()
                .unwrap_or_else(|| n.as_f64().unwrap_or(0.0) as i64),
            None => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SafetensorsIndex {
    /// tensor name -> shard file name
    pub weight_map: BTreeMap<String, String>,
    pub total_size: i64,
}

impl SafetensorsIndex {
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let raw: RawIndex = serde_json::from_str(&raw)?;
        let total_size = raw.metadata.map(|m| m.total_size_i64()).unwrap_or(0);
        if total_size <= 0 {
            anyhow::bail!("index metadata.total_size missing or non-positive");
        }
        Ok(Self {
            weight_map: raw.weight_map,
            total_size,
        })
    }

    pub fn shard_count(&self) -> usize {
        self.weight_map
            .values()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }
}
