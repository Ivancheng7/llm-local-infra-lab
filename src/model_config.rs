use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct RawTop {
    architectures: Option<Vec<String>>,
    model_type: Option<String>,
    text_config: RawText,
    #[serde(default)]
    vision_config: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct RawText {
    hidden_size: i64,
    num_hidden_layers: i64,
    #[serde(default)]
    layer_types: Vec<String>,
    num_attention_heads: i64,
    #[serde(default)]
    num_key_value_heads: Option<i64>,
    head_dim: i64,
    intermediate_size: i64,
    vocab_size: i64,
    #[serde(default)]
    max_position_embeddings: Option<i64>,
    // Kept for future module breakdown; not used by v1 estimates.
    #[allow(dead_code)]
    #[serde(default)]
    full_attention_interval: Option<i64>,
    #[serde(default)]
    mtp_num_hidden_layers: i64,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    dtype: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub model_type: String,
    pub hidden_size: i64,
    pub num_layers: i64,
    #[allow(dead_code)]
    pub layer_types: Vec<LayerKind>,
    pub num_attention_heads: i64,
    pub num_key_value_heads: i64,
    pub head_dim: i64,
    pub intermediate_size: i64,
    pub vocab_size: i64,
    pub max_position_embeddings: i64,
    pub full_attention_layers: i64,
    pub linear_attention_layers: i64,
    pub mtp_layers: i64,
    pub tie_word_embeddings: bool,
    pub declared_dtype: String,
    pub has_vision: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    FullAttention,
    LinearAttention,
    Other,
}

impl ModelConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_str(&raw)
    }

    pub fn from_str(raw: &str) -> Result<Self> {
        let raw: RawTop = serde_json::from_str(raw)?;
        let t = raw.text_config;
        if t.layer_types.len() != t.num_hidden_layers as usize {
            bail!(
                "layer_types has {} entries but num_hidden_layers = {}",
                t.layer_types.len(),
                t.num_hidden_layers
            );
        }
        let layer_types: Vec<LayerKind> = t
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" => LayerKind::FullAttention,
                "linear_attention" => LayerKind::LinearAttention,
                _ => LayerKind::Other,
            })
            .collect();
        Ok(Self {
            architecture: raw
                .architectures
                .and_then(|a| a.into_iter().next())
                .unwrap_or_else(|| "unknown".into()),
            model_type: raw.model_type.unwrap_or_else(|| "unknown".into()),
            hidden_size: t.hidden_size,
            num_layers: t.num_hidden_layers,
            full_attention_layers: layer_types
                .iter()
                .filter(|k| **k == LayerKind::FullAttention)
                .count() as i64,
            linear_attention_layers: layer_types
                .iter()
                .filter(|k| **k == LayerKind::LinearAttention)
                .count() as i64,
            layer_types,
            num_attention_heads: t.num_attention_heads,
            num_key_value_heads: t.num_key_value_heads.unwrap_or(t.num_attention_heads),
            head_dim: t.head_dim,
            intermediate_size: t.intermediate_size,
            vocab_size: t.vocab_size,
            max_position_embeddings: t.max_position_embeddings.unwrap_or(0),
            mtp_layers: t.mtp_num_hidden_layers,
            tie_word_embeddings: t.tie_word_embeddings.unwrap_or(false),
            declared_dtype: t.dtype.unwrap_or_else(|| "unknown".into()),
            has_vision: raw.vision_config.is_some(),
        })
    }
}
