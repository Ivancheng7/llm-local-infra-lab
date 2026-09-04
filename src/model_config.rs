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
    // Linear-attention (gated delta net) geometry
    #[serde(default)]
    linear_num_key_heads: Option<i64>,
    #[serde(default)]
    linear_num_value_heads: Option<i64>,
    #[serde(default)]
    linear_key_head_dim: Option<i64>,
    #[serde(default)]
    linear_value_head_dim: Option<i64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct VisionConfig {
    pub depth: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub out_hidden_size: i64,
    pub num_position_embeddings: i64,
    pub patch_size: i64,
    pub spatial_merge_size: i64,
    pub in_channels: i64,
    pub temporal_patch_size: i64,
}

#[derive(Deserialize)]
struct RawVision {
    depth: i64,
    hidden_size: i64,
    intermediate_size: i64,
    #[serde(default)]
    out_hidden_size: Option<i64>,
    #[serde(default)]
    num_position_embeddings: i64,
    #[serde(default)]
    patch_size: i64,
    #[serde(default)]
    spatial_merge_size: i64,
    #[serde(default)]
    in_channels: i64,
    #[serde(default)]
    temporal_patch_size: i64,
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
    /// Linear-attention head geometry (None when the model has no linear layers).
    pub linear: Option<LinearGeometry>,
    pub vision: Option<VisionConfig>,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearGeometry {
    pub num_key_heads: i64,
    pub num_value_heads: i64,
    pub key_head_dim: i64,
    pub value_head_dim: i64,
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
            linear: if t.linear_num_key_heads.is_some() {
                Some(LinearGeometry {
                    num_key_heads: t.linear_num_key_heads.unwrap_or(0),
                    num_value_heads: t.linear_num_value_heads.unwrap_or(0),
                    key_head_dim: t.linear_key_head_dim.unwrap_or(0),
                    value_head_dim: t.linear_value_head_dim.unwrap_or(0),
                })
            } else {
                None
            },
            vision: raw.vision_config.map(|v| {
                let v: RawVision = serde_json::from_value(v).unwrap_or(RawVision {
                    depth: 0,
                    hidden_size: 0,
                    intermediate_size: 0,
                    out_hidden_size: None,
                    num_position_embeddings: 0,
                    patch_size: 0,
                    spatial_merge_size: 0,
                    in_channels: 0,
                    temporal_patch_size: 0,
                });
                VisionConfig {
                    depth: v.depth,
                    hidden_size: v.hidden_size,
                    intermediate_size: v.intermediate_size,
                    out_hidden_size: v.out_hidden_size.unwrap_or(v.hidden_size),
                    num_position_embeddings: v.num_position_embeddings,
                    patch_size: v.patch_size,
                    spatial_merge_size: v.spatial_merge_size,
                    in_channels: v.in_channels,
                    temporal_patch_size: v.temporal_patch_size,
                }
            }),
        })
    }
}
