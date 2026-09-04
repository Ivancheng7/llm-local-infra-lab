use crate::safetensors_index::SafetensorsIndex;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub enum TensorModule {
    Embedding,
    LmHead,
    FullAttention,
    LinearAttention,
    Mlp,
    LayerNorm,
    Vision,
    Mtp,
    Unknown,
}

impl TensorModule {
    pub fn label(&self) -> &'static str {
        match self {
            TensorModule::Embedding => "embedding",
            TensorModule::LmHead => "lm_head",
            TensorModule::FullAttention => "full_attention",
            TensorModule::LinearAttention => "linear_attention",
            TensorModule::Mlp => "mlp",
            TensorModule::LayerNorm => "layernorm",
            TensorModule::Vision => "vision",
            TensorModule::Mtp => "mtp",
            TensorModule::Unknown => "unknown",
        }
    }
}

/// Classify a single tensor name based on the observed Qwen3.5 layout:
/// model.language_model.layers.N.<...>, model.visual.<...>, mtp.<...>,
/// model.language_model.embed_tokens.weight, lm_head.weight.
pub fn classify(name: &str) -> TensorModule {
    let parts: Vec<&str> = name.split('.').collect();
    if name == "lm_head.weight" {
        return TensorModule::LmHead;
    }
    if parts.len() < 3 {
        return TensorModule::Unknown;
    }
    match (parts[0], parts[1], parts.get(2).copied().unwrap_or("")) {
        ("model", "visual", _) => TensorModule::Vision,
        ("mtp", _, _) => TensorModule::Mtp,
        ("model", "language_model", "embed_tokens") => TensorModule::Embedding,
        ("model", "language_model", "norm") => TensorModule::LayerNorm,
        ("model", "language_model", "layers") => {
            let rest: Vec<&str> = parts[3..].to_vec();
            let within = match rest.first().copied() {
                Some(n) if n.parse::<u32>().is_ok() => &rest[1..],
                _ => rest.as_slice(),
            };
            let joined = within.join(".");
            if joined.starts_with("self_attn") || joined.starts_with("full_attn") {
                TensorModule::FullAttention
            } else if joined.starts_with("linear_attn")
                || joined.starts_with("linear_attention")
                || joined.starts_with("mamba")
            {
                TensorModule::LinearAttention
            } else if joined.starts_with("mlp") {
                TensorModule::Mlp
            } else if joined.contains("norm") || joined.ends_with("norm.weight") {
                TensorModule::LayerNorm
            } else {
                TensorModule::Unknown
            }
        }
        _ => TensorModule::Unknown,
    }
}

pub fn module_counts(index: &SafetensorsIndex) -> BTreeMap<TensorModule, usize> {
    let mut counts: BTreeMap<TensorModule, usize> = BTreeMap::new();
    for name in index.weight_map.keys() {
        *counts.entry(classify(name)).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_qwen35_names() {
        assert_eq!(
            classify("model.language_model.layers.0.self_attn.q_proj.weight"),
            TensorModule::FullAttention
        );
        assert_eq!(
            classify("model.language_model.layers.3.linear_attn.A_log"),
            TensorModule::LinearAttention
        );
        assert_eq!(
            classify("model.language_model.layers.3.mlp.gate_proj.weight"),
            TensorModule::Mlp
        );
        assert_eq!(
            classify("model.language_model.layers.3.input_layernorm.weight"),
            TensorModule::LayerNorm
        );
        assert_eq!(
            classify("model.visual.blocks.0.attn.qkv.weight"),
            TensorModule::Vision
        );
        assert_eq!(classify("mtp.fc.weight"), TensorModule::Mtp);
        assert_eq!(classify("lm_head.weight"), TensorModule::LmHead);
        assert_eq!(
            classify("model.language_model.embed_tokens.weight"),
            TensorModule::Embedding
        );
        assert_eq!(classify("totally.bogus.name"), TensorModule::Unknown);
    }
}
