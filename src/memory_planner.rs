use crate::model_config::ModelConfig;
use crate::safetensors_index::SafetensorsIndex;
use crate::tensor_classifier::{self, TensorModule};
use crate::Precision;
use std::collections::BTreeMap;

/// Bytes per parameter for each supported target precision.
pub fn bytes_per_param(p: Precision) -> f64 {
    match p {
        Precision::Bf16 | Precision::Fp16 => 2.0,
        Precision::Int8 => 1.0,
        Precision::Int4 => 0.5,
    }
}

pub struct PrecisionEstimate {
    pub precision: Precision,
    /// Estimated total weight bytes at this precision (scaled from declared total).
    pub total_bytes: f64,
    pub gib: f64,
}

pub struct MemoryPlan {
    /// Parameter count implied by declared total_size at the checkpoint dtype (BF16 here).
    pub implied_params: f64,
    pub estimates: Vec<PrecisionEstimate>,
    /// Full-attention KV cache bytes for a given context length (computed on demand).
    pub module_counts: BTreeMap<TensorModule, usize>,
}

pub struct KvEstimate {
    pub context_tokens: i64,
    /// Bytes per token per full-attention layer: 2 (K+V) * kv_heads * head_dim * 2 (bf16)
    #[allow(dead_code)]
    pub bytes_per_token_per_layer: f64,
    pub total_bytes: f64,
}

pub fn build_plan(
    _config: &ModelConfig,
    index: &SafetensorsIndex,
    precisions: &[Precision],
) -> MemoryPlan {
    // The checkpoint is stored in BF16 (config dtype bfloat16), so total_size
    // corresponds to 2 bytes per parameter. This is an anchor, not a truth claim:
    // the index has no per-tensor dtype, so all precision numbers are estimates.
    let implied_params = index.total_size as f64 / 2.0;
    let estimates = precisions
        .iter()
        .map(|&p| {
            let total_bytes = implied_params * bytes_per_param(p);
            PrecisionEstimate {
                precision: p,
                total_bytes,
                gib: total_bytes / (1024.0 * 1024.0 * 1024.0),
            }
        })
        .collect();
    MemoryPlan {
        implied_params,
        estimates,
        module_counts: tensor_classifier::module_counts(index),
    }
}

impl MemoryPlan {
    pub fn kv_estimate(&self, config: &ModelConfig, context_tokens: i64) -> KvEstimate {
        // Full-attention layers only. Linear-attention layers carry a fixed-size
        // recurrent state, not a per-token KV cache, so they are excluded here.
        let bytes_per_token_per_layer =
            2.0 * config.num_key_value_heads as f64 * config.head_dim as f64 * 2.0;
        let total_bytes =
            bytes_per_token_per_layer * config.full_attention_layers as f64 * context_tokens as f64;
        KvEstimate {
            context_tokens,
            bytes_per_token_per_layer,
            total_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ModelConfig {
        ModelConfig::from_str(
            r#"{
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {
                    "hidden_size": 5120,
                    "num_hidden_layers": 4,
                    "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "intermediate_size": 17408,
                    "vocab_size": 248320,
                    "dtype": "bfloat16"
                }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn kv_estimate_full_attention_only() {
        let cfg = config();
        let idx = SafetensorsIndex {
            weight_map: Default::default(),
            total_size: 100,
        };
        let plan = build_plan(&cfg, &idx, &[Precision::Bf16]);
        let kv = plan.kv_estimate(&cfg, 4096);
        // 2 * 4 * 256 * 2 = 4096 bytes/token/layer, 1 full layer, 4096 tokens
        assert_eq!(kv.bytes_per_token_per_layer, 4096.0);
        assert_eq!(kv.total_bytes, 4096.0 * 4096.0);
    }

    #[test]
    fn precision_scaling() {
        let cfg = config();
        let idx = SafetensorsIndex {
            weight_map: Default::default(),
            total_size: 2_000_000,
        };
        let plan = build_plan(&cfg, &idx, &[Precision::Bf16, Precision::Int4]);
        assert_eq!(plan.implied_params, 1_000_000.0);
        assert_eq!(plan.estimates[0].total_bytes, 2_000_000.0);
        assert_eq!(plan.estimates[1].total_bytes, 500_000.0);
    }
}
