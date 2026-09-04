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

/// Parameter estimates derived from config geometry, in BF16-equivalent units.
///
/// These are analytic estimates from hidden_size / intermediate_size / vocab_size
/// and standard Qwen layer shapes. They do NOT come from per-tensor shapes
/// (the safetensors index carries none), so they are labelled as estimates.
#[derive(Debug, Clone)]
pub struct ModuleParamEstimate {
    pub label: &'static str,
    /// Estimated parameter count at BF16 storage.
    pub params: f64,
}

/// Estimate language-model parameter groups from config geometry.
pub fn module_param_estimates(config: &ModelConfig) -> Vec<ModuleParamEstimate> {
    let h = config.hidden_size as f64;
    let i = config.intermediate_size as f64;
    let v = config.vocab_size as f64;
    let f = config.full_attention_layers as f64;
    let l = config.linear_attention_layers as f64;
    let nh = config.num_attention_heads as f64;
    let nkv = config.num_key_value_heads as f64;
    let hd = config.head_dim as f64;

    let mut out = Vec::new();

    // Embedding + untied LM head: both project vocab <-> hidden.
    let embed_params = v * h;
    out.push(ModuleParamEstimate {
        label: "embedding",
        params: embed_params,
    });
    if !config.tie_word_embeddings {
        out.push(ModuleParamEstimate {
            label: "lm_head",
            params: v * h,
        });
    }

    // Full attention per layer: Q proj (h×nh·hd) + K/V (h×nkv·hd, gated Qwen3.5
    // doubles K via attn_output_gate but gate shares the K/V projection shape
    // only on Q side; keep the standard QKVO accounting) + O proj.
    let q = h * nh * hd;
    let kv = 2.0 * h * nkv * hd;
    let o = nh * hd * h;
    let full_attn_per_layer = q + kv + o;
    out.push(ModuleParamEstimate {
        label: "full_attention",
        params: full_attn_per_layer * f,
    });

    // Linear attention layers: keep a conservative placeholder from observed
    // tensor counts (12 tensors/layer) — exact per-tensor shapes are not in the
    // index, so these bytes stay inside the residual rather than being guessed.
    let linear_tensors: f64 = l;
    let _ = linear_tensors;

    // MLP per layer (SwiGLU: gate + up + down).
    let mlp_per_layer = 3.0 * h * i;
    out.push(ModuleParamEstimate {
        label: "mlp",
        params: mlp_per_layer * config.num_layers as f64,
    });

    // LayerNorms: RMSNorm is parameter-light (~2h per layer + final norm).
    let norm_params = (2.0 * config.num_layers as f64 + 1.0) * h;
    out.push(ModuleParamEstimate {
        label: "layernorm",
        params: norm_params,
    });

    out
}

/// Residual bytes: whatever the geometry-based estimates do not explain
/// (linear-attention internals, vision tower, MTP, embeddings rounding).
pub fn residual_bytes(config: &ModelConfig, index: &SafetensorsIndex) -> f64 {
    let explained: f64 = module_param_estimates(config)
        .iter()
        .map(|m| m.params)
        .sum();
    (index.total_size as f64 / 2.0 - explained).max(0.0)
}

pub struct KvEstimate {
    pub context_tokens: i64,
    /// Bytes per token per full-attention layer: 2 (K+V) * kv_heads * head_dim * 2 (bf16)
    #[allow(dead_code)]
    pub bytes_per_token_per_layer: f64,
    pub total_bytes: f64,
}

pub fn build_plan(index: &SafetensorsIndex, precisions: &[Precision]) -> MemoryPlan {
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
        let plan = build_plan(&idx, &[Precision::Bf16]);
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
        let plan = build_plan(&idx, &[Precision::Bf16, Precision::Int4]);
        assert_eq!(plan.implied_params, 1_000_000.0);
        assert_eq!(plan.estimates[0].total_bytes, 2_000_000.0);
        assert_eq!(plan.estimates[1].total_bytes, 500_000.0);
    }

    #[test]
    fn module_estimates_are_positive_and_sum_below_total() {
        let cfg = config();
        let idx = SafetensorsIndex {
            weight_map: Default::default(),
            total_size: 400_000_000_000,
        };
        let estimates = module_param_estimates(&cfg);
        assert!(!estimates.is_empty());
        assert!(estimates.iter().all(|m| m.params > 0.0));
        let explained: f64 = estimates.iter().map(|m| m.params).sum();
        assert!(explained < idx.total_size as f64 / 2.0);
        assert!(residual_bytes(&cfg, &idx) > 0.0);
    }
}
