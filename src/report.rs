use crate::memory_planner::{self, MemoryPlan};
use crate::model_config::ModelConfig;
use crate::safetensors_index::SafetensorsIndex;
use anyhow::Result;
use std::collections::BTreeMap;

fn gib(bytes: f64) -> String {
    format!("{:.2} GiB", bytes / (1024.0 * 1024.0 * 1024.0))
}

fn thousands(n: i64) -> String {
    let s = n.abs().to_string();
    let mut out = String::new();
    let chars: Vec<char> = s.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*c);
    }
    if n < 0 {
        format!("-{}", out)
    } else {
        out
    }
}

pub fn render_config(config: &ModelConfig) -> String {
    let mut out = String::new();
    out.push_str(&format!("Architecture:          {}\n", config.architecture));
    out.push_str(&format!("Model type:            {}\n", config.model_type));
    out.push_str(&format!("Hidden size:           {}\n", config.hidden_size));
    out.push_str(&format!("Layers:                {}\n", config.num_layers));
    out.push_str(&format!(
        "  Full attention:      {}\n",
        config.full_attention_layers
    ));
    out.push_str(&format!(
        "  Linear attention:    {}\n",
        config.linear_attention_layers
    ));
    out.push_str(&format!(
        "Attention heads:       {} (kv: {})\n",
        config.num_attention_heads, config.num_key_value_heads
    ));
    out.push_str(&format!("Head dim:              {}\n", config.head_dim));
    out.push_str(&format!(
        "Intermediate size:     {}\n",
        config.intermediate_size
    ));
    out.push_str(&format!("Vocab size:            {}\n", config.vocab_size));
    out.push_str(&format!(
        "Max position:          {}\n",
        config.max_position_embeddings
    ));
    out.push_str(&format!("MTP layers:            {}\n", config.mtp_layers));
    out.push_str(&format!(
        "Tie word embeddings:   {}\n",
        config.tie_word_embeddings
    ));
    out.push_str(&format!(
        "Declared dtype:        {}\n",
        config.declared_dtype
    ));
    out.push_str(&format!(
        "Vision tower:          {}\n",
        if config.has_vision { "yes" } else { "no" }
    ));
    out
}

pub fn render_index(index: &SafetensorsIndex) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Tensors:               {}\n",
        index.tensor_count()
    ));
    out.push_str(&format!("Shards:                {}\n", index.shard_count()));
    out.push_str(&format!(
        "Declared total bytes:  {}\n",
        thousands(index.total_size)
    ));
    out.push_str(&format!(
        "Declared total size:   {}\n",
        gib(index.total_size as f64)
    ));

    let mut shard_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for shard in index.weight_map.values() {
        *shard_counts.entry(shard.as_str()).or_insert(0) += 1;
    }
    out.push_str("\nPer-shard tensor counts:\n");
    for (shard, count) in &shard_counts {
        out.push_str(&format!("  {:<45} {}\n", shard, count));
    }
    out
}

pub fn render_plan_markdown(
    config: &ModelConfig,
    index: &SafetensorsIndex,
    plan: &MemoryPlan,
) -> String {
    let mut out = String::new();
    out.push_str("# Memory Plan\n\n");
    out.push_str(&format!(
        "- Model: {} ({})\n",
        config.architecture, config.model_type
    ));
    out.push_str(&format!(
        "- Layers: {} ({} full attention, {} linear attention)\n",
        config.num_layers, config.full_attention_layers, config.linear_attention_layers
    ));
    out.push_str(&format!("- Shards: {}\n", index.shard_count()));
    out.push_str(&format!("- Tensors: {}\n", index.tensor_count()));
    out.push_str(&format!(
        "- Declared weight bytes: {}\n",
        thousands(index.total_size)
    ));
    out.push_str(&format!(
        "- Implied parameter count (from BF16 total): ~{:.2}B\n\n",
        plan.implied_params / 1e9
    ));

    out.push_str("## Precision estimates\n\n");
    out.push_str("| Precision | Bytes/param | Estimated size |\n");
    out.push_str("|---|---|---|\n");
    for est in &plan.estimates {
        let bpp = memory_planner::bytes_per_param(est.precision);
        let bpp = if bpp.fract() == 0.0 {
            format!("{:.1}", bpp)
        } else {
            format!("{}", bpp)
        };
        out.push_str(&format!(
            "| {:?} | {} | {} |\n",
            est.precision,
            bpp,
            gib(est.total_bytes)
        ));
    }
    out.push('\n');

    out.push_str("## Module estimates (from config geometry, BF16 params)\n\n");
    out.push_str("| Module | Est. params | Est. BF16 size |\n");
    out.push_str("|---|---|---|\n");
    for m in memory_planner::module_param_estimates(config) {
        out.push_str(&format!(
            "| {} | ~{:.2}B | {} |\n",
            m.label,
            m.params / 1e9,
            gib(m.params * 2.0)
        ));
    }
    out.push_str(&format!(
        "| residual (linear-attn internals, vision, MTP) | ~{:.2}B | {} |\n\n",
        memory_planner::residual_bytes(config, index) / 1e9,
        gib(memory_planner::residual_bytes(config, index) * 2.0)
    ));

    out.push_str("## Tensor classification (by name)\n\n");
    out.push_str("| Module | Tensor count |\n|---|---|\n");
    for (module, count) in &plan.module_counts {
        out.push_str(&format!("| {} | {} |\n", module.label(), count));
    }
    out.push('\n');

    out.push_str("## KV cache estimate (full-attention layers, BF16)\n\n");
    let bytes_per_token = 2.0 * config.num_key_value_heads as f64 * config.head_dim as f64;
    for ctx in [4096i64, 32768, 131072, config.max_position_embeddings] {
        let kv = plan.kv_estimate(config, ctx);
        out.push_str(&format!(
            "- ctx {:>7}: {} ({} bytes/token/layer × {} layers)\n",
            thousands(kv.context_tokens),
            gib(kv.total_bytes),
            thousands(bytes_per_token as i64),
            config.full_attention_layers
        ));
    }
    out.push('\n');

    out.push_str("## Evidence boundary\n\n");
    out.push_str(
        "- Precision sizes are estimates scaled from the checkpoint's declared\n  BF16 total_size; the safetensors index carries no per-tensor dtype.\n",
    );
    out.push_str(
        "- Module parameter estimates come from config geometry (standard Qwen\n  layer shapes), not per-tensor shapes; the residual bucket collects\n  linear-attention internals, vision tower, and MTP bytes that geometry\n  does not pin down.\n",
    );
    out.push_str(
        "- KV estimates cover full-attention layers only; linear-attention\n  layers hold fixed-size recurrent state, not per-token KV.\n",
    );
    out
}

pub fn render_plan_json(
    config: &ModelConfig,
    index: &SafetensorsIndex,
    plan: &MemoryPlan,
) -> Result<String> {
    let module_counts: BTreeMap<&str, &usize> = plan
        .module_counts
        .iter()
        .map(|(m, c)| (m.label(), c))
        .collect();
    let doc = serde_json::json!({
        "model": {
            "architecture": config.architecture,
            "model_type": config.model_type,
            "layers": config.num_layers,
            "full_attention_layers": config.full_attention_layers,
            "linear_attention_layers": config.linear_attention_layers,
        },
        "index": {
            "shards": index.shard_count(),
            "tensors": index.tensor_count(),
            "declared_total_bytes": index.total_size,
        },
        "implied_params": plan.implied_params,
        "precision_estimates": plan.estimates.iter().map(|e| serde_json::json!({
            "precision": format!("{:?}", e.precision),
            "total_bytes": e.total_bytes,
            "gib": e.gib,
        })).collect::<Vec<_>>(),
        "module_param_estimates": memory_planner::module_param_estimates(config).iter().map(|m| serde_json::json!({
            "module": m.label,
            "params": m.params,
            "bf16_bytes": m.params * 2.0,
        })).collect::<Vec<_>>(),
        "residual_params": memory_planner::residual_bytes(config, index),
        "module_tensor_counts": module_counts,
        "kv_bytes_full_attention": {
            "bytes_per_token_per_layer": 2.0 * config.num_key_value_heads as f64 * config.head_dim as f64 * 2.0,
            "full_attention_layers": config.full_attention_layers,
        },
    });
    Ok(serde_json::to_string_pretty(&doc)? + "\n")
}
