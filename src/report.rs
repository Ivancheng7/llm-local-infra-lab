use crate::memory_planner::MemoryPlan;
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
        if i > 0 && (chars.len() - i) % 3 == 0 {
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

pub fn print_config(config: &ModelConfig) {
    println!("Architecture:          {}", config.architecture);
    println!("Model type:            {}", config.model_type);
    println!("Hidden size:           {}", config.hidden_size);
    println!("Layers:                {}", config.num_layers);
    println!(
        "  Full attention:      {}",
        config.full_attention_layers
    );
    println!(
        "  Linear attention:    {}",
        config.linear_attention_layers
    );
    println!("Attention heads:       {} (kv: {})", config.num_attention_heads, config.num_key_value_heads);
    println!("Head dim:              {}", config.head_dim);
    println!("Intermediate size:     {}", config.intermediate_size);
    println!("Vocab size:            {}", config.vocab_size);
    println!("Max position:          {}", config.max_position_embeddings);
    println!("MTP layers:            {}", config.mtp_layers);
    println!("Tie word embeddings:   {}", config.tie_word_embeddings);
    println!("Declared dtype:        {}", config.declared_dtype);
    println!("Vision tower:          {}", if config.has_vision { "yes" } else { "no" });
}

pub fn print_index(index: &SafetensorsIndex) {
    println!("Tensors:               {}", index.tensor_count());
    println!("Shards:                {}", index.shard_count());
    println!("Declared total bytes:  {}", thousands(index.total_size));
    println!("Declared total size:   {}", gib(index.total_size as f64));

    let mut shard_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for shard in index.weight_map.values() {
        *shard_counts.entry(shard.as_str()).or_insert(0) += 1;
    }
    println!("\nPer-shard tensor counts:");
    for (shard, count) in &shard_counts {
        println!("  {:<45} {}", shard, count);
    }
}

pub fn print_plan_markdown(config: &ModelConfig, index: &SafetensorsIndex, plan: &MemoryPlan) {
    println!("# Memory Plan");
    println!();
    println!("- Model: {} ({})", config.architecture, config.model_type);
    println!(
        "- Layers: {} ({} full attention, {} linear attention)",
        config.num_layers, config.full_attention_layers, config.linear_attention_layers
    );
    println!("- Shards: {}", index.shard_count());
    println!("- Tensors: {}", index.tensor_count());
    println!("- Declared weight bytes: {}", thousands(index.total_size));
    println!(
        "- Implied parameter count (from BF16 total): ~{:.2}B",
        plan.implied_params / 1e9
    );
    println!();
    println!("## Precision estimates");
    println!();
    println!("| Precision | Bytes/param | Estimated size |");
    println!("|---|---|---|");
    for est in &plan.estimates {
        let bpp = crate::memory_planner::bytes_per_param(est.precision);
        let bpp = if bpp.fract() == 0.0 {
            format!("{:.1}", bpp)
        } else {
            format!("{}", bpp)
        };
        println!(
            "| {:?} | {} | {} |",
            est.precision,
            bpp,
            gib(est.total_bytes)
        );
    }
    println!();
    println!("## Tensor classification (by name)");
    println!();
    println!("| Module | Tensor count |");
    println!("|---|---|");
    for (module, count) in &plan.module_counts {
        println!("| {} | {} |", module.label(), count);
    }
    println!();
    println!("## KV cache estimate (full-attention layers, BF16)");
    println!();
    let bytes_per_token =
        2.0 * config.num_key_value_heads as f64 * config.head_dim as f64;
    for ctx in [4096i64, 32768, 131072, config.max_position_embeddings] {
        let kv = plan.kv_estimate(config, ctx);
        println!(
            "- ctx {:>7}: {} ({} bytes/token/layer × {} layers)",
            thousands(kv.context_tokens),
            gib(kv.total_bytes),
            thousands(bytes_per_token as i64),
            config.full_attention_layers
        );
    }
    println!();
    println!("## Evidence boundary");
    println!();
    println!("- Precision sizes are estimates scaled from the checkpoint's declared");
    println!("  BF16 total_size; the safetensors index carries no per-tensor dtype.");
    println!("- KV estimates cover full-attention layers only; linear-attention");
    println!("  layers hold fixed-size recurrent state, not per-token KV.");
    println!("- Vision, MTP, and embedding bytes are counted inside the total but");
    println!("  not yet broken out per-module (v1 counts tensors by name only).");
}

pub fn print_plan_json(
    config: &ModelConfig,
    index: &SafetensorsIndex,
    plan: &MemoryPlan,
) -> Result<()> {
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
        "module_tensor_counts": module_counts,
        "kv_bytes_full_attention": {
            "bytes_per_token_per_layer": 2.0 * config.num_key_value_heads as f64 * config.head_dim as f64 * 2.0,
            "full_attention_layers": config.full_attention_layers,
        },
    });
    println!("{}", serde_json::to_string_pretty(&doc)?);
    Ok(())
}
