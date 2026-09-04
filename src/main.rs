use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

mod memory_planner;
mod model_config;
mod report;
mod safetensors_index;
mod tensor_classifier;

#[derive(ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Precision {
    Bf16,
    Fp16,
    Int8,
    Int4,
}

#[derive(Parser)]
#[command(
    name = "llm-local-infra-lab",
    about = "Local LLM metadata analyzer and memory planner"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parse and inspect a HF config.json
    InspectConfig {
        #[arg(long)]
        path: PathBuf,
    },
    /// Parse and inspect a safetensors index
    InspectIndex {
        #[arg(long)]
        path: PathBuf,
    },
    /// Generate a precision-aware memory plan from a metadata directory
    Plan {
        /// Directory containing config.json and model.safetensors.index.json
        #[arg(long)]
        metadata: PathBuf,
        /// Comma-separated precisions, e.g. bf16,int8,int4
        #[arg(long, default_value = "bf16,int8,int4")]
        precision: String,
        /// Output format
        #[arg(long, value_enum, default_value = "markdown")]
        format: OutputFormat,
        /// Write the report to this file instead of stdout
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(ValueEnum, Clone, Copy)]
enum OutputFormat {
    Markdown,
    Json,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::InspectConfig { path } => {
            let config = model_config::ModelConfig::from_file(&path)
                .with_context(|| format!("reading config {}", path.display()))?;
            print!("{}", report::render_config(&config));
        }
        Command::InspectIndex { path } => {
            let index = safetensors_index::SafetensorsIndex::from_file(&path)
                .with_context(|| format!("reading index {}", path.display()))?;
            print!("{}", report::render_index(&index));
        }
        Command::Plan {
            metadata,
            precision,
            format,
            output,
        } => {
            let config = model_config::ModelConfig::from_file(&metadata.join("config.json"))
                .with_context(|| format!("reading config in {}", metadata.display()))?;
            let index = safetensors_index::SafetensorsIndex::from_file(
                &metadata.join("model.safetensors.index.json"),
            )
            .with_context(|| format!("reading index in {}", metadata.display()))?;
            let precisions = parse_precisions(&precision)?;
            let plan = memory_planner::build_plan(&index, &precisions);
            let rendered = match format {
                OutputFormat::Json => report::render_plan_json(&config, &index, &plan)?,
                OutputFormat::Markdown => report::render_plan_markdown(&config, &index, &plan),
            };
            match output {
                Some(path) => {
                    if let Some(parent) = path.parent() {
                        std::fs::create_dir_all(parent)
                            .with_context(|| format!("creating {}", parent.display()))?;
                    }
                    std::fs::write(&path, &rendered)
                        .with_context(|| format!("writing {}", path.display()))?;
                    eprintln!("report written to {}", path.display());
                }
                None => print!("{}", rendered),
            }
        }
    }
    Ok(())
}

fn parse_precisions(spec: &str) -> Result<Vec<Precision>> {
    spec.split(',')
        .map(|part| {
            Precision::from_str(part.trim(), true)
                .map_err(|e| anyhow::anyhow!("invalid precision {:?}: {}", part.trim(), e))
        })
        .collect()
}
