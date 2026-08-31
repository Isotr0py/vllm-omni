// SPDX-License-Identifier: Apache-2.0

//! CLI definition for the vLLM-Omni Rust frontend.

use clap::{Args, Parser, Subcommand};

/// Environment variable overriding the Python interpreter used by `serve`.
pub const PYTHON_ENV: &str = "VLLM_OMNI_RS_PYTHON";

/// Default Python interpreter used by `serve`.
pub const DEFAULT_PYTHON: &str = "python3";

#[derive(Debug, Parser)]
#[command(
    name = "vllm-omni-rs",
    version,
    about = "vLLM-Omni OpenAI-compatible Rust frontend (Bootstrapped engine transport)."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the frontend against engines that connect to pre-bootstrapped
    /// transport addresses.
    Frontend(FrontendArgs),
    /// Spawn the vllm-omni Python facade process and run the frontend against
    /// it on self-allocated loopback transport addresses.
    Serve(ServeArgs),
}

/// Frontend arguments shared by both subcommands.
#[derive(Debug, Clone, Args)]
pub struct CommonArgs {
    /// Backend model identifier (HF id or local path). The frontend loads the
    /// tokenizer and chat template itself, so the model must be locally
    /// readable.
    #[arg(long)]
    pub model: String,
    /// HTTP bind host for the OpenAI-compatible server.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// HTTP bind port for the OpenAI-compatible server.
    #[arg(long, default_value_t = 8091)]
    pub port: u16,
    /// Model name(s) exposed to clients via the OpenAI API. When unset, falls
    /// back to `--model`.
    #[arg(long)]
    pub served_model_name: Vec<String>,
    /// Server-default chat template override, as a file path or inline
    /// template.
    #[arg(long)]
    pub chat_template: Option<String>,
    /// API key accepted as a bearer token for guarded routes.
    #[arg(long)]
    pub api_key: Option<String>,
    /// Total number of engines expected to register on this transport.
    #[arg(long, default_value_t = 1)]
    pub engine_count: usize,
    /// Maximum time to wait for the expected engines to register.
    #[arg(long, default_value_t = 600)]
    pub ready_timeout_secs: u64,
}

/// Arguments for the `frontend` subcommand.
#[derive(Debug, Args)]
pub struct FrontendArgs {
    #[command(flatten)]
    pub common: CommonArgs,
    /// Frontend input ROUTER socket address that the engine facade connects to
    /// for requests.
    #[arg(long)]
    pub input_address: String,
    /// Frontend output PULL socket address that the engine facade pushes
    /// responses to.
    #[arg(long)]
    pub output_address: String,
}

/// Arguments for the `serve` subcommand.
#[derive(Debug, Args)]
pub struct ServeArgs {
    #[command(flatten)]
    pub common: CommonArgs,
    /// Python interpreter used to run the vllm-omni facade.
    #[arg(long, default_value = DEFAULT_PYTHON, env = PYTHON_ENV)]
    pub python: String,
    /// Maximum time to wait for the facade to stop after SIGTERM before
    /// SIGKILLing its process group.
    #[arg(long, default_value_t = 30)]
    pub shutdown_timeout_secs: u64,
    /// Arguments forwarded to `vllm_omni.entrypoints.rust_frontend` after `--`.
    #[arg(last = true, allow_hyphen_values = true)]
    pub omni_args: Vec<String>,
}
