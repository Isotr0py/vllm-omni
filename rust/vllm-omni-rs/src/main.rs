// SPDX-License-Identifier: Apache-2.0

//! vLLM-Omni OpenAI-compatible Rust frontend binary.

mod cli;
mod config;
mod managed;

use std::process::ExitCode;

use anyhow::{Context as _, Result};
use clap::Parser as _;
use thiserror_ext::AsReport as _;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

use crate::cli::{Cli, Command};

/// Cancellation token tripped by Ctrl-C or SIGTERM.
fn shutdown_signal() -> CancellationToken {
    let token = CancellationToken::new();
    let shutdown = token.clone();

    tokio::spawn(async move {
        let ctrl_c = async {
            tokio::signal::ctrl_c().await.expect("failed to install Ctrl-C signal handler");
        };

        let sigterm = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install SIGTERM signal handler")
                .recv()
                .await;
        };

        tokio::select! {
            _ = ctrl_c => info!("received shutdown signal (Ctrl-C), shutting down..."),
            _ = sigterm => info!("received shutdown signal (SIGTERM), shutting down..."),
        }

        shutdown.cancel();
    });

    token
}

fn main() -> ExitCode {
    vllm_tracing::init_tracing("RustFrontend");

    let cli = Cli::parse();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build Tokio runtime");

    let result = runtime.and_then(|runtime| runtime.block_on(async_main(cli)));

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            error!("process failed with error: {:#?}", error.as_report());
            ExitCode::FAILURE
        }
    }
}

async fn async_main(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Frontend(args) => {
            let config = args.common.build_config(args.input_address, args.output_address);
            vllm_server::serve(config, shutdown_signal()).await
        }
        Command::Serve(args) => managed::run(args, shutdown_signal()).await,
    }
}
