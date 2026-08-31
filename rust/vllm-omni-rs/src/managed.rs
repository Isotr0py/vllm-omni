// SPDX-License-Identifier: Apache-2.0

//! Spawn and supervise the vllm-omni Python facade process.

use std::net::TcpListener;
use std::process::ExitStatus;
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow};
use tokio::process::{Child, Command};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::cli::ServeArgs;

/// Python module implementing the vllm-omni engine facade.
const FACADE_MODULE: &str = "vllm_omni.entrypoints.rust_frontend";

/// Reason that caused a managed `serve` session to stop.
#[derive(Debug)]
enum ShutdownReason {
    Signal,
    Server(anyhow::Error),
    FacadeExited(ExitStatus),
}

/// The supervised vllm-omni Python facade process.
struct Facade {
    child: Child,
}

impl Facade {
    /// Spawn the facade connected to the given frontend transport addresses,
    /// in its own process group, with stdio inherited.
    fn spawn(args: &ServeArgs, input_address: &str, output_address: &str) -> Result<Self> {
        let mut command = Command::new(&args.python);
        command
            .arg("-m")
            .arg(FACADE_MODULE)
            .arg("--input-address")
            .arg(input_address)
            .arg("--output-address")
            .arg(output_address);
        if !args.omni_args.is_empty() {
            command.arg("--").args(&args.omni_args);
        }
        // Run the facade in its own process group so shutdown signals reach
        // any grandchildren it spawns (engine workers etc.).
        command.process_group(0);
        let child = command
            .spawn()
            .with_context(|| format!("failed to spawn Python facade via {:?}", args.python))?;
        info!(pid = child.id(), python = %args.python, "spawned vllm-omni Python facade");
        Ok(Self { child })
    }

    /// Terminate the facade process group: SIGTERM first, SIGKILL after
    /// `timeout` if it has not exited.
    async fn shutdown(&mut self, timeout: Duration) -> Result<()> {
        let Some(pid) = self.child.id() else {
            // Already reaped by a `wait` in the select loop.
            return Ok(());
        };
        let pid = i32::try_from(pid).context("facade pid does not fit i32")?;
        // SAFETY: a negative pid targets the process group, which was created
        // for this child via `process_group(0)` at spawn. ESRCH (already gone)
        // is fine.
        if unsafe { libc::kill(-pid, libc::SIGTERM) } != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::ESRCH) {
                warn!(%err, "failed to SIGTERM facade process group");
            }
        }
        match tokio::time::timeout(timeout, self.child.wait()).await {
            Ok(status) => {
                info!(?status, "facade shut down gracefully");
                Ok(())
            }
            Err(_) => {
                warn!(?timeout, "facade did not exit after SIGTERM, sending SIGKILL");
                // SAFETY: same process-group targeting as the SIGTERM above.
                if unsafe { libc::kill(-pid, libc::SIGKILL) } != 0 {
                    let err = std::io::Error::last_os_error();
                    if err.raw_os_error() != Some(libc::ESRCH) {
                        warn!(%err, "failed to SIGKILL facade process group");
                    }
                }
                let status = self.child.wait().await.context("failed to reap facade process")?;
                Err(anyhow!("facade required SIGKILL, exited with {status}"))
            }
        }
    }
}

/// Allocate two loopback TCP addresses for the frontend ROUTER/PULL sockets.
///
/// The listeners are closed before the frontend binds the same ports, so there
/// is an inherent (small) race with other processes grabbing them.
// TODO: bind the ZMQ sockets first and pass inherited fds to avoid the race.
fn allocate_transport_addresses() -> Result<(String, String)> {
    let allocate_port = || {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .context("failed to allocate a loopback transport port")?;
        Ok::<_, anyhow::Error>(listener.local_addr()?.port())
    };
    let input_port = allocate_port()?;
    let output_port = allocate_port()?;
    Ok((
        format!("tcp://127.0.0.1:{input_port}"),
        format!("tcp://127.0.0.1:{output_port}"),
    ))
}

/// Spawn the facade and run the frontend against it until a shutdown signal,
/// a server exit, or a facade exit stops the session.
pub async fn run(args: ServeArgs, shutdown: CancellationToken) -> Result<()> {
    let (input_address, output_address) = allocate_transport_addresses()?;
    info!(%input_address, %output_address, "allocated frontend transport addresses");

    let mut facade = Facade::spawn(&args, &input_address, &output_address)?;
    let shutdown_timeout = Duration::from_secs(args.shutdown_timeout_secs);

    let mut config = args.common.build_config(input_address, output_address);
    config.shutdown_timeout = shutdown_timeout;
    let serve_shutdown = shutdown.clone();
    let mut serve_task = tokio::spawn(async move { vllm_server::serve(config, serve_shutdown).await });

    let shutdown_reason = tokio::select! {
        biased;

        // Received shutdown signal via Ctrl-C or SIGTERM.
        _ = shutdown.cancelled() => ShutdownReason::Signal,

        // Facade process exited unexpectedly.
        status = facade.child.wait() => {
            let status = status.context("failed to wait on facade process")?;
            warn!(%status, "vllm-omni Python facade exited, shutting down...");
            ShutdownReason::FacadeExited(status)
        }

        // Serve task exited unexpectedly.
        result = &mut serve_task => {
            let result = result.context("serve task join failed")?;
            match result {
                Ok(()) => ShutdownReason::Server(anyhow!(
                    "OpenAI server shut down unexpectedly without error"
                )),
                Err(error) => ShutdownReason::Server(error),
            }
        }
    };
    // Regardless of the shutdown reason, broadcast shutdown here to ensure all
    // serving tasks are notified.
    shutdown.cancel();

    // Terminate the facade first, then wait for the API server to drain.
    facade.shutdown(shutdown_timeout).await?;
    if !matches!(shutdown_reason, ShutdownReason::Server(_)) {
        serve_task.await.context("serve task join failed")??;
    }

    match shutdown_reason {
        ShutdownReason::Signal => Ok(()),
        ShutdownReason::Server(error) => {
            Err(error.context("OpenAI server shut down unexpectedly"))
        }
        ShutdownReason::FacadeExited(status) => Err(anyhow!(
            "vllm-omni Python facade exited unexpectedly with status {status}"
        )),
    }
}
