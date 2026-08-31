// SPDX-License-Identifier: Apache-2.0

//! End-to-end smoke test: run the `frontend` subcommand binary against a mock
//! engine connected over Bootstrapped transport addresses (no HELLO
//! handshake), and verify that a streaming chat completion flows through.
//!
//! Skipped when the test model is not present in the local HF cache. Set
//! `VLLM_OMNI_RS_TEST_MODEL` (and `HF_HOME`) to point at a readable model.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use futures::StreamExt as _;
use tokio::process::{Child, Command};
use tokio::time::{Instant, sleep};
use vllm_engine_core_client::mock_engine::{
    MockEngineConfig, connect_to_bootstrapped_frontend, default_ready_response,
};
use vllm_engine_core_client::protocol::output::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
};
use vllm_engine_core_client::protocol::request::{EngineCoreRequest, EngineCoreRequestType};
use vllm_engine_core_client::protocol::{decode_msgpack, encode_msgpack};
use vllm_engine_core_client::EngineId;
use zeromq::ZmqMessage;
use zeromq::prelude::{SocketRecv as _, SocketSend as _};

const TEST_MODEL_ENV: &str = "VLLM_OMNI_RS_TEST_MODEL";
const DEFAULT_TEST_MODEL: &str = "Qwen/Qwen3-0.6B";

/// Maximum time to wait for the frontend to load the model and bind its
/// sockets, and for the HTTP listener to come up.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);

/// Return the test model id, or `None` when its HF cache snapshot is missing.
fn test_model() -> Option<String> {
    let model = std::env::var(TEST_MODEL_ENV).unwrap_or_else(|_| DEFAULT_TEST_MODEL.to_string());
    if model.contains('/') {
        let hf_home = std::env::var_os("HF_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/huggingface")))?;
        let snapshot_dir = hf_home
            .join("hub")
            .join(format!("models--{}", model.replace('/', "--")));
        if !snapshot_dir.is_dir() {
            return None;
        }
    } else if !PathBuf::from(&model).is_dir() {
        // Local path model that does not exist.
        return None;
    }
    Some(model)
}

/// Allocate a free loopback TCP port.
fn allocate_port() -> u16 {
    std::net::TcpListener::bind(("127.0.0.1", 0))
        .expect("bind loopback port")
        .local_addr()
        .expect("local addr")
        .port()
}

/// Spawn the frontend binary in Bootstrapped mode.
fn spawn_frontend(
    model: &str,
    port: u16,
    input_address: &str,
    output_address: &str,
) -> Result<Child> {
    Command::new(env!("CARGO_BIN_EXE_vllm-omni-rs"))
        .arg("frontend")
        .arg("--model")
        .arg(model)
        .arg("--port")
        .arg(port.to_string())
        .arg("--input-address")
        .arg(input_address)
        .arg("--output-address")
        .arg(output_address)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("failed to spawn vllm-omni-rs frontend")
}

/// Connect one mock engine in Bootstrapped mode, answer exactly one request
/// with two streamed chunks and a STOP finish, then keep the sockets alive.
async fn run_mock_engine(input_address: String, output_address: String) -> Result<()> {
    let config = MockEngineConfig {
        local: true,
        headless: true,
        ready_response: default_ready_response(),
        connect_timeout: STARTUP_TIMEOUT,
    };
    let (mut dealer, mut push) = connect_to_bootstrapped_frontend(
        &input_address,
        &output_address,
        EngineId::from_engine_index(0),
        config,
    )
    .await
    .context("mock engine failed to connect to frontend")?;

    let frames = dealer.recv().await.context("recv engine message")?.into_vec();
    assert_eq!(frames.len(), 2, "expected [type, payload] frames");
    assert_eq!(
        EngineCoreRequestType::from_frame(&frames[0]),
        Some(EngineCoreRequestType::Add),
        "expected an ADD request"
    );
    let request: EngineCoreRequest = decode_msgpack(&frames[1]).context("decode ADD request")?;

    for (new_token_ids, finish_reason) in [
        (vec![9707u32], None),
        (vec![198u32], Some(EngineCoreFinishReason::Stop)),
    ] {
        let outputs: EngineCoreOutputs = RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: request.request_id.clone(),
                new_token_ids,
                finish_reason,
                ..Default::default()
            }],
            ..Default::default()
        }
        .into();
        push.send(ZmqMessage::from(encode_msgpack(&outputs)?))
            .await
            .context("send engine outputs")?;
    }

    // Keep the sockets alive until the test kills the frontend.
    futures::future::pending::<()>().await;
    Ok(())
}

/// Wait until the frontend's HTTP listener accepts connections.
async fn wait_for_http(port: u16) -> Result<()> {
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    loop {
        match tokio::net::TcpStream::connect(("127.0.0.1", port)).await {
            Ok(_) => return Ok(()),
            Err(_) if Instant::now() < deadline => sleep(Duration::from_millis(100)).await,
            Err(error) => bail!("HTTP listener never came up: {error}"),
        }
    }
}

#[tokio::test]
async fn frontend_streams_from_bootstrapped_mock_engine() -> Result<()> {
    let Some(model) = test_model() else {
        eprintln!("skipping: test model not found in local HF cache");
        return Ok(());
    };

    let port = allocate_port();
    let input_address = format!("tcp://127.0.0.1:{}", allocate_port());
    let output_address = format!("tcp://127.0.0.1:{}", allocate_port());

    let mut frontend = spawn_frontend(&model, port, &input_address, &output_address)?;
    let engine_task = tokio::spawn(run_mock_engine(input_address, output_address));

    let result = run_streaming_request(&model, port).await;

    // SIGTERM the frontend; it should shut down gracefully.
    let pid = frontend.id().context("frontend pid")?;
    // SAFETY: signaling our own child process.
    assert_eq!(unsafe { libc::kill(pid as i32, libc::SIGTERM) }, 0);
    let status = tokio::time::timeout(Duration::from_secs(30), frontend.wait())
        .await
        .context("frontend did not exit after SIGTERM")??;
    assert!(status.success(), "frontend exited with {status}");

    engine_task.abort();
    result
}

async fn run_streaming_request(model: &str, port: u16) -> Result<()> {
    wait_for_http(port).await?;

    let response = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "max_tokens": 8,
        }))
        .send()
        .await
        .context("send chat completion request")?;
    assert!(
        response.status().is_success(),
        "chat completion failed with status {}",
        response.status()
    );

    let mut body = String::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        body.push_str(std::str::from_utf8(&chunk.context("read SSE chunk")?)?);
    }

    eprintln!("SSE response body:\n{body}");
    assert!(body.contains("data: "), "expected SSE data events, got: {body}");
    assert!(body.contains("\"delta\""), "expected streaming deltas, got: {body}");
    assert!(body.contains("data: [DONE]"), "expected [DONE] sentinel, got: {body}");
    Ok(())
}
