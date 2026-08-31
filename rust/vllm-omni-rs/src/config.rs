// SPDX-License-Identifier: Apache-2.0

//! Conversion from CLI arguments to `vllm_server::Config`.

use std::time::Duration;

use vllm_chat::multimodal::MmLimitPerPrompt;
use vllm_engine_core_client::TransportMode;
use vllm_server::{
    ApiServerOptions, ChatTemplateContentFormatOption, Config, CoordinatorMode, CorsConfig,
    DEFAULT_KEEP_ALIVE_TIMEOUT, GenerationConfigMode, HttpListenerMode, ParserSelection,
    RendererSelection,
};

use crate::cli::CommonArgs;

/// Maximum time to wait for active requests to drain during shutdown.
pub const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

impl CommonArgs {
    /// Build the OpenAI-server config for the Bootstrapped transport mode: the
    /// frontend binds ROUTER(input)/PULL(output) at the given addresses and
    /// waits for the expected engines to register, with no HELLO handshake.
    pub fn build_config(&self, input_address: String, output_address: String) -> Config {
        Config {
            transport_mode: TransportMode::Bootstrapped {
                input_address,
                output_address,
                engine_start_index: 0,
                engine_count: self.engine_count,
                data_parallel_size: self.engine_count,
                ready_timeout: Duration::from_secs(self.ready_timeout_secs),
            },
            coordinator_mode: CoordinatorMode::None,
            model: self.model.clone(),
            generation_config: GenerationConfigMode::Auto,
            served_model_name: self.served_model_name.clone(),
            listener_mode: HttpListenerMode::BindTcp {
                host: self.host.clone(),
                port: self.port,
            },
            tool_call_parser: ParserSelection::Auto,
            reasoning_parser: ParserSelection::Auto,
            renderer: RendererSelection::Auto,
            // Phase 1 is text-only; the Python facade owns multimodal
            // preprocessing.
            // TODO(omni): lift this once the facade ships preprocessed inputs.
            language_model_only: true,
            chat_template: self.chat_template.clone(),
            default_chat_template_kwargs: None,
            limit_mm_per_prompt: MmLimitPerPrompt::default(),
            chat_template_content_format: ChatTemplateContentFormatOption::Auto,
            max_logprobs: None,
            api_server_options: ApiServerOptions::default(),
            cors: CorsConfig::default(),
            tls: None,
            api_keys: self.api_key.iter().cloned().collect(),
            disable_log_stats: false,
            grpc_port: None,
            shutdown_timeout: DEFAULT_SHUTDOWN_TIMEOUT,
            keep_alive_timeout: DEFAULT_KEEP_ALIVE_TIMEOUT,
            profiler: None,
        }
    }
}
