use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use system_none_gateway::{api, AppState, GatewayConfig};

/// Realtime classification gateway for SystemNone.
#[derive(Parser, Debug)]
#[command(version, about)]
#[expect(
    clippy::doc_markdown,
    reason = "these doc comments are the --help text, not Markdown"
)]
struct Args {
    /// Address to listen on.
    #[arg(long, env = "SN_LISTEN", default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
    /// SQLite database for configuration, prompts, usage and jobs.
    #[arg(long, env = "SN_DB", default_value = "gateway-data/gateway.db")]
    db: PathBuf,
    /// Where inputs and students are stored: file:///abs/path or s3://bucket/prefix (AWS_* env vars).
    #[arg(long, env = "SN_STORE")]
    store: Option<String>,
    /// Local folder for downloaded students.
    #[arg(long, env = "SN_CACHE", default_value = "gateway-data/models")]
    cache: PathBuf,
    /// ONNX Runtime shared library (default: ORT_DYLIB_PATH, else the system library).
    #[arg(long, env = "SN_ONNXRUNTIME")]
    onnxruntime: Option<PathBuf>,
    /// Threads per student inference (default: ONNX Runtime's choice).
    #[arg(long, env = "SN_THREADS")]
    threads: Option<usize>,
    /// Seconds between input-log flushes.
    #[arg(long, env = "SN_FLUSH_SECS", default_value_t = 5)]
    flush_secs: u64,
    /// Flush early once this many inputs are buffered.
    #[arg(long, env = "SN_LOG_BATCH", default_value_t = 1000)]
    log_batch: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args = Args::parse();
    let env = |var: &str| std::env::var(var).ok().filter(|t| !t.is_empty());
    let admin_token = env("SN_ADMIN_TOKEN")
        .context("SN_ADMIN_TOKEN is not set: choose the admin token and set it before starting")?;
    // Never the admin token: a worker machine must not be able to change the gateway's settings.
    // Without one, the gateway takes a random token that it never shows, so no worker gets in.
    let worker_token = env("SN_WORKER_TOKEN").unwrap_or_else(|| {
        tracing::warn!(
            "SN_WORKER_TOKEN is not set, so no worker can claim training jobs: set the same one for the gateway and the worker"
        );
        uuid::Uuid::new_v4().simple().to_string()
    });
    let store = if let Some(s) = args.store {
        s
    } else {
        let dir = std::env::current_dir()?.join("gateway-data/store");
        format!("file://{}", dir.display())
    };
    let cfg = GatewayConfig {
        db_path: args.db,
        store_url: store,
        cache_dir: args.cache,
        admin_token,
        worker_token,
        viewer_token: std::env::var("SN_VIEWER_TOKEN")
            .ok()
            .filter(|t| !t.is_empty()),
        onnxruntime: args.onnxruntime,
        threads: args.threads,
        flush_secs: args.flush_secs,
        log_batch: args.log_batch,
    };
    let state = AppState::start(cfg).await?;
    state.spawn_background();
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    tracing::info!(
        "listening on http://{} (store {})",
        listener.local_addr()?,
        state.store.url
    );
    let app = api::router(state.clone());
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await?;
    state.flush_all().await?;
    tracing::info!("flushed logs; bye");
    Ok(())
}

async fn shutdown() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("signal handler");
        tokio::select! { _ = ctrl_c => {}, _ = term.recv() => {} }
    }
    #[cfg(not(unix))]
    ctrl_c.await.ok();
}
