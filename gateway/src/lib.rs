//! system-none-gateway: the realtime side of SystemNone.
//!
//! * `POST /v1/classify` answers classification requests. A prompt's distilled student answers
//!   when it is confident enough; otherwise the configured System One model (or LLM) does.
//! * Every input is logged to a local folder or S3 (inputs only, never the answers).
//! * When a prompt has received enough requests, a training job is queued for the Python worker,
//!   which judges the logged inputs, fine-tunes the chosen base model and publishes an ONNX
//!   student. The gateway then loads it and serves it in-process with ONNX Runtime. A job the
//!   worker reports failed is retried, each retry waiting twice as long as the one before.
//! * `/` shows prompt usage; `/admin` configures models, judges, thresholds and per-prompt choices.

pub mod api;
pub mod cache;
pub mod db;
mod decisions;
pub mod language;
pub mod monitor;
pub mod prompt;
pub mod references;
pub mod store;
pub mod student;
pub mod upstream;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::Duration;

use anyhow::Result;
use serde_json::{json, Value};

use db::{DayUsage, Db, Deployment, PromptOverrides, Settings};
use prompt::PromptSpec;
use store::{InputLog, Store};
use student::Student;

#[derive(Clone, Default)]
pub struct GatewayConfig {
    pub db_path: PathBuf,
    pub store_url: String,
    pub cache_dir: PathBuf,
    pub admin_token: String,
    pub worker_token: String,
    pub viewer_token: Option<String>,
    pub onnxruntime: Option<PathBuf>,
    pub threads: Option<usize>,
    pub flush_secs: u64,
    pub log_batch: usize,
}

/// Says whether each token is set, never what it is, so no `{:?}` puts one in a log.
impl std::fmt::Debug for GatewayConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let token = |t: Option<&str>| match t {
            Some(t) if !t.is_empty() => "<set>",
            _ => "<not set>",
        };
        f.debug_struct("GatewayConfig")
            .field("db_path", &self.db_path)
            .field("store_url", &self.store_url)
            .field("cache_dir", &self.cache_dir)
            .field("admin_token", &token(Some(&self.admin_token)))
            .field("worker_token", &token(Some(&self.worker_token)))
            .field("viewer_token", &token(self.viewer_token.as_deref()))
            .field("onnxruntime", &self.onnxruntime)
            .field("threads", &self.threads)
            .field("flush_secs", &self.flush_secs)
            .field("log_batch", &self.log_batch)
            .finish()
    }
}

pub struct Deployed {
    pub student: Student,
    /// Known answers published with the student, to audit it against.
    pub references: references::References,
    pub threshold: f64,
    pub job_id: String,
    pub monitor: Mutex<monitor::MonitorState>,
    /// Students of their own for the languages the router is sure of (none unless the prompt
    /// trains a student per language).
    pub languages: language::Students,
}

impl Deployed {
    /// The student that answers `text`, its threshold, and its language when the router sent it
    /// to that language's student.
    pub fn route(&self, text: &str) -> (&Student, f64, Option<&str>) {
        match self.languages.route(text) {
            Some((code, student, threshold)) => (student, threshold, Some(code)),
            None => (&self.student, self.threshold, None),
        }
    }
}

pub struct PromptState {
    pub id: String,
    pub spec: PromptSpec,
    pub total: AtomicU64,
    pub overrides: RwLock<PromptOverrides>,
    pub status: RwLock<String>,
    pub deployed: RwLock<Option<Arc<Deployed>>>,
}

impl PromptState {
    pub fn status(&self) -> String {
        self.status
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
    pub fn set_status(&self, s: &str) {
        *self.status.write().unwrap_or_else(PoisonError::into_inner) = s.to_string();
    }
    pub fn deployed(&self) -> Option<Arc<Deployed>> {
        self.deployed
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
    pub fn overrides(&self) -> PromptOverrides {
        self.overrides
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

pub struct AppState {
    pub cfg: GatewayConfig,
    pub db: Db,
    pub store: Arc<Store>,
    pub inputs: InputLog,
    pub http: reqwest::Client,
    pub prompts: RwLock<HashMap<String, Arc<PromptState>>>,
    pub usage: Mutex<HashMap<(String, String), DayUsage>>,
    /// Request latencies counted by (prompt, day, route as JSON, microseconds), flushed with the
    /// usage counters. A SQLite write per request would cap the gateway at a few hundred
    /// requests a second, every request queueing behind the last one's write.
    pub latencies: Mutex<HashMap<(String, String, String, i64), u64>>,
    /// Serializes "threshold reached -> queue a job" so a prompt gets one job, and every change
    /// to a job's status, a prompt's status or its student, so memory says what SQLite says.
    pub queue_lock: Mutex<()>,
    /// Admin configuration read on every request, cached; refreshed on every admin write.
    pub config_cache: RwLock<(Settings, Vec<db::SystemOneModel>, Vec<db::JudgeModel>)>,
    pub audit_slots: Arc<tokio::sync::Semaphore>,
    /// System One's answers, when the settings turn the cache on (see `cache`).
    pub answers: cache::AnswerCache,
}

pub type Shared = Arc<AppState>;

impl AppState {
    pub async fn start(cfg: GatewayConfig) -> Result<Shared> {
        let db = Db::open(&cfg.db_path)?;
        let store = Arc::new(Store::open(&cfg.store_url)?);
        let inputs = InputLog::new(store.clone(), cfg.log_batch.max(1));
        let http = reqwest::Client::builder()
            .pool_max_idle_per_host(64)
            .build()?;
        let state = Arc::new(AppState {
            cfg,
            db,
            store,
            inputs,
            http,
            prompts: RwLock::new(HashMap::new()),
            usage: Mutex::new(HashMap::new()),
            latencies: Mutex::new(HashMap::new()),
            queue_lock: Mutex::new(()),
            config_cache: RwLock::new((Settings::default(), vec![], vec![])),
            audit_slots: Arc::new(tokio::sync::Semaphore::new(32)),
            answers: cache::AnswerCache::default(),
        });
        state.reload_config()?;
        let deployments: HashMap<String, Deployment> = state
            .db
            .deployments()?
            .into_iter()
            .filter(|d| d.active)
            .map(|d| (d.prompt_id.clone(), d))
            .collect();
        for row in state.db.prompts()? {
            let ps = Arc::new(PromptState {
                id: row.id.clone(),
                spec: row.spec,
                total: AtomicU64::new(row.total),
                overrides: RwLock::new(row.overrides),
                status: RwLock::new(row.status),
                deployed: RwLock::new(None),
            });
            if let Some(d) = deployments.get(&row.id) {
                state.restore_deployment(&ps, d).await?;
            }
            state
                .prompts
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(row.id, ps);
        }
        tracing::info!(
            "loaded {} prompts, {} active students",
            state
                .prompts
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .len(),
            deployments.len()
        );
        Ok(state)
    }

    /// Serve a prompt's active student again after a restart, with its audit as it was; a
    /// student that can't be loaded leaves System One answering.
    async fn restore_deployment(&self, ps: &Arc<PromptState>, d: &Deployment) -> Result<()> {
        let (s, references, languages) = match self
            .load_student(&ps.id, &d.job_id, &d.model_prefix, ps.spec.labels.len())
            .await
        {
            Ok(loaded) => loaded,
            Err(e) => {
                tracing::error!(
                    "prompt {}: could not load its student ({e:#}); serving with System One",
                    ps.id
                );
                return Ok(());
            }
        };
        let baseline = d.metrics["cascade"]["target_accuracy"]
            .as_f64()
            .filter(|a| a.is_finite() && (0.0..=1.0).contains(a))
            .unwrap_or(
                ps.overrides()
                    .target_accuracy
                    .unwrap_or(self.settings().default_target_accuracy),
            );
        let monitor = self
            .db
            .monitor(&ps.id)?
            .filter(|m| m.job_id == d.job_id)
            .unwrap_or_else(|| {
                monitor::MonitorState::for_deployment(
                    d,
                    baseline,
                    self.judge_for(ps),
                    self.settings().monitor.enabled,
                )
            });
        if monitor.phase == "shadow" && ps.status() == "deployed" {
            self.db.set_status(&ps.id, "shadow")?;
            ps.set_status("shadow");
        }
        *ps.deployed.write().unwrap_or_else(PoisonError::into_inner) = Some(Arc::new(Deployed {
            student: s,
            references,
            threshold: d.threshold,
            job_id: d.job_id.clone(),
            monitor: Mutex::new(monitor),
            languages,
        }));
        Ok(())
    }

    /// Background work: flush logged inputs and usage counters.
    pub fn spawn_background(self: &Shared) {
        let s = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = tokio::time::sleep(Duration::from_secs(s.cfg.flush_secs.max(1))) => {}
                    () = s.inputs.wake.notified() => {}
                }
                if let Err(e) = s.inputs.flush().await {
                    tracing::error!("flushing inputs: {e:#}");
                }
            }
        });
        let s = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                if let Err(e) = s.flush_usage() {
                    tracing::error!("flushing usage: {e:#}");
                }
            }
        });
    }

    pub async fn flush_all(&self) -> Result<()> {
        self.inputs.flush().await?;
        self.flush_usage()
    }

    /// Write the usage counters and the latency samples counted in memory to SQLite.
    pub fn flush_usage(&self) -> Result<()> {
        let taken = std::mem::take(&mut *self.usage.lock().unwrap_or_else(PoisonError::into_inner));
        if !taken.is_empty() {
            let rows: Vec<(String, String, DayUsage)> =
                taken.into_iter().map(|((p, d), u)| (p, d, u)).collect();
            self.db.add_usage(&rows)?;
        }
        self.flush_latencies()
    }

    /// Write the latency samples counted in memory to SQLite, as the usage page does before it
    /// reads them.
    #[expect(
        clippy::cast_precision_loss,
        reason = "microseconds of one request, far below 2^52"
    )]
    pub fn flush_latencies(&self) -> Result<()> {
        let taken = std::mem::take(
            &mut *self
                .latencies
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        );
        if taken.is_empty() {
            return Ok(());
        }
        let rows: Vec<db::PathLatency> = taken
            .into_iter()
            .map(|((prompt_id, day, path, us), count)| db::PathLatency {
                prompt_id,
                day,
                path,
                latency_ms: us as f64 / 1000.0,
                count,
            })
            .collect();
        self.db.add_path_latencies(&rows)
    }

    /// Count a request answered by `source` (`student`, `system_one`, `cache` or an error) through `path`,
    /// how long it took, and what its System One calls cost. Only memory: `flush_usage` writes
    /// it to SQLite every second.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a request's microseconds fit an i64 for 292,000 years"
    )]
    pub fn record(
        &self,
        prompt_id: &str,
        source: &str,
        path: &[String],
        latency_ms: f64,
        system_one_usd: f64,
    ) {
        let day = db::today();
        let route = serde_json::to_string(path).unwrap_or_default();
        *self
            .latencies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry((
                prompt_id.to_string(),
                day.clone(),
                route,
                (latency_ms * 1000.0).round() as i64,
            ))
            .or_default() += 1;
        let mut u = self.usage.lock().unwrap_or_else(PoisonError::into_inner);
        let e = u
            .entry((prompt_id.to_string(), day.clone()))
            .or_insert_with(|| DayUsage {
                day,
                ..Default::default()
            });
        match source {
            "student" => e.student += 1,
            "system_one" => e.system_one += 1,
            "cache" => e.cache += 1,
            _ => e.errors += 1,
        }
        e.latency_ms_sum += latency_ms;
        e.system_one_usd += system_one_usd;
    }

    pub fn settings(&self) -> Settings {
        self.config_cache
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .0
            .clone()
    }

    pub fn system_one_models(&self) -> Vec<db::SystemOneModel> {
        self.config_cache
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .1
            .clone()
    }

    /// Re-read settings and System One models after an admin change.
    /// SQLite is read under the write lock, so a reload that read it earlier can't replace the
    /// cache after one that read it later.
    pub fn reload_config(&self) -> Result<()> {
        let mut cache = self
            .config_cache
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        *cache = (
            self.db.settings()?,
            self.db.system_one_models()?,
            self.db.judges()?,
        );
        self.answers.configure(&cache.0.cache);
        Ok(())
    }

    pub fn judge_for(&self, p: &PromptState) -> Option<db::JudgeModel> {
        let cache = self
            .config_cache
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        let name = p.overrides().judge.or(cache.0.default_judge.clone())?;
        cache
            .2
            .iter()
            .find(|j| j.name == name && j.enabled && j.check().is_ok())
            .cloned()
    }

    pub fn prompt(&self, id: &str) -> Option<Arc<PromptState>> {
        self.prompts
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(id)
            .cloned()
    }

    /// Find or register a prompt (new prompts are saved to the database and the store).
    pub async fn resolve_prompt(&self, spec: PromptSpec) -> Result<Arc<PromptState>> {
        let id = spec.id();
        if let Some(p) = self.prompt(&id) {
            return Ok(p);
        }
        self.db.insert_prompt(&spec)?;
        if let Err(e) = self.store.put_prompt(&spec).await {
            tracing::warn!("could not write prompt {id} to the store: {e:#}");
        }
        let mut map = self.prompts.write().unwrap_or_else(PoisonError::into_inner);
        Ok(map
            .entry(id.clone())
            .or_insert_with(|| {
                Arc::new(PromptState {
                    id,
                    spec,
                    total: AtomicU64::new(0),
                    overrides: RwLock::new(PromptOverrides::default()),
                    status: RwLock::new("collecting".into()),
                    deployed: RwLock::new(None),
                })
            })
            .clone())
    }

    pub fn threshold_for(&self, p: &PromptState, s: &Settings) -> u64 {
        p.overrides().threshold.unwrap_or(s.default_threshold)
    }

    /// Count a request; queue training once the prompt reaches its threshold.
    pub fn count_request(&self, p: &PromptState) {
        let total = p.total.fetch_add(1, Ordering::Relaxed) + 1;
        if p.status() != "collecting" {
            return;
        }
        let s = self.settings();
        if total >= self.threshold_for(p, &s) {
            if let Err(e) = self.queue_job(p, "threshold") {
                tracing::error!("queueing training for {}: {e:#}", p.id);
            }
        }
    }

    /// Queue a training job for the prompt unless it has one queued or running. "Train now"
    /// (`manual`) doesn't wait for a scheduled retry either: the retry becomes claimable at once,
    /// and is the job returned.
    pub fn queue_job(&self, p: &PromptState, reason: &str) -> Result<Option<db::Job>> {
        let _g = self
            .queue_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.queue_job_locked(p, reason)
    }

    pub(crate) fn queue_job_locked(
        &self,
        p: &PromptState,
        reason: &str,
    ) -> Result<Option<db::Job>> {
        // count_request and requeue_ready saw a collecting prompt before they took the lock; it
        // may have been queued, trained or cancelled since.
        if reason == "threshold" && p.status() != "collecting" {
            return Ok(None);
        }
        if let Some(open) = self.db.open_job_for(&p.id)? {
            if reason == "manual" && self.db.skip_wait(&open.id)? {
                tracing::info!("prompt {}: retry {} runs now ({reason})", p.id, open.id);
                return self.db.job(&open.id);
            }
            return Ok(None);
        }
        let job = self.db.create_job(
            &p.id,
            &json!({"reason": reason, "requests": p.total.load(Ordering::Relaxed)}),
        )?;
        self.db.set_status(&p.id, "queued")?;
        p.set_status("queued");
        tracing::info!("prompt {}: queued training job {} ({reason})", p.id, job.id);
        Ok(Some(job))
    }

    /// Everything a worker needs to train this prompt's student, from the current admin settings.
    pub fn job_spec(&self, job: &db::Job) -> std::result::Result<Value, String> {
        let p = self
            .prompt(&job.prompt_id)
            .ok_or("the prompt no longer exists")?;
        let s = self.settings();
        let o = p.overrides();
        let judge_name = o
            .judge
            .clone()
            .or(s.default_judge.clone())
            .ok_or("no judge chosen: set a default judge or one for this prompt")?;
        let judges = self.db.judges().map_err(|e| e.to_string())?;
        let judge = judges
            .into_iter()
            .find(|j| j.name == judge_name && j.enabled)
            .ok_or(format!("judge '{judge_name}' is not configured"))?;
        judge.check()?;
        let base_model = o.base_model.clone().unwrap_or(s.default_base_model.clone());
        Ok(json!({
            "job_id": job.id,
            "prompt_id": p.id,
            "prompt": p.spec,
            "base_model": base_model,
            "judge": {
                "name": judge.name, "base_url": judge.base_url, "model": judge.model, "api_key_env": judge.api_key_env,
                "mode": judge.mode, "samples": judge.samples, "license_ack": judge.license_ack
            },
            "target_accuracy": o.target_accuracy.unwrap_or(s.default_target_accuracy),
            "per_language": {"enabled": o.per_language.unwrap_or(false), "min_confidence": s.per_language.min_confidence,
                             "min_examples": s.per_language.min_examples, "base_model": s.per_language.base_model},
            "train": s.train,
            "calibration": s.calibration,
            "store": self.store.url,
            "inputs_prefix": format!("inputs/{}/", p.id),
            "models_prefix": format!("models/{}/{}", p.id, job.id),
        }))
    }

    /// A published student, the known answers to audit it against, and its students per
    /// language, if any.
    pub async fn load_student(
        &self,
        prompt_id: &str,
        job_id: &str,
        prefix: &str,
        num_labels: usize,
    ) -> Result<(Student, references::References, language::Students)> {
        student::init_runtime(self.cfg.onnxruntime.as_deref())?;
        let dir = self.cfg.cache_dir.join(prompt_id).join(job_id);
        if !dir.join("model.onnx").exists() {
            self.store.download_model(prefix, &dir).await?;
        }
        let threads = self.cfg.threads;
        tokio::task::spawn_blocking(move || {
            Ok((
                Student::load(&dir, threads)?,
                references::References::load(&dir, num_labels)?,
                language::Students::load(&dir, threads)?,
            ))
        })
        .await?
    }
}
