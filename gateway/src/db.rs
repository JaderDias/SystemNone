//! SQLite persistence for configuration (admin), prompts, daily usage, training jobs and deployments.

use std::path::Path;
use std::sync::{Mutex, PoisonError};

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::prompt::PromptSpec;

/// Services whose terms forbid training on their outputs: they may serve traffic, never judge.
pub const DENYLIST: &[(&str, &str)] = &[
    ("typesafe", "TypeSafe's Master Customer Agreement §2.3(b) prohibits using Jev outputs for model distillation."),
    ("jev", "TypeSafe's Master Customer Agreement §2.3(b) prohibits using Jev outputs for model distillation."),
    ("fastino", "Fastino's Terms of Service §4.A prohibit using its inference responses to train, fine-tune or distill models that compete with Fastino."),
];

pub fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

pub fn today() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}

/// The longest a retry may wait: a week.
pub const MAX_RETRY_SECS: u64 = 7 * 24 * 3600;

/// `secs` seconds from now (at most `MAX_RETRY_SECS`), in `now()`'s format, whose strings sort
/// as their times do.
pub fn secs_from_now(secs: u64) -> String {
    let secs = i64::try_from(secs.min(MAX_RETRY_SECS)).unwrap_or_default();
    (chrono::Utc::now() + chrono::Duration::seconds(secs))
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn yes() -> bool {
    true
}

/// Environment variables that may hold the deployment's own tokens (SN_*) and store credentials
/// (AWS_*). QS_* is on the list too: nothing reads it, but a deployment from before the project
/// was renamed SystemNone may still hold the same tokens under that prefix.
const OWN_CREDENTIAL_PREFIXES: [&str; 3] = ["SN_", "QS_", "AWS_"];

/// An endpoint's API key is read from the environment variable the admin names, and sent to the
/// endpoint's URL, so a name with one of the `OWN_CREDENTIAL_PREFIXES` is refused
/// (case-insensitively, as Windows reads the environment).
pub fn check_key_env(name: Option<&str>) -> Result<(), String> {
    let n = name.unwrap_or_default();
    let upper = n.to_ascii_uppercase();
    if OWN_CREDENTIAL_PREFIXES.iter().any(|p| upper.starts_with(p)) {
        return Err(format!("api_key_env '{n}' names one of the gateway's own credentials (SN_*, QS_*, AWS_*); use a variable that holds only this endpoint's key."));
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SystemOneModel {
    #[serde(default)]
    pub name: String,
    /// `openai_chat`: any OpenAI-compatible chat endpoint, asked for the option's letter
    /// (for the label's name above 26 labels).
    /// `decision_http`: POST {model, instructions, labels, input} -> {label, confidence?}.
    /// `systemone`: `POST {base_url}/v1/systemone`, TypeSafe's wire format, which Decider and
    /// Fastino's GLiDE serve.
    /// `openai_decisions`: `POST {base_url}/decisions`; `base_url` includes `/v1`.
    pub kind: String,
    pub base_url: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    #[serde(default = "yes")]
    pub logprobs: bool,
    #[serde(default = "yes")]
    pub enabled: bool,
    /// A stronger System One model (an LLM, say) that answers instead when this one's
    /// confidence is below `escalate_below`. Answers without a confidence never escalate.
    #[serde(default)]
    pub escalate_to: Option<String>,
    #[serde(default)]
    pub escalate_below: Option<f64>,
    /// What a call costs, for the usage page's estimate. Every model starts at TypeSafe Jev's rates.
    #[serde(default)]
    pub pricing: Pricing,
}

fn default_timeout() -> u64 {
    30_000
}

/// TypeSafe Jev's list price: $42 per billion input tokens, output tokens free
/// (typesafe.ai, October 2026).
pub const JEV_USD_PER_MILLION_INPUT_TOKENS: f64 = 0.042;

/// What one System One call costs, in US dollars: a price per call, plus prices per million input
/// and output tokens. The token counts are the ones the model reports (`systemone` and
/// `openai_decisions` answers, and the `usage` of `OpenAI`-compatible ones), otherwise an
/// estimate from the text sent (`upstream::estimated_input_tokens`, with no output tokens).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Pricing {
    pub usd_per_call: f64,
    pub usd_per_million_input_tokens: f64,
    pub usd_per_million_output_tokens: f64,
}

impl Default for Pricing {
    fn default() -> Self {
        Pricing {
            usd_per_call: 0.0,
            usd_per_million_input_tokens: JEV_USD_PER_MILLION_INPUT_TOKENS,
            usd_per_million_output_tokens: 0.0,
        }
    }
}

impl Pricing {
    pub fn check(&self) -> Result<(), String> {
        let prices = [
            self.usd_per_call,
            self.usd_per_million_input_tokens,
            self.usd_per_million_output_tokens,
        ];
        if prices.iter().all(|p| p.is_finite() && *p >= 0.0) {
            Ok(())
        } else {
            Err("pricing: prices must be zero or more".into())
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "token counts of one call, far below 2^52"
    )]
    pub fn cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        self.usd_per_call
            + input_tokens as f64 * self.usd_per_million_input_tokens / 1e6
            + output_tokens as f64 * self.usd_per_million_output_tokens / 1e6
    }
}

impl SystemOneModel {
    pub fn check(&self) -> Result<(), String> {
        if ![
            "openai_chat",
            "decision_http",
            "systemone",
            "openai_decisions",
        ]
        .contains(&self.kind.as_str())
        {
            return Err(
                "kind must be 'openai_chat', 'decision_http', 'systemone' or 'openai_decisions'"
                    .into(),
            );
        }
        self.pricing.check()?;
        check_key_env(self.api_key_env.as_deref())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct JudgeModel {
    #[serde(default)]
    pub name: String,
    pub base_url: String,
    pub model: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// `logprobs` or `sample` for an OpenAI-compatible endpoint; `systemone` for a
    /// `POST {base_url}/v1/systemone` server such as Decider, whose probabilities are the soft labels.
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_samples")]
    pub samples: u32,
    /// The admin confirms the model's license and provider terms allow training on its outputs.
    #[serde(default)]
    pub license_ack: bool,
    #[serde(default = "yes")]
    pub enabled: bool,
}

fn default_mode() -> String {
    "logprobs".into()
}
fn default_samples() -> u32 {
    7
}

impl JudgeModel {
    pub fn check(&self) -> Result<(), String> {
        let ident = format!("{} {} {}", self.name, self.model, self.base_url).to_lowercase();
        for (key, reason) in DENYLIST {
            if ident.contains(key) {
                return Err(format!("'{}' can't be a judge: {reason} It can still serve traffic as a System One model.", self.model));
            }
        }
        if !self.license_ack {
            return Err("Confirm that this model's license and your provider's terms allow training on its outputs (license_ack).".into());
        }
        if !["logprobs", "sample", "systemone", "openai_decisions"].contains(&self.mode.as_str()) {
            return Err(
                "mode must be 'logprobs', 'sample', 'systemone' or 'openai_decisions'".into(),
            );
        }
        check_key_env(self.api_key_env.as_deref())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CalibrationSettings {
    pub method: String,
    pub confidence_level: f64,
}

impl Default for CalibrationSettings {
    fn default() -> Self {
        Self {
            method: "empirical".into(),
            confidence_level: 0.95,
        }
    }
}

impl CalibrationSettings {
    pub fn check(&self) -> Result<(), String> {
        if !matches!(self.method.as_str(), "empirical" | "clopper_pearson") {
            return Err("calibration method must be empirical or clopper_pearson".into());
        }
        if !(0.0 < self.confidence_level && self.confidence_level < 1.0) {
            return Err("calibration confidence_level must be between 0 and 1 (exclusive)".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Settings {
    #[serde(default)]
    pub monitor: crate::monitor::MonitorSettings,
    /// Requests a prompt must receive before its student is trained.
    pub default_threshold: u64,
    pub default_system_one: Option<String>,
    pub default_judge: Option<String>,
    pub default_base_model: String,
    pub default_target_accuracy: f64,
    #[serde(default)]
    pub calibration: CalibrationSettings,
    /// Extra `train:` settings passed to the worker (epochs, lr, `max_length`, ...).
    pub train: Value,
    /// Write every request's input to the store (needed for training).
    pub log_inputs: bool,
    /// Attempts a training gets: a job the worker reports failed is retried until this many
    /// have failed in a row (1: never retried).
    #[serde(default = "default_max_attempts")]
    pub max_attempts: u32,
    /// Seconds the first retry waits. Each retry after it waits twice as long as the one
    /// before, up to `retry_max_secs`.
    #[serde(default = "default_retry_secs")]
    pub retry_secs: u64,
    #[serde(default = "default_retry_max_secs")]
    pub retry_max_secs: u64,
    /// A cache of System One's answers (off unless turned on).
    #[serde(default)]
    pub cache: crate::cache::CacheSettings,
    /// How prompts with a student per language train and route them.
    #[serde(default)]
    pub per_language: PerLanguageSettings,
}

// Settings saved before retries existed read with these.
fn default_max_attempts() -> u32 {
    5
}
fn default_retry_secs() -> u64 {
    60
}
fn default_retry_max_secs() -> u64 {
    3600
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            monitor: crate::monitor::MonitorSettings::default(),
            default_threshold: 10_000,
            default_system_one: None,
            default_judge: None,
            default_base_model: "jhu-clsp/ettin-encoder-17m".into(),
            default_target_accuracy: 0.95,
            calibration: CalibrationSettings::default(),
            train: serde_json::json!({}),
            log_inputs: true,
            max_attempts: default_max_attempts(),
            retry_secs: default_retry_secs(),
            retry_max_secs: default_retry_max_secs(),
            cache: crate::cache::CacheSettings::default(),
            per_language: PerLanguageSettings::default(),
        }
    }
}

impl Settings {
    /// How many seconds the retry of a job whose attempt `attempt` failed waits: `retry_secs`,
    /// doubled for each failure in a row before this one, at most `retry_max_secs`. None when
    /// that was the last attempt.
    pub fn retry_delay(&self, attempt: u32) -> Option<u64> {
        (attempt < self.max_attempts).then(|| {
            let doublings = 2u64.saturating_pow(attempt.saturating_sub(1));
            self.retry_secs
                .saturating_mul(doublings)
                .min(self.retry_max_secs)
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PromptOverrides {
    pub system_one_model: Option<String>,
    pub judge: Option<String>,
    pub base_model: Option<String>,
    pub threshold: Option<u64>,
    pub target_accuracy: Option<f64>,
    /// Train a student per language the router is sure of, besides the prompt's student.
    pub per_language: Option<bool>,
}

/// Students per language (`PromptOverrides::per_language`): how sure the language router must be
/// for a text to count as its language, in training and in serving, how many such texts a
/// language needs for a student of its own, and the base model of those students.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PerLanguageSettings {
    pub min_confidence: f64,
    pub min_examples: u64,
    pub base_model: String,
}

impl Default for PerLanguageSettings {
    fn default() -> Self {
        PerLanguageSettings {
            // Measured on banking77 and the ai4privacy slice: from 0.7, English goes to English every
            // time, and Hindi and Telugu as often as the slice's own language tags agree (92-95%),
            // which a higher bar doesn't raise; it only routes fewer texts.
            min_confidence: 0.7,
            min_examples: 300,
            base_model: "jhu-clsp/ettin-encoder-17m".into(),
        }
    }
}

impl PerLanguageSettings {
    pub fn check(&self) -> Result<(), String> {
        if !(self.min_confidence > 0.0 && self.min_confidence <= 1.0) {
            return Err("per_language min_confidence must be in (0, 1]".into());
        }
        if self.min_examples < 20 {
            return Err("per_language min_examples must be at least 20".into());
        }
        if self.base_model.trim().is_empty() {
            return Err("per_language base_model must not be empty".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PromptRow {
    pub id: String,
    pub spec: PromptSpec,
    pub created_at: String,
    pub last_seen: String,
    pub total: u64,
    pub overrides: PromptOverrides,
    /// collecting | queued | training | deployed | failed
    pub status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub prompt_id: String,
    pub status: String,
    pub created_at: String,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub message: Option<String>,
    /// How far a running job's stage (its message) has got, when the worker counts it, such as
    /// inputs judged or training steps: `done` of `total`. A message without counts clears them.
    pub done: Option<u64>,
    pub total: Option<u64>,
    /// Last reported stage and its counts survive failure or cancellation.
    pub last_stage: Option<String>,
    pub stage_done: Option<u64>,
    pub stage_total: Option<u64>,
    pub spec: Value,
    /// Which attempt at the prompt's training this is: 1 for a job the threshold, "Train now" or
    /// an admin's Retry queued, one more than the failed job's for its retry.
    pub attempt: u32,
    /// A retry's time: no worker claims the job before it. None: claimable at once.
    pub not_before: Option<String>,
}

/// The message a job an admin cancels fails with.
pub const CANCELLED_BY_ADMIN: &str = "cancelled by an admin";

pub fn progress_stage(message: &str) -> Option<&'static str> {
    [
        "judging",
        "training",
        "calibrating",
        "exporting",
        "validating",
    ]
    .into_iter()
    .find(|stage| {
        message
            .strip_prefix(stage)
            .is_some_and(|rest| rest.starts_with(' '))
    })
}

impl Job {
    /// An admin cancelled the job, rather than it failing on its own. The worker's progress can't
    /// reword a finished job (`job_message`), so the message stays.
    pub fn cancelled_by_admin(&self) -> bool {
        self.status == "failed" && self.message.as_deref() == Some(CANCELLED_BY_ADMIN)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Deployment {
    pub prompt_id: String,
    pub job_id: String,
    pub model_prefix: String,
    pub threshold: f64,
    pub metrics: Value,
    pub deployed_at: String,
    pub active: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DayUsage {
    pub day: String,
    pub student: u64,
    pub system_one: u64,
    /// Requests the cache answered.
    pub cache: u64,
    pub errors: u64,
    pub latency_ms_sum: f64,
    /// What the System One calls behind `system_one` cost (escalations included), in US dollars,
    /// at each model's `pricing`.
    pub system_one_usd: f64,
}

/// `count` requests of a prompt that took `latency_ms` on `path` (a JSON list of the models that
/// answered, in order) on `day`.
#[derive(Clone, Debug)]
pub struct PathLatency {
    pub prompt_id: String,
    pub day: String,
    pub path: String,
    pub latency_ms: f64,
    pub count: u64,
}

pub struct Db {
    conn: Mutex<Connection>,
}

const SCHEMA: &str = "
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS system_one_models (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS judges (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS prompts (
  id TEXT PRIMARY KEY, spec TEXT NOT NULL, created_at TEXT NOT NULL, last_seen TEXT NOT NULL,
  total INTEGER NOT NULL DEFAULT 0, overrides TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'collecting');
CREATE TABLE IF NOT EXISTS usage_daily (
  prompt_id TEXT NOT NULL, day TEXT NOT NULL, student INTEGER NOT NULL DEFAULT 0, system_one INTEGER NOT NULL DEFAULT 0,
  errors INTEGER NOT NULL DEFAULT 0, latency_ms_sum REAL NOT NULL DEFAULT 0, system_one_usd REAL NOT NULL DEFAULT 0,
  cache INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (prompt_id, day));
CREATE TABLE IF NOT EXISTS path_latencies (
  prompt_id TEXT NOT NULL, day TEXT NOT NULL, path TEXT NOT NULL, latency_ms REAL NOT NULL,
  count INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(prompt_id, day, path, latency_ms));
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, prompt_id TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
  started_at TEXT, finished_at TEXT, message TEXT, spec TEXT NOT NULL, done INTEGER, total INTEGER,
  attempt INTEGER NOT NULL DEFAULT 1, not_before TEXT, last_stage TEXT, stage_done INTEGER, stage_total INTEGER);
CREATE TABLE IF NOT EXISTS deployments (
  prompt_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, model_prefix TEXT NOT NULL, threshold REAL NOT NULL,
  metrics TEXT NOT NULL, deployed_at TEXT NOT NULL, active INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS student_monitors (prompt_id TEXT PRIMARY KEY, value TEXT NOT NULL);
";

/// Columns added to `SCHEMA`'s tables since their first release: (table, column, type).
/// `CREATE TABLE IF NOT EXISTS` leaves an older database's tables as they were, so `Db::open`
/// adds the ones it lacks.
const ADDED_COLUMNS: &[(&str, &str, &str)] = &[
    ("jobs", "done", "INTEGER"),
    ("jobs", "total", "INTEGER"),
    ("jobs", "attempt", "INTEGER NOT NULL DEFAULT 1"),
    ("jobs", "not_before", "TEXT"),
    ("jobs", "last_stage", "TEXT"),
    ("jobs", "stage_done", "INTEGER"),
    ("jobs", "stage_total", "INTEGER"),
    ("usage_daily", "system_one_usd", "REAL NOT NULL DEFAULT 0"),
    ("usage_daily", "cache", "INTEGER NOT NULL DEFAULT 0"),
];

impl Db {
    pub fn open(path: &Path) -> Result<Db> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        conn.execute_batch(SCHEMA)?;
        for (table, column, kind) in ADDED_COLUMNS {
            let has: bool = conn.query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info(?1) WHERE name=?2",
                params![table, column],
                |r| r.get(0),
            )?;
            if !has {
                conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {kind}"))?;
            }
        }
        Ok(Db {
            conn: Mutex::new(conn),
        })
    }

    fn c(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(PoisonError::into_inner)
    }

    // ---- settings -------------------------------------------------------------------------
    pub fn settings(&self) -> Result<Settings> {
        let v: Option<String> = self
            .c()
            .query_row("SELECT value FROM settings WHERE key='settings'", [], |r| {
                r.get(0)
            })
            .optional()?;
        Ok(match v {
            Some(s) => serde_json::from_str(&s)?,
            None => Settings::default(),
        })
    }

    pub fn put_settings(&self, s: &Settings) -> Result<()> {
        self.c().execute(
            "INSERT INTO settings(key, value) VALUES('settings', ?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![serde_json::to_string(s)?],
        )?;
        Ok(())
    }

    // ---- named JSON tables (System One models, judges) ------------------------------------
    fn list_named<T: serde::de::DeserializeOwned>(&self, table: &str) -> Result<Vec<T>> {
        let c = self.c();
        let mut st = c.prepare(&format!("SELECT value FROM {table} ORDER BY name"))?;
        let rows = st.query_map([], |r| r.get::<_, String>(0))?;
        let mut out = vec![];
        for r in rows {
            out.push(serde_json::from_str(&r?)?);
        }
        Ok(out)
    }

    fn put_named<T: Serialize>(&self, table: &str, name: &str, v: &T) -> Result<()> {
        self.c().execute(
            &format!("INSERT INTO {table}(name, value) VALUES(?1, ?2) ON CONFLICT(name) DO UPDATE SET value=excluded.value"),
            params![name, serde_json::to_string(v)?],
        )?;
        Ok(())
    }

    fn delete_named(&self, table: &str, name: &str) -> Result<bool> {
        Ok(self
            .c()
            .execute(&format!("DELETE FROM {table} WHERE name=?1"), params![name])?
            > 0)
    }

    pub fn system_one_models(&self) -> Result<Vec<SystemOneModel>> {
        self.list_named("system_one_models")
    }
    pub fn put_system_one_model(&self, m: &SystemOneModel) -> Result<()> {
        self.put_named("system_one_models", &m.name, m)
    }
    pub fn delete_system_one_model(&self, name: &str) -> Result<bool> {
        self.delete_named("system_one_models", name)
    }
    pub fn judges(&self) -> Result<Vec<JudgeModel>> {
        self.list_named("judges")
    }
    pub fn put_judge(&self, j: &JudgeModel) -> Result<()> {
        self.put_named("judges", &j.name, j)
    }
    pub fn delete_judge(&self, name: &str) -> Result<bool> {
        self.delete_named("judges", name)
    }

    // ---- prompts --------------------------------------------------------------------------
    pub fn prompts(&self) -> Result<Vec<PromptRow>> {
        let c = self.c();
        let mut st = c.prepare("SELECT id, spec, created_at, last_seen, total, overrides, status FROM prompts ORDER BY total DESC")?;
        let rows = st.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, u64>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, String>(6)?,
            ))
        })?;
        let mut out = vec![];
        for r in rows {
            let (id, spec, created_at, last_seen, total, overrides, status) = r?;
            out.push(PromptRow {
                id,
                spec: serde_json::from_str(&spec)?,
                created_at,
                last_seen,
                total,
                overrides: serde_json::from_str(&overrides).unwrap_or_default(),
                status,
            });
        }
        Ok(out)
    }

    pub fn insert_prompt(&self, spec: &PromptSpec) -> Result<()> {
        let t = now();
        self.c().execute(
            "INSERT OR IGNORE INTO prompts(id, spec, created_at, last_seen) VALUES(?1, ?2, ?3, ?3)",
            params![spec.id(), serde_json::to_string(spec)?, t],
        )?;
        Ok(())
    }

    pub fn set_overrides(&self, id: &str, o: &PromptOverrides) -> Result<bool> {
        Ok(self.c().execute(
            "UPDATE prompts SET overrides=?2 WHERE id=?1",
            params![id, serde_json::to_string(o)?],
        )? > 0)
    }

    pub fn set_status(&self, id: &str, status: &str) -> Result<()> {
        self.c().execute(
            "UPDATE prompts SET status=?2 WHERE id=?1",
            params![id, status],
        )?;
        Ok(())
    }

    // ---- usage ----------------------------------------------------------------------------
    /// Add counter deltas (flushed from memory every few seconds).
    pub fn add_usage(&self, deltas: &[(String, String, DayUsage)]) -> Result<()> {
        let mut c = self.c();
        let tx = c.transaction()?;
        let t = now();
        for (prompt_id, day, d) in deltas {
            tx.execute(
                "INSERT INTO usage_daily(prompt_id, day, student, system_one, errors, latency_ms_sum, system_one_usd, cache)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(prompt_id, day) DO UPDATE SET student=student+excluded.student, system_one=system_one+excluded.system_one,
                 errors=errors+excluded.errors, latency_ms_sum=latency_ms_sum+excluded.latency_ms_sum,
                 system_one_usd=system_one_usd+excluded.system_one_usd, cache=cache+excluded.cache",
                params![prompt_id, day, d.student, d.system_one, d.errors, d.latency_ms_sum, d.system_one_usd, d.cache],
            )?;
            tx.execute(
                "UPDATE prompts SET total=total+?2, last_seen=?3 WHERE id=?1",
                params![prompt_id, d.student + d.system_one + d.cache, t],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Daily usage for every prompt since `since_day` (inclusive).
    pub fn usage_since(&self, since_day: &str) -> Result<Vec<(String, DayUsage)>> {
        let c = self.c();
        let mut st = c.prepare(
            "SELECT prompt_id, day, student, system_one, errors, latency_ms_sum, system_one_usd, cache FROM usage_daily
             WHERE day >= ?1 ORDER BY day",
        )?;
        let rows = st.query_map(params![since_day], |r| {
            Ok((
                r.get::<_, String>(0)?,
                DayUsage {
                    day: r.get(1)?,
                    student: r.get(2)?,
                    system_one: r.get(3)?,
                    errors: r.get(4)?,
                    latency_ms_sum: r.get(5)?,
                    system_one_usd: r.get(6)?,
                    cache: r.get(7)?,
                },
            ))
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// A histogram of whole-request latency for each ordered route, retained across restarts.
    pub fn record_path(&self, prompt_id: &str, path: &[String], latency_ms: f64) -> Result<()> {
        let sample = PathLatency {
            prompt_id: prompt_id.into(),
            day: today(),
            path: serde_json::to_string(path)?,
            latency_ms,
            count: 1,
        };
        self.add_path_latencies(&[sample])
    }

    /// Add latency samples (flushed from memory every second) in one transaction. Latencies are
    /// kept to the microsecond.
    pub fn add_path_latencies(&self, samples: &[PathLatency]) -> Result<()> {
        let mut c = self.c();
        let tx = c.transaction()?;
        for x in samples {
            tx.execute(
                "INSERT INTO path_latencies(prompt_id, day, path, latency_ms, count) VALUES(?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(prompt_id, day, path, latency_ms) DO UPDATE SET count=count+excluded.count",
                params![x.prompt_id, x.day, x.path, (x.latency_ms * 1000.0).round() / 1000.0, x.count],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Exact nearest-rank p50 and maximum for each route in the requested window.
    pub fn path_latencies(&self, prompt_id: &str, since: &str) -> Result<Vec<Value>> {
        self.latency_histograms(prompt_id, since, false)
    }

    /// Aggregate the original histogram samples by the last answering model, never averaging medians.
    pub fn answering_model_latencies(&self, prompt_id: &str, since: &str) -> Result<Vec<Value>> {
        self.latency_histograms(prompt_id, since, true)
    }

    fn latency_histograms(
        &self,
        prompt_id: &str,
        since: &str,
        by_model: bool,
    ) -> Result<Vec<Value>> {
        let c = self.c();
        let mut st = c.prepare(
            "SELECT path, latency_ms, SUM(count) FROM path_latencies WHERE prompt_id=?1 AND day>=?2
             GROUP BY path, latency_ms ORDER BY path, latency_ms",
        )?;
        let rows = st.query_map(params![prompt_id, since], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, f64>(1)?,
                r.get::<_, u64>(2)?,
            ))
        })?;
        let mut routes = std::collections::BTreeMap::<String, Vec<(f64, u64)>>::new();
        for row in rows {
            let (path, ms, count) = row?;
            let key = if by_model {
                let route: Vec<String> = serde_json::from_str(&path)?;
                let Some(model) = route.last() else {
                    continue;
                };
                serde_json::to_string(&vec![model])?
            } else {
                path
            };
            routes.entry(key).or_default().push((ms, count));
        }
        let mut out = vec![];
        for (path, mut samples) in routes {
            samples.sort_by(|a, b| a.0.total_cmp(&b.0));
            let count: u64 = samples.iter().map(|(_, n)| n).sum();
            let rank = count.div_ceil(2);
            let mut seen = 0;
            let p50 = samples
                .iter()
                .find(|(_, n)| {
                    seen += n;
                    seen >= rank
                })
                .map(|(ms, _)| ms);
            let route: Value = serde_json::from_str(&path)?;
            out.push(
                serde_json::json!({"path": route, "model": if by_model { route[0].clone() } else { Value::Null }, "count": count,
                "p50_ms": p50, "max_ms": samples.last().map(|(ms, _)| ms)}),
            );
        }
        Ok(out)
    }

    // ---- jobs -----------------------------------------------------------------------------
    /// Queue the first attempt at training a prompt.
    pub fn create_job(&self, prompt_id: &str, spec: &Value) -> Result<Job> {
        self.insert_job(prompt_id, spec, 1, None)
    }

    /// Queue the next attempt after `failed`, which no worker claims before `not_before`.
    pub fn create_retry(&self, failed: &Job, spec: &Value, not_before: &str) -> Result<Job> {
        self.insert_job(
            &failed.prompt_id,
            spec,
            failed.attempt + 1,
            Some(not_before.into()),
        )
    }

    fn insert_job(
        &self,
        prompt_id: &str,
        spec: &Value,
        attempt: u32,
        not_before: Option<String>,
    ) -> Result<Job> {
        let job = Job {
            id: uuid::Uuid::new_v4().simple().to_string()[..12].to_string(),
            prompt_id: prompt_id.into(),
            status: "queued".into(),
            created_at: now(),
            started_at: None,
            finished_at: None,
            message: None,
            done: None,
            total: None,
            last_stage: None,
            stage_done: None,
            stage_total: None,
            spec: spec.clone(),
            attempt,
            not_before,
        };
        self.c().execute(
            "INSERT INTO jobs(id, prompt_id, status, created_at, spec, attempt, not_before)
             VALUES(?1, ?2, 'queued', ?3, ?4, ?5, ?6)",
            params![
                job.id,
                job.prompt_id,
                job.created_at,
                serde_json::to_string(&job.spec)?,
                job.attempt,
                job.not_before
            ],
        )?;
        Ok(job)
    }

    fn job_from_row(r: &rusqlite::Row) -> rusqlite::Result<Job> {
        let spec: String = r.get(7)?;
        Ok(Job {
            id: r.get(0)?,
            prompt_id: r.get(1)?,
            status: r.get(2)?,
            created_at: r.get(3)?,
            started_at: r.get(4)?,
            finished_at: r.get(5)?,
            message: r.get(6)?,
            done: r.get(8)?,
            total: r.get(9)?,
            spec: serde_json::from_str(&spec).unwrap_or(Value::Null),
            attempt: r.get(10)?,
            not_before: r.get(11)?,
            last_stage: r.get(12)?,
            stage_done: r.get(13)?,
            stage_total: r.get(14)?,
        })
    }

    const JOB_COLS: &'static str =
        "id, prompt_id, status, created_at, started_at, finished_at, message, spec, done, total, attempt, not_before, last_stage, stage_done, stage_total";

    pub fn jobs(&self, limit: usize) -> Result<Vec<Job>> {
        let c = self.c();
        let mut st = c.prepare(&format!(
            "SELECT {} FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?1",
            Self::JOB_COLS
        ))?;
        let rows = st.query_map(params![limit], Self::job_from_row)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn jobs_for(&self, prompt_id: &str, limit: usize) -> Result<Vec<Job>> {
        let c = self.c();
        let mut st = c.prepare(&format!(
            "SELECT {} FROM jobs WHERE prompt_id=?1 ORDER BY created_at DESC, rowid DESC LIMIT ?2",
            Self::JOB_COLS
        ))?;
        let rows = st.query_map(params![prompt_id, limit], Self::job_from_row)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn job(&self, id: &str) -> Result<Option<Job>> {
        Ok(self
            .c()
            .query_row(
                &format!("SELECT {} FROM jobs WHERE id=?1", Self::JOB_COLS),
                params![id],
                Self::job_from_row,
            )
            .optional()?)
    }

    pub fn open_job_for(&self, prompt_id: &str) -> Result<Option<Job>> {
        Ok(self
            .c()
            .query_row(
                &format!("SELECT {} FROM jobs WHERE prompt_id=?1 AND status IN ('queued','running') LIMIT 1", Self::JOB_COLS),
                params![prompt_id],
                Self::job_from_row,
            )
            .optional()?)
    }

    pub fn progress_job_for(&self, prompt_id: &str) -> Result<Option<Job>> {
        Ok(self.c().query_row(
            &format!("SELECT {} FROM jobs WHERE prompt_id=?1 ORDER BY CASE WHEN status IN ('queued','running') THEN 0 ELSE 1 END, created_at DESC, rowid DESC LIMIT 1", Self::JOB_COLS),
            params![prompt_id], Self::job_from_row,
        ).optional()?)
    }

    /// Atomically hand the oldest queued job to a worker, leaving the retries whose time hasn't
    /// come in the queue.
    pub fn claim_job(&self) -> Result<Option<Job>> {
        let mut c = self.c();
        let tx = c.transaction()?;
        let job = tx
            .query_row(
                &format!(
                    "SELECT {} FROM jobs WHERE status='queued' AND (not_before IS NULL OR not_before <= ?1)
                     ORDER BY created_at, rowid LIMIT 1",
                    Self::JOB_COLS
                ),
                params![now()],
                Self::job_from_row,
            )
            .optional()?;
        if let Some(j) = &job {
            tx.execute(
                "UPDATE jobs SET status='running', started_at=?2 WHERE id=?1",
                params![j.id, now()],
            )?;
            tx.execute(
                "UPDATE prompts SET status='training' WHERE id=?1",
                params![j.prompt_id],
            )?;
        }
        tx.commit()?;
        Ok(job.map(|mut j| {
            j.status = "running".into();
            j
        }))
    }

    /// A queued retry stops waiting for its time: the next claim may take it. False if the job
    /// isn't a queued retry that waits.
    pub fn skip_wait(&self, id: &str) -> Result<bool> {
        Ok(self.c().execute(
            "UPDATE jobs SET not_before=NULL WHERE id=?1 AND status='queued' AND not_before IS NOT NULL",
            params![id],
        )? > 0)
    }

    /// Finish a queued or running job. False if it had finished already, as when an admin
    /// cancels it while its completion is being handled: whichever came first stands.
    pub fn finish_job(&self, id: &str, ok: bool, message: Option<&str>) -> Result<bool> {
        Ok(self.c().execute(
            "UPDATE jobs SET status=?2, finished_at=?3, message=?4, done=NULL, total=NULL
             WHERE id=?1 AND status IN ('queued','running')",
            params![id, if ok { "done" } else { "failed" }, now(), message],
        )? > 0)
    }

    /// A running job's progress: its message, and `(done, total)` when the worker counts the
    /// stage. A message without counts clears the last ones, so a finished stage's bar doesn't
    /// stay. False if the job isn't running: a cancelled job's worker keeps reporting until it
    /// finishes, and must not replace the reason it was stopped.
    pub fn job_message(&self, id: &str, message: &str, counts: Option<(u64, u64)>) -> Result<bool> {
        let (done, total) = counts.unzip();
        Ok(self.c().execute(
            "UPDATE jobs SET message=?2, done=?3, total=?4, last_stage=COALESCE(?5,last_stage),
             stage_done=CASE WHEN ?5 IS NOT NULL THEN ?3 ELSE stage_done END,
             stage_total=CASE WHEN ?5 IS NOT NULL THEN ?4 ELSE stage_total END WHERE id=?1 AND status='running'",
            params![id, message, done, total, progress_stage(message)],
        )? > 0)
    }

    // ---- deployments ----------------------------------------------------------------------
    pub fn monitor(&self, prompt_id: &str) -> Result<Option<crate::monitor::MonitorState>> {
        let value: Option<String> = self
            .c()
            .query_row(
                "SELECT value FROM student_monitors WHERE prompt_id=?1",
                [prompt_id],
                |r| r.get(0),
            )
            .optional()?;
        value
            .map(|v| serde_json::from_str(&v).map_err(Into::into))
            .transpose()
    }

    pub fn put_monitor(&self, prompt_id: &str, state: &crate::monitor::MonitorState) -> Result<()> {
        self.c().execute(
            "INSERT INTO student_monitors(prompt_id,value) VALUES(?1,?2) ON CONFLICT(prompt_id) DO UPDATE SET value=excluded.value",
            params![prompt_id, serde_json::to_string(state)?],
        )?;
        Ok(())
    }
    pub fn deployments(&self) -> Result<Vec<Deployment>> {
        let c = self.c();
        let mut st = c.prepare("SELECT prompt_id, job_id, model_prefix, threshold, metrics, deployed_at, active FROM deployments")?;
        let rows = st.query_map([], |r| {
            let metrics: String = r.get(4)?;
            Ok(Deployment {
                prompt_id: r.get(0)?,
                job_id: r.get(1)?,
                model_prefix: r.get(2)?,
                threshold: r.get(3)?,
                metrics: serde_json::from_str(&metrics).unwrap_or(Value::Null),
                deployed_at: r.get(5)?,
                active: r.get::<_, i64>(6)? != 0,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn put_deployment(&self, d: &Deployment) -> Result<()> {
        self.c().execute(
            "INSERT INTO deployments(prompt_id, job_id, model_prefix, threshold, metrics, deployed_at, active)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7) ON CONFLICT(prompt_id) DO UPDATE SET job_id=excluded.job_id,
             model_prefix=excluded.model_prefix, threshold=excluded.threshold, metrics=excluded.metrics,
             deployed_at=excluded.deployed_at, active=excluded.active",
            params![d.prompt_id, d.job_id, d.model_prefix, d.threshold, serde_json::to_string(&d.metrics)?, d.deployed_at, d.active],
        )?;
        Ok(())
    }

    pub fn set_deployment_active(&self, prompt_id: &str, active: bool) -> Result<bool> {
        Ok(self.c().execute(
            "UPDATE deployments SET active=?2 WHERE prompt_id=?1",
            params![prompt_id, active],
        )? > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> PromptSpec {
        serde_json::from_value(serde_json::json!({"instructions": "q?", "labels": ["a", "b"]}))
            .unwrap()
    }

    #[test]
    fn jobs_are_claimed_once() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(&dir.path().join("g.db")).unwrap();
        db.insert_prompt(&spec()).unwrap();
        let id = spec().id();
        let j = db.create_job(&id, &serde_json::json!({})).unwrap();
        assert_eq!(db.open_job_for(&id).unwrap().unwrap().id, j.id);
        assert_eq!(db.claim_job().unwrap().unwrap().id, j.id);
        assert!(db.claim_job().unwrap().is_none());
        assert_eq!(db.prompts().unwrap()[0].status, "training");
        db.finish_job(&j.id, true, None).unwrap();
        assert!(db.open_job_for(&id).unwrap().is_none());
    }

    /// verification/JobLifecycle.tla, `FinishedJobsStay`: an admin cancels a running job while
    /// its completion is loading the student, then the completion and the worker's progress
    /// arrive. Neither may turn the cancelled job into a finished one or reword it.
    #[test]
    fn finished_jobs_stay_finished() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(&dir.path().join("g.db")).unwrap();
        db.insert_prompt(&spec()).unwrap();
        let j = db.create_job(&spec().id(), &serde_json::json!({})).unwrap();
        db.claim_job().unwrap();
        db.finish_job(&j.id, false, Some(CANCELLED_BY_ADMIN))
            .unwrap();
        db.finish_job(&j.id, true, Some("test agreement 0.97"))
            .unwrap();
        db.job_message(&j.id, "fine-tuning", Some((5, 50))).unwrap();
        let j = db.job(&j.id).unwrap().unwrap();
        assert_eq!(j.status, "failed");
        assert_eq!(j.message.as_deref(), Some("cancelled by an admin"));
        assert!(j.cancelled_by_admin());
        assert_eq!((j.done, j.total), (None, None));
    }

    /// The worker's counts go with its message: a message without counts clears the last ones,
    /// so the bar of a stage that has ended doesn't stay, and a finished job keeps none.
    #[test]
    fn progress_counts_follow_the_message() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(&dir.path().join("g.db")).unwrap();
        db.insert_prompt(&spec()).unwrap();
        let j = db.create_job(&spec().id(), &serde_json::json!({})).unwrap();
        let progress = |db: &Db| {
            let j = db.job(&j.id).unwrap().unwrap();
            (j.message, j.done, j.total)
        };
        assert!(!db.job_message(&j.id, "judging", Some((0, 20))).unwrap());
        assert_eq!(
            progress(&db),
            (None, None, None),
            "a queued job has no progress"
        );
        db.claim_job().unwrap();
        let judging = "judging 3928 inputs with Mapika/decider-2b";
        assert!(db.job_message(&j.id, judging, Some((1200, 3928))).unwrap());
        assert_eq!(
            progress(&db),
            (Some(judging.into()), Some(1200), Some(3928))
        );
        let listed = &db.jobs(10).unwrap()[0];
        assert_eq!((listed.done, listed.total), (Some(1200), Some(3928)));
        let training = "training a student on google/embeddinggemma-300m with 3928 judged inputs";
        assert!(db.job_message(&j.id, training, None).unwrap());
        assert_eq!(progress(&db), (Some(training.into()), None, None));
        db.job_message(&j.id, training, Some((40, 3300))).unwrap();
        db.finish_job(&j.id, true, Some("test agreement 0.97"))
            .unwrap();
        assert_eq!(
            progress(&db),
            (Some("test agreement 0.97".into()), None, None)
        );
    }

    #[test]
    fn failed_stage_progress_survives_restart_without_resuming_weights() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.db");
        let db = Db::open(&path).unwrap();
        db.insert_prompt(&spec()).unwrap();
        let job = db.create_job(&spec().id(), &serde_json::json!({})).unwrap();
        db.claim_job().unwrap();
        db.job_message(
            &job.id,
            "training a student on private/model",
            Some((35, 100)),
        )
        .unwrap();
        db.finish_job(&job.id, false, Some("private error body"))
            .unwrap();
        drop(db);
        let db = Db::open(&path).unwrap();
        let failed = db.progress_job_for(&spec().id()).unwrap().unwrap();
        assert_eq!(failed.last_stage.as_deref(), Some("training"));
        assert_eq!(
            (failed.stage_done, failed.stage_total),
            (Some(35), Some(100))
        );
        assert_eq!((failed.done, failed.total), (None, None));
        assert_eq!(failed.status, "failed");
    }

    /// A database from before the progress counts and the retries opens: `Db::open` adds the
    /// columns, its jobs read as first attempts and take counts, its queued job can be claimed,
    /// and its settings get the default retries.
    #[test]
    fn an_older_database_gets_the_new_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.db");
        Connection::open(&path)
            .unwrap()
            .execute_batch(
                r#"CREATE TABLE jobs (
                   id TEXT PRIMARY KEY, prompt_id TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                   started_at TEXT, finished_at TEXT, message TEXT, spec TEXT NOT NULL);
                 INSERT INTO jobs VALUES('j1', 'p', 'running', '2026-09-28T09:00:00Z', NULL, NULL, 'judging', '{}');
                 INSERT INTO jobs VALUES('j2', 'q', 'queued', '2026-09-28T09:05:00Z', NULL, NULL, NULL, '{}');
                 CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 INSERT INTO settings VALUES('settings', '{"default_threshold": 500, "default_system_one": null,
                   "default_judge": null, "default_base_model": "auto", "default_target_accuracy": 0.9,
                   "train": {}, "log_inputs": true}');"#,
            )
            .unwrap();
        let db = Db::open(&path).unwrap();
        let j = db.job("j1").unwrap().unwrap();
        assert_eq!(
            (
                j.message.as_deref(),
                j.done,
                j.total,
                j.attempt,
                j.not_before
            ),
            (Some("judging"), None, None, 1, None)
        );
        assert!(db
            .job_message("j1", "judging 20 inputs", Some((3, 20)))
            .unwrap());
        assert_eq!(db.claim_job().unwrap().unwrap().id, "j2");
        let s = db.settings().unwrap();
        assert_eq!(s.default_threshold, 500);
        assert_eq!(s.calibration, CalibrationSettings::default());
        assert_eq!(
            (s.max_attempts, s.retry_secs, s.retry_max_secs),
            (5, 60, 3600)
        );
        drop(db);
        let j = Db::open(&path).unwrap().job("j1").unwrap().unwrap(); // the columns are there now
        assert_eq!((j.done, j.total), (Some(3), Some(20)));
    }

    /// A failed job's retry waits `retry_secs`, twice that after two failures in a row, four
    /// times after three, and so on up to `retry_max_secs`. The last attempt gets no retry.
    #[test]
    fn retry_waits_double_up_to_a_cap() {
        let s = Settings::default();
        let waits: Vec<_> = (1..=5).map(|attempt| s.retry_delay(attempt)).collect();
        assert_eq!(
            waits,
            [Some(60), Some(120), Some(240), Some(480), None],
            "5 attempts: 4 retries"
        );
        let s = Settings {
            max_attempts: 10,
            ..Settings::default()
        };
        assert_eq!(s.retry_delay(6), Some(1920));
        assert_eq!(s.retry_delay(7), Some(3600), "not 3840: an hour at most");
        assert_eq!(s.retry_delay(9), Some(3600));
        assert_eq!(s.retry_delay(10), None);
        let s = Settings {
            max_attempts: u32::MAX,
            ..Settings::default()
        };
        assert_eq!(s.retry_delay(200), Some(3600), "no overflow");
        let s = Settings {
            max_attempts: 1,
            ..Settings::default()
        };
        assert_eq!(s.retry_delay(1), None, "one attempt: never retried");
    }

    /// A retry is queued, and is its prompt's open job, but no claim takes it before its time,
    /// unless it stops waiting; jobs queued after it may go first.
    #[test]
    fn claims_leave_a_retry_until_its_time() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(&dir.path().join("g.db")).unwrap();
        db.insert_prompt(&spec()).unwrap();
        let id = spec().id();
        let first = db.create_job(&id, &serde_json::json!({})).unwrap();
        assert_eq!((first.attempt, first.not_before.as_deref()), (1, None));
        db.claim_job().unwrap();
        db.finish_job(&first.id, false, Some("judge unreachable"))
            .unwrap();
        let first = db.job(&first.id).unwrap().unwrap();
        assert!(!first.cancelled_by_admin(), "it failed on its own");
        let retry = db
            .create_retry(
                &first,
                &serde_json::json!({"reason": "retry"}),
                &secs_from_now(60),
            )
            .unwrap();
        assert_eq!((retry.status.as_str(), retry.attempt), ("queued", 2));
        assert_eq!(db.open_job_for(&id).unwrap().unwrap().id, retry.id);
        assert!(db.claim_job().unwrap().is_none(), "not before its time");

        let other: PromptSpec =
            serde_json::from_value(serde_json::json!({"instructions": "r?", "labels": ["c", "d"]}))
                .unwrap();
        db.insert_prompt(&other).unwrap();
        let later = db.create_job(&other.id(), &serde_json::json!({})).unwrap();
        assert_eq!(db.claim_job().unwrap().unwrap().id, later.id);

        assert!(db.skip_wait(&retry.id).unwrap());
        assert!(!db.skip_wait(&retry.id).unwrap(), "it no longer waits");
        let claimed = db.claim_job().unwrap().unwrap();
        assert_eq!(
            (claimed.id.as_str(), claimed.attempt),
            (retry.id.as_str(), 2)
        );
        assert!(
            !db.skip_wait(&retry.id).unwrap(),
            "a running job doesn't wait"
        );

        // A retry whose time has come is claimed like any queued job.
        db.finish_job(&retry.id, false, Some("judge unreachable"))
            .unwrap();
        let again = db
            .create_retry(&claimed, &serde_json::json!({}), &secs_from_now(0))
            .unwrap();
        assert_eq!(again.attempt, 3);
        assert_eq!(db.claim_job().unwrap().unwrap().id, again.id);
    }

    #[test]
    fn usage_accumulates() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(&dir.path().join("g.db")).unwrap();
        db.insert_prompt(&spec()).unwrap();
        let id = spec().id();
        let d = DayUsage {
            day: today(),
            student: 2,
            system_one: 3,
            errors: 0,
            latency_ms_sum: 5.0,
            system_one_usd: 0.25,
            cache: 1,
        };
        db.add_usage(&[(id.clone(), today(), d.clone()), (id.clone(), today(), d)])
            .unwrap();
        assert_eq!(db.prompts().unwrap()[0].total, 12);
        let u = db.usage_since(&today()).unwrap();
        assert_eq!((u[0].1.student, u[0].1.system_one, u[0].1.cache), (4, 6, 2));
        assert!((u[0].1.system_one_usd - 0.5).abs() < 1e-12);
    }

    /// A database from before costs were counted gets the column, and its days cost nothing.
    #[test]
    fn older_usage_gets_a_cost_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.db");
        Connection::open(&path)
            .unwrap()
            .execute_batch(
                "CREATE TABLE usage_daily (prompt_id TEXT NOT NULL, day TEXT NOT NULL,
                 student INTEGER NOT NULL DEFAULT 0, system_one INTEGER NOT NULL DEFAULT 0,
                 errors INTEGER NOT NULL DEFAULT 0, latency_ms_sum REAL NOT NULL DEFAULT 0,
                 PRIMARY KEY (prompt_id, day));
                 INSERT INTO usage_daily VALUES('p', '2026-10-01', 1, 2, 0, 3.0);",
            )
            .unwrap();
        let u = Db::open(&path).unwrap().usage_since("2026-10-01").unwrap();
        assert_eq!((u[0].1.system_one, u[0].1.system_one_usd), (2, 0.0));
    }

    #[test]
    fn pricing_defaults_to_jev_and_refuses_negative_prices() {
        let m: SystemOneModel = serde_json::from_value(
            serde_json::json!({"kind": "systemone", "base_url": "http://x"}),
        )
        .unwrap();
        assert_eq!(m.pricing, Pricing::default());
        // 300 input tokens, Jev's small call: $0.0000126; output tokens are free.
        assert!((m.pricing.cost(300, 5) - 0.000_012_6).abs() < 1e-15);
        let custom = Pricing {
            usd_per_call: 0.001,
            usd_per_million_input_tokens: 2.0,
            usd_per_million_output_tokens: 10.0,
        };
        assert!((custom.cost(1000, 100) - (0.001 + 0.002 + 0.001)).abs() < 1e-15);
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let m = SystemOneModel {
                pricing: Pricing {
                    usd_per_call: bad,
                    ..Pricing::default()
                },
                ..m.clone()
            };
            assert!(m.check().unwrap_err().contains("pricing"));
        }
    }

    #[test]
    fn denylisted_and_unacknowledged_judges_are_refused() {
        let mut j = JudgeModel {
            name: "j".into(),
            base_url: "http://x".into(),
            model: "typesafe/jev-2".into(),
            api_key_env: None,
            mode: "logprobs".into(),
            samples: 7,
            license_ack: true,
            enabled: true,
        };
        assert!(j.check().unwrap_err().contains("2.3(b)"));
        // Fastino's GLiDE speaks /v1/systemone too, and its terms forbid training competing models.
        j.mode = "systemone".into();
        j.model = "fastino/glide".into();
        assert!(j.check().unwrap_err().contains("§4.A"));
        j.model = "glide".into();
        j.base_url = "https://api.fastino.ai".into();
        assert!(j.check().unwrap_err().contains("§4.A"));
        j.base_url = "http://x".into();
        j.mode = "logprobs".into();
        j.model = "qwen3-32b".into();
        assert!(j.check().is_ok());
        j.mode = "systemone".into(); // Decider's POST /v1/systemone
        assert!(j.check().is_ok());
        j.mode = "letters".into();
        assert!(j.check().unwrap_err().contains("mode"));
        j.mode = "logprobs".into();
        j.license_ack = false;
        assert!(j.check().is_err());
    }

    #[test]
    fn own_credentials_are_never_api_keys() {
        for name in [
            "SN_WORKER_TOKEN",
            "sn_admin_token",
            "AWS_SECRET_ACCESS_KEY",
            "sn_worker_token",
        ] {
            assert!(check_key_env(Some(name)).is_err(), "{name}");
        }
        assert!(check_key_env(Some("OPENAI_API_KEY")).is_ok());
        assert!(check_key_env(None).is_ok());
        let m = SystemOneModel {
            name: "s1".into(),
            kind: "openai_chat".into(),
            base_url: "http://x/v1".into(),
            model: "m".into(),
            api_key_env: Some("SN_ADMIN_TOKEN".into()),
            timeout_ms: 1000,
            logprobs: true,
            enabled: true,
            escalate_to: None,
            escalate_below: None,
            pricing: Pricing::default(),
        };
        assert!(m.check().unwrap_err().contains("SN_ADMIN_TOKEN"));
        let mut m = SystemOneModel {
            api_key_env: None,
            ..m
        };
        for kind in [
            "openai_chat",
            "decision_http",
            "systemone",
            "openai_decisions",
        ] {
            m.kind = kind.into();
            assert!(m.check().is_ok(), "{kind}");
        }
        m.kind = "grpc".into();
        assert!(m.check().unwrap_err().contains("kind"));
    }
}
