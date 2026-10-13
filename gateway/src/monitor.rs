//! Audits, shadow promotion and automatic fallback for each student version.
//!
//! An audit compares the student's prediction for a request with a reference answer:
//!
//! * a **known answer**, when the request's text has a reference label the student was not
//!   trained on (see `references`). It costs nothing, so every such request is checked;
//! * otherwise the prompt's **judge**, the model the student learned from, called in the
//!   background for a random share of requests.
//!
//! The serving fallback (System One) is never the reference: a student trained on one model's
//! answers can't be expected to match a different model's.
//!
//! A new student runs in **shadow** (System One answers) until its confident predictions agree
//! with the reference often enough. Live students keep being audited; one whose agreement is
//! clearly below the bar stops serving and is retrained once. A student that fails its shadow
//! audit stops and waits for an admin: retraining it on the same data would fail the same way.

use std::collections::VecDeque;
use std::sync::{Arc, PoisonError};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::db::JudgeModel;
use crate::references::Reference;
use crate::{db, upstream, Deployed, PromptState, Shared};

/// One-sided z for "clearly below the bar" (99%). A point estimate a little under the bar on a
/// few hundred audits is noise as often as not; this needs the whole likely range to be under it.
const EVIDENCE_Z: f64 = 2.326;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MonitorSettings {
    pub enabled: bool,
    pub sample_rate: f64,
    pub uncertain_sample_rate: f64,
    pub inconclusive_margin: f64,
    pub window: usize,
    pub min_samples: usize,
    pub tolerance: f64,
}

impl Default for MonitorSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_rate: 0.02,
            uncertain_sample_rate: 0.10,
            inconclusive_margin: 0.02,
            window: 500,
            min_samples: 100,
            tolerance: 0.03,
        }
    }
}

impl MonitorSettings {
    pub fn check(&self) -> Result<(), String> {
        if !self.sample_rate.is_finite()
            || !(0.0..=1.0).contains(&self.sample_rate)
            || (self.enabled && self.sample_rate == 0.0)
            || !self.uncertain_sample_rate.is_finite()
            || !(0.0..=1.0).contains(&self.uncertain_sample_rate)
            || !self.inconclusive_margin.is_finite()
            || !(0.0..=0.5).contains(&self.inconclusive_margin)
            || self.min_samples == 0
            || self.min_samples > self.window
            || self.window > 10_000
            || !self.tolerance.is_finite()
            || !(0.0..=0.5).contains(&self.tolerance)
        {
            return Err("monitor: sample_rate in (0, 1] when enabled, uncertain_sample_rate in [0, 1], inconclusive_margin and tolerance in [0, 0.5], 1 <= min_samples <= window <= 10000".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MonitorState {
    pub enabled: bool,
    pub job_id: String,
    /// `shadow`, `live`, `rejected` (failed its shadow audit) or `demoted` (failed while live).
    pub phase: String,
    pub judge: Option<JudgeModel>,
    pub baseline: f64,
    pub audits: u64,
    /// Audits against known answers; the rest were against the judge.
    #[serde(default)]
    pub known_answer_audits: u64,
    pub errors: u64,
    pub skipped: u64,
    pub last_student_label: Option<usize>,
    pub last_judge_label: Option<usize>,
    pub last_confidence: Option<f64>,
    pub updated_at: String,
    /// Only the audited answers the candidate would serve enter its agreement window.
    pub agreements: VecDeque<bool>,
    /// All successful audits, including predictions that routed to fallback.
    /// Older databases have only confident-answer evidence, so this window starts empty.
    #[serde(default)]
    pub traffic_agreements: VecDeque<bool>,
    #[serde(default)]
    pub activity: AuditActivity,
    #[serde(default)]
    pub failure: Option<AuditFailure>,
    /// The judge's version this evidence is against: the model its answers name, which an
    /// alias such as `-latest` resolves to. At first, the version the student learned from.
    #[serde(default)]
    pub judge_version: Option<String>,
    /// The last time the judge's version changed under the same configuration, which started
    /// this test again.
    #[serde(default)]
    pub judge_change: Option<JudgeChange>,
}

/// The judge answered as another version than the student's evidence was against.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct JudgeChange {
    pub from: String,
    pub to: String,
    pub at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditFailure {
    pub phase: String,
    pub agreement: f64,
    pub required_agreement: f64,
    pub samples: usize,
    pub disagreements: usize,
    pub occurred_at: String,
    #[serde(default)]
    pub known_answer_audits: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
struct ActivityBucket {
    second: i64,
    requests: u64,
    selected: u64,
    known: u64,
    completed: u64,
    qualifying: u64,
    errors: u64,
    timeouts: u64,
    skipped: u64,
    latency_ms_sum: u64,
}

/// What audits did in the last five minutes, for the usage page.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditActivity {
    tracking_since: i64,
    last_request_at: Option<i64>,
    last_finished_at: Option<i64>,
    buckets: VecDeque<ActivityBucket>,
    #[serde(skip)]
    in_flight: Vec<i64>,
}

impl Default for AuditActivity {
    fn default() -> Self {
        Self {
            tracking_since: chrono::Utc::now().timestamp(),
            last_request_at: None,
            last_finished_at: None,
            buckets: VecDeque::new(),
            in_flight: vec![],
        }
    }
}

#[derive(Clone, Copy)]
enum ActivityEvent {
    Request,
    /// A judge call was chosen for this request.
    Selected,
    /// Checked against a known answer, at once.
    Known {
        qualifying: bool,
    },
    Skipped,
    Finished {
        error: bool,
        timeout: bool,
        latency_ms: u64,
        qualifying: bool,
    },
}

impl AuditActivity {
    fn record(&mut self, event: ActivityEvent) {
        self.record_at(chrono::Utc::now().timestamp(), event);
    }

    fn record_at(&mut self, now: i64, event: ActivityEvent) {
        while self.buckets.front().is_some_and(|b| b.second <= now - 300) {
            self.buckets.pop_front();
        }
        if self.buckets.back().is_none_or(|b| b.second != now) {
            self.buckets.push_back(ActivityBucket {
                second: now,
                ..Default::default()
            });
        }
        let bucket = self.buckets.back_mut().unwrap();
        match event {
            ActivityEvent::Request => {
                bucket.requests += 1;
                self.last_request_at = Some(now);
            }
            ActivityEvent::Selected => bucket.selected += 1,
            ActivityEvent::Known { qualifying } => {
                bucket.known += 1;
                bucket.qualifying += u64::from(qualifying);
                self.last_finished_at = Some(now);
            }
            ActivityEvent::Skipped => bucket.skipped += 1,
            ActivityEvent::Finished {
                error,
                timeout,
                latency_ms,
                qualifying,
            } => {
                bucket.completed += 1;
                bucket.qualifying += u64::from(qualifying);
                bucket.errors += u64::from(error);
                bucket.timeouts += u64::from(timeout);
                bucket.latency_ms_sum += latency_ms;
                self.last_finished_at = Some(now);
            }
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "recent audit durations and counts are small"
    )]
    fn summary_at(&self, now: i64) -> serde_json::Value {
        let mut counts = ActivityBucket::default();
        for b in self.buckets.iter().filter(|b| b.second > now - 300) {
            counts.requests += b.requests;
            counts.selected += b.selected;
            counts.known += b.known;
            counts.completed += b.completed;
            counts.qualifying += b.qualifying;
            counts.errors += b.errors;
            counts.timeouts += b.timeouts;
            counts.skipped += b.skipped;
            counts.latency_ms_sum += b.latency_ms_sum;
        }
        serde_json::json!({"tracking_since": self.tracking_since, "window_seconds": 300,
            "requests": counts.requests, "selected": counts.selected, "known_answers": counts.known,
            "completed": counts.completed, "errors": counts.errors, "timeouts": counts.timeouts,
            "skipped": counts.skipped, "qualifying": counts.qualifying,
            "in_flight": self.in_flight.len(), "oldest_in_flight_ms": self.in_flight.iter().min().map(|t| (now * 1000 - t).max(0)),
            "avg_judge_latency_ms": (counts.completed > 0).then(|| counts.latency_ms_sum as f64 / counts.completed as f64),
            "last_request_at": self.last_request_at, "last_finished_at": self.last_finished_at})
    }
}

/// Upper end of the one-sided Wilson score interval for `k` agreements in `n` audits.
#[expect(
    clippy::cast_precision_loss,
    reason = "agreement windows are capped at 10000 samples"
)]
fn wilson_upper(k: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let (n, p, z2) = (n as f64, k as f64 / n as f64, EVIDENCE_Z * EVIDENCE_Z);
    let centre = p + z2 / (2.0 * n);
    let spread = EVIDENCE_Z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    ((centre + spread) / (1.0 + z2 / n)).min(1.0)
}

impl MonitorState {
    pub fn summary(&self, cfg: &MonitorSettings) -> serde_json::Value {
        let n = self.agreements.len();
        let k = self.agreements.iter().filter(|&&a| a).count();
        serde_json::json!({"job_id": self.job_id, "phase": self.phase, "enabled": self.enabled,
            "judge": self.judge.as_ref().map(|j| &j.model),
            "failure": self.failure, "baseline": self.baseline, "agreement": self.agreement(),
            "agreement_upper": (n > 0).then(|| wilson_upper(k, n)), "samples": n,
            "verdict": self.verdict(cfg),
            "audits": self.audits, "known_answer_audits": self.known_answer_audits,
            "judge_audits": self.audits.saturating_sub(self.known_answer_audits),
            "errors": self.errors, "skipped": self.skipped,
            "traffic_samples": self.traffic_agreements.len(), "traffic_agreement": agreement(&self.traffic_agreements),
            "effective_sample_rate": self.effective_sample_rate(cfg), "evidence": self.evidence(cfg),
            "base_sample_rate": cfg.sample_rate,
            "activity": self.activity.summary_at(chrono::Utc::now().timestamp()),
            "required_samples": cfg.min_samples, "max_samples": cfg.window,
            "agreement_boundary": self.required(cfg),
            "last_student_label": self.last_student_label, "last_reference_label": self.last_judge_label,
            "last_confidence": self.last_confidence, "updated_at": self.updated_at,
            "judge_version": self.judge_version, "judge_change": self.judge_change})
    }

    pub fn new(job_id: &str, baseline: f64, judge: Option<JudgeModel>, enabled: bool) -> Self {
        Self {
            enabled,
            job_id: job_id.into(),
            phase: if enabled { "shadow" } else { "live" }.into(),
            judge,
            baseline,
            audits: 0,
            known_answer_audits: 0,
            errors: 0,
            skipped: 0,
            last_student_label: None,
            last_judge_label: None,
            last_confidence: None,
            updated_at: db::now(),
            agreements: VecDeque::new(),
            traffic_agreements: VecDeque::new(),
            activity: AuditActivity::default(),
            failure: None,
            judge_version: None,
            judge_change: None,
        }
    }

    /// A deployment's first evidence: against the judge's version the student learned from, as
    /// its training reported it (`judge_version` in its metrics).
    pub fn for_deployment(
        d: &db::Deployment,
        baseline: f64,
        judge: Option<JudgeModel>,
        enabled: bool,
    ) -> Self {
        let mut m = Self::new(&d.job_id, baseline, judge, enabled);
        m.judge_version = d.metrics["judge_version"].as_str().map(String::from);
        m
    }

    /// The judge now answers as `version`. A first version is taken as the one this evidence is
    /// against. Another one is a new training target, so the evidence starts again, in shadow:
    /// the student is tested against what the judge answers now. Says whether it started again.
    pub fn judge_answered_as(&mut self, version: &str) -> bool {
        let Some(old) = self.judge_version.clone() else {
            self.judge_version = Some(version.into());
            return false;
        };
        if old == version {
            return false;
        }
        let mut fresh = MonitorState::new(
            &self.job_id,
            self.baseline,
            self.judge.clone(),
            self.enabled,
        );
        fresh.activity = std::mem::take(&mut self.activity);
        fresh.judge_version = Some(version.into());
        fresh.judge_change = Some(JudgeChange {
            from: old,
            to: version.into(),
            at: db::now(),
        });
        *self = fresh;
        true
    }

    pub fn agreement(&self) -> Option<f64> {
        agreement(&self.agreements)
    }

    /// The agreement a student's confident answers must show.
    pub fn required(&self, cfg: &MonitorSettings) -> f64 {
        self.baseline - cfg.tolerance
    }

    pub fn stopped(&self) -> bool {
        self.phase == "demoted" || self.phase == "rejected"
    }

    pub fn evidence(&self, cfg: &MonitorSettings) -> &'static str {
        if !cfg.enabled || !self.enabled {
            return "disabled";
        }
        if self.stopped() {
            return "stopped";
        }
        if self.agreements.len() < cfg.min_samples {
            return "insufficient";
        }
        if self
            .agreement()
            .is_some_and(|a| a <= self.required(cfg) + cfg.inconclusive_margin)
        {
            return "near_boundary";
        }
        "healthy"
    }

    /// Where the evidence stands: `collecting` (too few audits), `passing`, `inconclusive`
    /// (under the bar, but it could be noise) or `failing` (clearly under the bar).
    pub fn verdict(&self, cfg: &MonitorSettings) -> &'static str {
        let n = self.agreements.len();
        if n < cfg.min_samples {
            return "collecting";
        }
        let k = self.agreements.iter().filter(|&&a| a).count();
        if self.agreement().unwrap_or_default() >= self.required(cfg) {
            "passing"
        } else if wilson_upper(k, n) < self.required(cfg) {
            "failing"
        } else {
            "inconclusive"
        }
    }

    /// Share of requests without a known answer that get a judge call.
    pub fn effective_sample_rate(&self, cfg: &MonitorSettings) -> f64 {
        match self.evidence(cfg) {
            "disabled" | "stopped" => 0.0,
            "insufficient" | "near_boundary" => cfg.sample_rate.max(cfg.uncertain_sample_rate),
            _ => cfg.sample_rate,
        }
    }

    pub fn observe(
        &mut self,
        student: usize,
        confidence: f64,
        teacher: usize,
        threshold: f64,
        cfg: &MonitorSettings,
    ) {
        if self.stopped() {
            return;
        }
        self.audits += 1;
        self.last_student_label = Some(student);
        self.last_judge_label = Some(teacher);
        self.last_confidence = Some(confidence);
        self.updated_at = db::now();
        self.traffic_agreements.push_back(student == teacher);
        while self.traffic_agreements.len() > cfg.window {
            self.traffic_agreements.pop_front();
        }
        if confidence < threshold {
            return;
        }
        self.agreements.push_back(student == teacher);
        while self.agreements.len() > cfg.window {
            self.agreements.pop_front();
        }
        let failed = match (self.phase.as_str(), self.verdict(cfg)) {
            ("shadow", "passing") => {
                self.phase = "live".into();
                // Subsequent drift is evaluated against fresh traffic, not the promotion sample.
                self.agreements.clear();
                return;
            }
            // A shadow audit decides by the end of its window at the latest.
            ("shadow", "inconclusive") => self.agreements.len() >= cfg.window,
            (_, "failing") => true,
            _ => false,
        };
        if failed {
            self.failure = Some(AuditFailure {
                phase: self.phase.clone(),
                agreement: self.agreement().unwrap_or_default(),
                required_agreement: self.required(cfg),
                samples: self.agreements.len(),
                disagreements: self.agreements.iter().filter(|&&a| !a).count(),
                occurred_at: db::now(),
                known_answer_audits: self.known_answer_audits,
            });
            self.phase = if self.phase == "shadow" {
                "rejected"
            } else {
                "demoted"
            }
            .into();
        }
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "agreement windows are capped at 10000 samples"
)]
fn agreement(window: &VecDeque<bool>) -> Option<f64> {
    (!window.is_empty()).then(|| window.iter().filter(|&&a| a).count() as f64 / window.len() as f64)
}

/// A student's answer to a request: its label, confidence, and the threshold of the student that
/// gave it (with students per language, each language's own).
#[derive(Clone, Copy, Debug)]
pub struct Prediction {
    pub label: usize,
    pub confidence: f64,
    pub threshold: f64,
}

/// Audit the student's prediction for `text`, and say whether the student may answer it.
/// Known answers are checked at once; judge calls run in the background, bounded, and never
/// delay a response.
pub fn audit(
    s: &Shared,
    p: &Arc<PromptState>,
    dep: &Arc<Deployed>,
    text: &str,
    prediction: &Prediction,
) -> bool {
    let Prediction {
        label,
        confidence,
        threshold,
    } = *prediction;
    let settings = s.settings();
    let judge = s.judge_for(p);
    let reference = dep.references.lookup(text);
    let _guard = s.queue_lock.lock().unwrap_or_else(PoisonError::into_inner);
    // Inference may have overlapped a swap or an admin's undeploy.
    if !p
        .deployed()
        .is_some_and(|current| Arc::ptr_eq(&current, dep))
    {
        return false;
    }
    let mut state = dep.monitor.lock().unwrap_or_else(PoisonError::into_inner);
    if state.enabled != settings.monitor.enabled || state.judge != judge {
        // Evidence gathered under another policy or judge doesn't carry over.
        *state = MonitorState::new(
            &dep.job_id,
            state.baseline,
            judge.clone(),
            settings.monitor.enabled,
        );
        if let Err(e) = s.db.put_monitor(&p.id, &state) {
            tracing::error!("saving monitor: {e:#}");
        }
        if s.db.open_job_for(&p.id).is_ok_and(|job| job.is_none()) {
            let status = if state.phase == "shadow" {
                "shadow"
            } else {
                "deployed"
            };
            if let Err(e) = s.db.set_status(&p.id, status) {
                tracing::error!("saving monitor status: {e:#}");
            } else {
                p.set_status(status);
            }
        }
    }
    if !settings.monitor.enabled {
        return true;
    }
    state.activity.record(ActivityEvent::Request);
    let live = state.phase == "live";
    match reference {
        Reference::Trained => live,
        Reference::Known(answer) => {
            let before = state.phase.clone();
            state.known_answer_audits += 1;
            state.activity.record(ActivityEvent::Known {
                qualifying: confidence >= threshold,
            });
            state.observe(label, confidence, answer, threshold, &settings.monitor);
            if let Err(e) = settle(s, p, dep, &state, &before) {
                tracing::error!("prompt {}: recording audit: {e:#}", p.id);
            }
            live
        }
        Reference::Judge => {
            // Without a judge there is no evidence: the candidate stays in shadow.
            let Some(judge) = judge else {
                return false;
            };
            if rand::random::<f64>() >= state.effective_sample_rate(&settings.monitor) {
                return live;
            }
            state.activity.record(ActivityEvent::Selected);
            let Ok(permit) = s.audit_slots.clone().try_acquire_owned() else {
                state.skipped += 1;
                state.activity.record(ActivityEvent::Skipped);
                return live;
            };
            let started_at = chrono::Utc::now().timestamp_millis();
            state.activity.in_flight.push(started_at);
            let (s, p, dep, text, prediction) = (
                s.clone(),
                p.clone(),
                dep.clone(),
                text.to_string(),
                *prediction,
            );
            tokio::spawn(async move {
                let _permit = permit;
                let answer = tokio::time::timeout(
                    std::time::Duration::from_secs(30),
                    upstream::judge(&s.http, &judge, &p.spec, &text),
                )
                .await
                .map_err(anyhow::Error::from)
                .and_then(std::convert::identity);
                let result = JudgeResult { answer, started_at };
                if let Err(e) = finish_audit(&s, &p, &dep, &judge, &prediction, result) {
                    tracing::error!("prompt {}: recording audit: {e:#}", p.id);
                }
            });
            live
        }
    }
}

struct JudgeResult {
    answer: Result<upstream::Answer>,
    started_at: i64,
}

fn finish_audit(
    s: &Shared,
    p: &Arc<PromptState>,
    dep: &Arc<Deployed>,
    judge: &JudgeModel,
    prediction: &Prediction,
    result: JudgeResult,
) -> Result<()> {
    let Prediction {
        label,
        confidence,
        threshold,
    } = *prediction;
    let _guard = s.queue_lock.lock().unwrap_or_else(PoisonError::into_inner);
    // A late result cannot change an undeployed or replaced student, or compare different judges.
    if !p
        .deployed()
        .is_some_and(|current| Arc::ptr_eq(&current, dep))
        || s.judge_for(p).as_ref() != Some(judge)
    {
        return Ok(());
    }
    let settings = s.settings();
    if !settings.monitor.enabled {
        return Ok(());
    }
    let mut state = dep.monitor.lock().unwrap_or_else(PoisonError::into_inner);
    if state.judge.as_ref() != Some(judge) {
        return Ok(());
    }
    if let Some(i) = state
        .activity
        .in_flight
        .iter()
        .position(|t| *t == result.started_at)
    {
        state.activity.in_flight.remove(i);
    }
    // The judge answering as another version (an alias such as `-latest` moved) is a new
    // training target: the student is tested again, in shadow, against what it answers now.
    if let Some(version) = result.answer.as_ref().ok().and_then(|a| a.model.as_deref()) {
        let from = state.judge_version.clone().unwrap_or_default();
        let was_live = state.phase == "live";
        if state.judge_answered_as(version) {
            tracing::warn!(
                "prompt {}: the judge {} now answers as {version} (was {from}); testing student {} again",
                p.id,
                judge.model,
                dep.job_id
            );
            if was_live && s.db.open_job_for(&p.id)?.is_none() {
                s.db.set_status(&p.id, "shadow")?;
                p.set_status("shadow");
            }
        }
    }
    let timeout = result.answer.as_ref().err().is_some_and(|e| {
        e.is::<tokio::time::error::Elapsed>()
            || e.downcast_ref::<reqwest::Error>()
                .is_some_and(reqwest::Error::is_timeout)
    });
    state.activity.record(ActivityEvent::Finished {
        error: result.answer.is_err(),
        timeout,
        latency_ms: u64::try_from(chrono::Utc::now().timestamp_millis() - result.started_at)
            .unwrap_or(0),
        qualifying: result.answer.is_ok() && confidence >= threshold,
    });
    let before = state.phase.clone();
    if let Ok(a) = result.answer {
        state.observe(label, confidence, a.label, threshold, &settings.monitor);
    } else {
        state.errors += 1;
        state.updated_at = db::now();
    }
    settle(s, p, dep, &state, &before)
}

/// Save the audit, and act on a change of phase. Called with `queue_lock` held.
fn settle(
    s: &Shared,
    p: &Arc<PromptState>,
    dep: &Arc<Deployed>,
    state: &MonitorState,
    before: &str,
) -> Result<()> {
    s.db.put_monitor(&p.id, state)?;
    if state.phase == before {
        return Ok(());
    }
    if state.stopped() {
        s.db.set_deployment_active(&p.id, false)?;
        *p.deployed.write().unwrap_or_else(PoisonError::into_inner) = None;
        let (agreement, required) = state
            .failure
            .as_ref()
            .map_or((0.0, 0.0), |f| (f.agreement, f.required_agreement));
        if state.phase == "demoted" {
            s.db.set_status(&p.id, "drifted")?;
            p.set_status("drifted");
            tracing::warn!(
                "prompt {}: student {} stopped serving: live agreement {:.1}% after {} audits is clearly below the required {:.1}%; retraining",
                p.id,
                dep.job_id,
                agreement * 100.0,
                state.audits,
                required * 100.0
            );
            s.queue_job_locked(p, "drift")?;
        } else {
            // The same data would train the same student: an admin decides what changes.
            if s.db.open_job_for(&p.id)?.is_none() {
                s.db.set_status(&p.id, "rejected")?;
                p.set_status("rejected");
            }
            tracing::warn!(
                "prompt {}: student {} failed its shadow audit ({:.1}% agreement, {:.1}% required, after {} audits); System One keeps answering",
                p.id,
                dep.job_id,
                agreement * 100.0,
                required * 100.0,
                state.audits
            );
        }
    } else if before == "shadow" && state.phase == "live" {
        // A manual retraining job may already own the prompt's queued/training status.
        if s.db.open_job_for(&p.id)?.is_none() {
            s.db.set_status(&p.id, "deployed")?;
            p.set_status("deployed");
        }
        tracing::info!(
            "prompt {}: student {} passed its shadow audit and now answers",
            p.id,
            dep.job_id
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An alias such as `-latest` resolving to another version is a new training target: the
    /// evidence starts again in shadow, keeping the activity the usage page shows.
    #[test]
    fn a_new_judge_version_starts_the_test_again() {
        let mut state = MonitorState::new("job", 0.9, None, true);
        assert!(!state.judge_answered_as("judge-2026-09"));
        assert_eq!(state.judge_version.as_deref(), Some("judge-2026-09"));
        state.phase = "live".into();
        state.audits = 40;
        state.agreements.extend([true; 40]);
        state.activity.record(ActivityEvent::Request);
        assert!(!state.judge_answered_as("judge-2026-09"));
        assert_eq!(state.phase, "live");
        assert!(state.judge_answered_as("judge-2026-10"));
        assert_eq!(
            (state.phase.as_str(), state.audits, state.agreements.len()),
            ("shadow", 0, 0)
        );
        assert_eq!(state.judge_version.as_deref(), Some("judge-2026-10"));
        let change = state.judge_change.clone().unwrap();
        assert_eq!(
            (change.from.as_str(), change.to.as_str()),
            ("judge-2026-09", "judge-2026-10")
        );
        assert!(state.activity.last_request_at.is_some());
        // The version a deployment's training reported is the first one.
        let d = db::Deployment {
            prompt_id: "p".into(),
            job_id: "job".into(),
            model_prefix: "m".into(),
            threshold: 0.5,
            metrics: serde_json::json!({"judge_version": "judge-2026-09"}),
            deployed_at: db::now(),
            active: true,
        };
        let mut state = MonitorState::for_deployment(&d, 0.9, None, true);
        assert!(state.judge_answered_as("judge-2026-10"));
    }

    #[test]
    fn activity_reports_recent_load_and_resets_pending_calls_on_restart() {
        let mut activity = AuditActivity::default();
        activity.record_at(600, ActivityEvent::Request);
        activity.record_at(1000, ActivityEvent::Request);
        activity.record_at(1000, ActivityEvent::Selected);
        activity.record_at(1000, ActivityEvent::Skipped);
        activity.record_at(1000, ActivityEvent::Known { qualifying: true });
        activity.record_at(
            1000,
            ActivityEvent::Finished {
                error: false,
                timeout: false,
                latency_ms: 1200,
                qualifying: true,
            },
        );
        activity.record_at(
            1000,
            ActivityEvent::Finished {
                error: true,
                timeout: true,
                latency_ms: 30000,
                qualifying: false,
            },
        );
        activity.in_flight.push(999_000);
        let summary = activity.summary_at(1001);
        assert_eq!(summary["requests"], 1);
        assert_eq!(summary["selected"], 1);
        assert_eq!(summary["known_answers"], 1);
        assert_eq!(summary["qualifying"], 2);
        assert_eq!(summary["errors"], 1);
        assert_eq!(summary["timeouts"], 1);
        assert_eq!(summary["skipped"], 1);
        assert_eq!(summary["in_flight"], 1);
        assert_eq!(summary["oldest_in_flight_ms"], 2000);
        assert_eq!(summary["avg_judge_latency_ms"], 15600.0);
        let restored: AuditActivity =
            serde_json::from_value(serde_json::to_value(activity).unwrap()).unwrap();
        assert_eq!(restored.summary_at(1001)["in_flight"], 0);
        assert_eq!(restored.summary_at(1300)["requests"], 0);
        assert_eq!(restored.summary_at(1300)["last_request_at"], 1000);
    }

    #[test]
    fn summary_reports_the_bar_and_where_the_evidence_stands() {
        let state = MonitorState::new("job", 0.95, None, true);
        let cfg = MonitorSettings {
            min_samples: 25,
            tolerance: 0.05,
            ..Default::default()
        };
        let summary = state.summary(&cfg);
        assert_eq!(summary["required_samples"], 25);
        assert_eq!(summary["max_samples"], 500);
        assert!((summary["base_sample_rate"].as_f64().unwrap() - cfg.sample_rate).abs() < 1e-10);
        assert!((summary["agreement_boundary"].as_f64().unwrap() - 0.9).abs() < 1e-10);
        assert_eq!(summary["samples"], 0);
        assert_eq!(summary["phase"], "shadow");
        assert_eq!(summary["verdict"], "collecting");
    }

    #[test]
    fn wilson_upper_bound() {
        assert!((wilson_upper(85, 100) - 0.915).abs() < 0.002);
        assert!(wilson_upper(85, 500) < 0.89);
        assert!((wilson_upper(0, 0) - 1.0).abs() < 1e-12);
        assert!((wilson_upper(10, 10) - 1.0).abs() < 1e-12);
    }

    /// 85% agreement against a required 87% is inconclusive after 100 audits (it could be
    /// noise), and only fails a shadow audit once the window is full.
    #[test]
    fn slightly_low_agreement_needs_more_evidence_before_failing() {
        let cfg = MonitorSettings::default();
        let mut state = MonitorState::new("job", 0.9, None, true);
        for i in 0..100 {
            state.observe(0, 0.9, usize::from(i % 20 < 3), 0.8, &cfg);
        }
        assert_eq!(state.phase, "shadow");
        assert_eq!(state.verdict(&cfg), "inconclusive");
        for i in 0..400 {
            state.observe(0, 0.9, usize::from(i % 20 < 3), 0.8, &cfg);
        }
        assert_eq!(state.phase, "rejected");
        let failure = state.failure.as_ref().unwrap();
        assert_eq!((failure.phase.as_str(), failure.samples), ("shadow", 500));
    }

    #[test]
    fn failure_snapshot_preserves_why_the_student_stopped() {
        let mut state = MonitorState::new("job", 0.95, None, true);
        let cfg = MonitorSettings {
            min_samples: 20,
            window: 20,
            ..Default::default()
        };
        for teacher in (0..20).map(|i| usize::from(i % 2 == 0)) {
            state.observe(0, 0.9, teacher, 0.7, &cfg);
        }
        assert_eq!(state.phase, "rejected");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        {
            let db = db::Db::open(&path).unwrap();
            db.put_monitor("prompt", &state).unwrap();
        }
        let restored = db::Db::open(&path)
            .unwrap()
            .monitor("prompt")
            .unwrap()
            .unwrap();
        let changed = MonitorSettings {
            tolerance: 0.5,
            ..cfg
        };
        let failure = &restored.summary(&changed)["failure"];
        assert_eq!(failure["phase"], "shadow");
        assert_eq!(failure["samples"], 20);
        assert_eq!(failure["disagreements"], 10);
        assert!((failure["required_agreement"].as_f64().unwrap() - 0.92).abs() < 1e-10);
        // A stopped student takes no more evidence.
        let mut stopped = restored;
        stopped.observe(0, 0.9, 0, 0.7, &cfg);
        assert_eq!(stopped.audits, 20);
    }

    #[test]
    fn shadow_then_live_then_drift_on_fresh_evidence() {
        let cfg = MonitorSettings {
            min_samples: 3,
            window: 30,
            tolerance: 0.0,
            ..Default::default()
        };
        let mut state = MonitorState::new("job", 1.0, None, true);
        // Audits are collected for low-confidence predictions too, but they don't qualify for promotion.
        state.observe(0, 0.2, 1, 0.8, &cfg);
        assert_eq!(state.audits, 1);
        assert_eq!(state.agreements.len(), 0);
        assert_eq!(agreement(&state.traffic_agreements), Some(0.0));
        for _ in 0..2 {
            state.observe(0, 0.9, 0, 0.8, &cfg);
        }
        assert_eq!(state.phase, "shadow");
        state.observe(0, 0.9, 0, 0.8, &cfg);
        assert_eq!(state.phase, "live");
        assert!(state.agreements.is_empty());
        // Three disagreements in a row are clearly below a bar of 100%.
        for _ in 0..2 {
            state.observe(0, 0.9, 1, 0.8, &cfg);
        }
        assert_eq!(state.phase, "live");
        state.observe(0, 0.9, 1, 0.8, &cfg);
        assert_eq!(state.phase, "demoted");
        assert_eq!(state.failure.as_ref().unwrap().phase, "live");
    }

    #[test]
    fn a_live_student_is_not_demoted_on_noise() {
        let cfg = MonitorSettings::default();
        let mut state = MonitorState::new("job", 0.9, None, true);
        state.phase = "live".into();
        // 86% on a bar of 87%, for thousands of audits: noise, not drift.
        for i in 0..5000 {
            state.observe(0, 0.9, usize::from(i % 50 < 7), 0.8, &cfg);
        }
        assert_eq!(state.phase, "live");
        // 70% is drift.
        for i in 0..500 {
            state.observe(0, 0.9, usize::from(i % 10 < 3), 0.8, &cfg);
        }
        assert_eq!(state.phase, "demoted");
    }

    #[test]
    fn weak_shadow_is_never_promoted_and_window_is_bounded() {
        let cfg = MonitorSettings {
            min_samples: 2,
            window: 3,
            ..Default::default()
        };
        let mut state = MonitorState::new("job", 0.95, None, true);
        for _ in 0..5 {
            state.observe(0, 0.2, 1, 0.8, &cfg);
        }
        // Low-confidence audits fill only the all-traffic window, which is bounded too.
        assert_eq!(state.traffic_agreements.len(), 3);
        assert_eq!(state.phase, "shadow");
        for _ in 0..5 {
            state.observe(0, 0.9, 1, 0.8, &cfg);
        }
        // Two confident disagreements out of two are clearly under 92%: no more evidence needed.
        assert_eq!(state.phase, "rejected");
        assert_eq!(state.agreements.len(), 2);
        assert_eq!(state.audits, 7);
    }

    #[test]
    fn monitoring_settings_reject_unusable_evidence_policies() {
        assert!(MonitorSettings::default().check().is_ok());
        for patch in [
            serde_json::json!({"sample_rate": 0}),
            serde_json::json!({"min_samples": 501}),
            serde_json::json!({"window": 10001}),
            serde_json::json!({"tolerance": -0.1}),
            serde_json::json!({"uncertain_sample_rate": 1.1}),
            serde_json::json!({"inconclusive_margin": -0.1}),
        ] {
            let cfg: MonitorSettings = serde_json::from_value(patch).unwrap();
            assert!(cfg.check().is_err());
        }
    }

    #[test]
    fn audit_window_and_phase_survive_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let mut state = MonitorState::new("job", 0.95, None, true);
        state.known_answer_audits = 1;
        state.observe(0, 0.9, 0, 0.8, &MonitorSettings::default());
        {
            let db = db::Db::open(&path).unwrap();
            db.put_monitor("prompt", &state).unwrap();
        }
        let restored = db::Db::open(&path)
            .unwrap()
            .monitor("prompt")
            .unwrap()
            .unwrap();
        assert_eq!(restored.phase, "shadow");
        assert_eq!(restored.agreements, state.agreements);
        assert_eq!(restored.traffic_agreements, state.traffic_agreements);
        assert_eq!((restored.audits, restored.known_answer_audits), (1, 1));
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "rates are selected configuration constants"
    )]
    fn judge_audit_share_adapts_to_prior_evidence_and_never_drops_below_base() {
        let cfg = MonitorSettings {
            min_samples: 100,
            window: 100,
            tolerance: 0.1,
            inconclusive_margin: 0.05,
            ..Default::default()
        };
        let mut state = MonitorState::new("job", 0.9, None, true);
        assert_eq!(state.effective_sample_rate(&cfg), 0.10);
        let disabled = MonitorSettings {
            enabled: false,
            ..cfg.clone()
        };
        assert_eq!(state.effective_sample_rate(&disabled), 0.0);
        for _ in 0..100 {
            state.observe(0, 0.9, 0, 0.8, &cfg);
        }
        assert_eq!(state.phase, "live");
        // Promotion starts fresh evidence, so auditing remains elevated.
        assert_eq!(state.effective_sample_rate(&cfg), 0.10);
        assert_eq!(state.evidence(&cfg), "insufficient");
        for _ in 0..100 {
            state.observe(0, 0.9, 0, 0.8, &cfg);
        }
        assert_eq!(state.effective_sample_rate(&cfg), 0.02);
        for _ in 0..16 {
            state.observe(0, 0.9, 1, 0.8, &cfg);
        }
        assert_eq!(state.phase, "live");
        assert_eq!(state.evidence(&cfg), "near_boundary");
        assert_eq!(state.effective_sample_rate(&cfg), 0.10);
        let high_base = MonitorSettings {
            sample_rate: 0.5,
            ..cfg.clone()
        };
        assert_eq!(state.effective_sample_rate(&high_base), 0.5);
        for _ in 0..30 {
            state.observe(0, 0.9, 1, 0.8, &cfg);
        }
        assert_eq!(state.phase, "demoted");
        assert_eq!(state.effective_sample_rate(&cfg), 0.0);
    }

    #[test]
    fn fallback_predictions_have_separate_bounded_traffic_evidence() {
        let cfg = MonitorSettings {
            window: 3,
            min_samples: 3,
            ..Default::default()
        };
        let mut state = MonitorState::new("job", 1.0, None, true);
        for teacher in [0, 0, 1, 1] {
            state.observe(0, 0.2, teacher, 0.8, &cfg);
        }
        assert_eq!(state.agreement(), None);
        assert_eq!(state.phase, "shadow");
        assert_eq!(state.traffic_agreements.len(), 3);
        assert_eq!(agreement(&state.traffic_agreements), Some(1.0 / 3.0));
        assert_eq!(state.last_student_label, Some(0));
        assert_eq!(state.last_judge_label, Some(1));
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "checking exact deserialized default constants"
    )]
    fn existing_settings_and_monitor_state_load_with_new_defaults() {
        let cfg: MonitorSettings =
            serde_json::from_value(serde_json::json!({"sample_rate": 0.02})).unwrap();
        assert_eq!(cfg.uncertain_sample_rate, 0.10);
        assert_eq!(cfg.inconclusive_margin, 0.02);
        let state = MonitorState::new("old", 0.95, None, true);
        let mut saved = serde_json::to_value(state).unwrap();
        let fields = saved.as_object_mut().unwrap();
        fields.remove("traffic_agreements");
        fields.remove("known_answer_audits");
        // Saved by the version that audited against System One.
        fields.insert("reference".into(), serde_json::json!("{}"));
        let restored: MonitorState = serde_json::from_value(saved).unwrap();
        assert!(restored.traffic_agreements.is_empty());
        assert_eq!(restored.known_answer_audits, 0);
        assert_eq!(restored.effective_sample_rate(&cfg), 0.10);
    }
}
