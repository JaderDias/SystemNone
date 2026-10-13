//! HTTP routes.
//!
//! Public:   POST /v1/classify · GET /health
//! Usage UI: GET / · GET /api/summary · GET /api/prompts · GET /api/prompts/{id}   (optional viewer token)
//! Training: GET /jobs · GET /api/jobs                                              (optional viewer token)
//! Admin:    GET /admin · /admin/api/... · POST /admin/api/labels (reference labels)    (admin token)
//!           GET and DELETE /admin/api/cache (the cache's numbers; empty it)
//! Worker:   POST /internal/jobs/claim · /internal/jobs/{id}/progress · /internal/jobs/{id}/complete

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, PoisonError};
use std::time::Instant;

use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::cache::CachedAnswer;
use crate::db::{self, Deployment, JudgeModel, PromptOverrides, Settings, SystemOneModel};
use crate::prompt::PromptSpec;
use crate::{upstream, Deployed, PromptState, Shared};

const INDEX_HTML: &str = include_str!("ui/index.html");
const FAVICON_SVG: &str = include_str!("ui/favicon.svg");
const ADMIN_HTML: &str = include_str!("ui/admin.html");

pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}

impl<E: std::fmt::Display> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    }
}

pub(crate) fn err(code: StatusCode, msg: impl Into<String>) -> ApiError {
    ApiError(code, msg.into())
}

type ApiResult<T> = Result<T, ApiError>;

pub fn router(state: Shared) -> Router {
    Router::new()
        .route("/", get(|| async { Html(INDEX_HTML) }))
        .route(
            "/favicon.svg",
            get(|| async { ([(header::CONTENT_TYPE, "image/svg+xml")], FAVICON_SVG) }),
        )
        // The usage page also shows the training jobs, at /jobs.
        .route("/jobs", get(|| async { Html(INDEX_HTML) }))
        .route("/admin", get(|| async { Html(ADMIN_HTML) }))
        .route("/health", get(|| async { Json(json!({"ok": true})) }))
        .route("/v1/classify", post(classify))
        .route("/v1/decisions", post(crate::decisions::decisions))
        .route("/api/summary", get(summary))
        .route("/api/prompts", get(list_prompts))
        .route("/api/prompts/{id}", get(get_prompt))
        .route("/api/jobs", get(list_jobs))
        .route("/admin/api/state", get(admin_state))
        .route("/admin/api/settings", put(put_settings))
        .route("/admin/api/cache", get(cache_stats).delete(clear_cache))
        .route(
            "/admin/api/system-one-models/{name}",
            put(put_system_one).delete(delete_system_one),
        )
        .route(
            "/admin/api/judges/{name}",
            put(put_judge).delete(delete_judge),
        )
        .route("/admin/api/prompts/{id}", put(put_prompt_overrides))
        .route("/admin/api/prompts/{id}/train", post(train_now))
        .route("/admin/api/prompts/{id}/audit", post(restart_audit))
        .route("/admin/api/labels", post(put_labels))
        .route("/admin/api/prompts/{id}/undeploy", post(undeploy))
        .route("/admin/api/jobs/{id}/cancel", post(cancel_job))
        .route("/admin/api/jobs/{id}/retry", post(retry_job))
        .route("/internal/jobs/claim", post(claim_job))
        .route("/internal/jobs/{id}/progress", post(job_progress))
        .route("/internal/jobs/{id}/complete", post(complete_job))
        .layer(axum::middleware::map_response(security_headers))
        .with_state(state)
}

/// The pages are self-contained (inline script and style, same-origin API calls), no other site
/// may frame them, and no response may be sniffed as another content type.
const CSP: &str = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; \
                   connect-src 'self'; img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

async fn security_headers(mut res: Response) -> Response {
    for (name, value) in [
        (header::CONTENT_SECURITY_POLICY, CSP),
        (header::X_FRAME_OPTIONS, "DENY"),
        (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        (header::REFERRER_POLICY, "no-referrer"),
    ] {
        res.headers_mut()
            .insert(name, HeaderValue::from_static(value));
    }
    res
}

fn bearer(h: &HeaderMap) -> Option<&str> {
    h.get(header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
}

/// Compares every byte, so the response time doesn't tell a caller how much of a guess was right.
fn same_token(given: &str, token: &str) -> bool {
    given.len() == token.len()
        && given
            .bytes()
            .zip(token.bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
}

fn require(h: &HeaderMap, token: &str) -> ApiResult<()> {
    match bearer(h) {
        Some(t) if !token.is_empty() && same_token(t, token) => Ok(()),
        _ => Err(err(
            StatusCode::UNAUTHORIZED,
            "missing or wrong bearer token",
        )),
    }
}

fn admin(s: &Shared, h: &HeaderMap) -> ApiResult<()> {
    require(h, &s.cfg.admin_token)
}

fn viewer(s: &Shared, h: &HeaderMap) -> ApiResult<()> {
    match &s.cfg.viewer_token {
        None => Ok(()),
        Some(v) => require(h, v).or_else(|_| admin(s, h)),
    }
}

// ---- classify -------------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ClassifyReq {
    pub prompt: Option<PromptSpec>,
    pub prompt_id: Option<String>,
    pub input: String,
    /// Force a System One model by name (default: the prompt's or the global default).
    pub model: Option<String>,
}

async fn classify(State(s): State<Shared>, Json(req): Json<ClassifyReq>) -> ApiResult<Json<Value>> {
    classify_inner(s, req, false).await
}

#[expect(
    clippy::too_many_lines,
    reason = "a request's whole path, in order: the student, System One, then the fallbacks"
)]
pub(crate) async fn classify_inner(
    s: Shared,
    req: ClassifyReq,
    decisions: bool,
) -> ApiResult<Json<Value>> {
    let t0 = Instant::now();
    let p: Arc<PromptState> = match (&req.prompt, &req.prompt_id) {
        (Some(spec), _) => {
            spec.validate()
                .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
            s.resolve_prompt(spec.clone()).await?
        }
        (None, Some(id)) => s
            .prompt(id)
            .ok_or_else(|| err(StatusCode::NOT_FOUND, format!("unknown prompt_id '{id}'")))?,
        (None, None) => {
            return Err(err(
                StatusCode::BAD_REQUEST,
                "send 'prompt' {instructions, labels} or a known 'prompt_id'",
            ))
        }
    };
    if req.input.trim().is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "'input' is empty"));
    }
    let settings = s.settings();
    if settings.log_inputs {
        s.inputs.push(&p.id, &req.input);
    }

    let name = req
        .model
        .clone()
        .or(p.overrides().system_one_model)
        .or(settings.default_system_one.clone());
    // A request that forces another System One model asks for that model, not the student.
    let forced_elsewhere = req.model.as_ref().is_some_and(|requested| {
        Some(requested)
            != p.overrides()
                .system_one_model
                .as_ref()
                .or(settings.default_system_one.as_ref())
    });
    let mut path = vec![];
    // 0. With the cache before the student, what System One answered for this text comes first.
    if settings.cache.before_student() {
        if let Some(out) = cached_answer(
            &s,
            &p,
            name.as_deref(),
            &req.input,
            decisions,
            &mut path,
            t0,
        ) {
            return Ok(out);
        }
    }
    // 1. The student answers when it is live and confident enough. In shadow it still predicts,
    //    for its audit, while System One answers.
    //    With students per language, the language router picks the student.
    let mut student_answer: Option<(usize, f64, Vec<f64>, String)> = None;
    if let Some(dep) = p.deployed().filter(|_| !forced_elsewhere) {
        let text = req.input.clone();
        let d2 = dep.clone();
        let predicted = tokio::task::spawn_blocking(move || {
            let (student, threshold, language) = d2.route(&text);
            let name = match language {
                Some(code) => format!("{}@{code}", student.name()),
                None => student.name(),
            };
            let language = language.map(String::from);
            student
                .predict_proba(&text)
                .map(|probs| (probs, threshold, language, name))
        })
        .await?;
        match predicted {
            Ok((probs, threshold, language, name)) => {
                let (i, c) = argmax(&probs);
                let prediction = crate::monitor::Prediction {
                    label: i,
                    confidence: c,
                    threshold,
                };
                let live = crate::monitor::audit(&s, &p, &dep, &req.input, &prediction);
                if live {
                    path.push(format!("student:{name}"));
                }
                if live && c >= threshold {
                    let mut out = respond(&p, i, Some(c), "student", &name, Some(&probs), t0);
                    if let Some(code) = &language {
                        out.0["language"] = json!(code);
                    }
                    if decisions {
                        out.0["probs"] = exact_probabilities(&p.spec, &probs);
                        out.0["label_mass"] = json!(1.0);
                        out.0["usage"] =
                            json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0});
                    }
                    s.record(&p.id, "student", &path, ms(t0), 0.0);
                    s.count_request(&p);
                    return Ok(out);
                }
                if live {
                    student_answer = Some((i, c, probs, name));
                }
            }
            Err(e) => tracing::error!("prompt {}: student failed: {e:#}", p.id),
        }
    }

    // 2. With the cache after the student, an alternative to it: what System One answered for
    //    this text before.
    if settings.cache.after_student() {
        if let Some(out) = cached_answer(
            &s,
            &p,
            name.as_deref(),
            &req.input,
            decisions,
            &mut path,
            t0,
        ) {
            return Ok(out);
        }
    }
    // 3. Otherwise today's System One model (or LLM) answers, or the stronger model it escalates
    //    to when it is unsure.
    let model = match &name {
        Some(n) => s
            .system_one_models()
            .into_iter()
            .find(|m| &m.name == n && m.enabled),
        None => None,
    };
    // With the cache between System One and the model it escalates to, escalations read it.
    let escalation_cache = settings.cache.before_escalation().then_some(p.id.as_str());
    let result = match &model {
        Some(m) => ask_system_one(&s, m, &p.spec, &req.input, &mut path, escalation_cache)
            .await
            .map_err(|e| format!("{e:#}")),
        None => Err(match name {
            Some(n) => format!("System One model '{n}' is not configured"),
            None => "no System One model configured (set a default in /admin)".into(),
        }),
    };
    match (result, student_answer) {
        (
            Ok(SystemOneAnswer {
                answer: a,
                answered_by,
                usd,
                cached,
            }),
            _,
        ) => {
            let m = model.expect("model exists when the call succeeded");
            // Before or after the student, the cache keeps System One's answer; before an
            // escalation, ask_system_one keeps the stronger model's.
            if !settings.cache.before_escalation() {
                s.answers
                    .put(&p.id, &m.name, &req.input, cached_from(&a, &answered_by));
            }
            let source = if cached { "cache" } else { "system_one" };
            let mut out = respond(&p, a.label, a.confidence, source, &answered_by, None, t0);
            if decisions {
                let probabilities = a.probabilities.as_deref().ok_or_else(|| {
                    err(
                        StatusCode::BAD_GATEWAY,
                        "answering model returned no complete probability distribution",
                    )
                })?;
                out.0["probs"] = exact_probabilities(&p.spec, probabilities);
                if let Some(mass) = a.label_mass {
                    out.0["label_mass"] = json!(mass);
                }
                if let Some(usage) = a.usage {
                    out.0["usage"] = usage;
                }
            }
            if answered_by != m.name {
                out.0["escalated_from"] = json!(m.name);
            }
            s.record(&p.id, source, &path, ms(t0), usd);
            s.count_request(&p);
            Ok(out)
        }
        // Degrade gracefully: a less confident student beats no answer.
        (Err(e), Some((i, c, probs, name))) => {
            tracing::warn!(
                "prompt {}: System One unavailable ({e}); answering with the student",
                p.id
            );
            let mut out = respond(&p, i, Some(c), "student", &name, Some(&probs), t0);
            out.0["degraded"] = json!(true);
            if decisions {
                out.0["probs"] = exact_probabilities(&p.spec, &probs);
                out.0["label_mass"] = json!(1.0);
                out.0["usage"] =
                    json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0});
            }
            s.record(&p.id, "student", &path, ms(t0), 0.0);
            s.count_request(&p);
            Ok(out)
        }
        (Err(e), None) => {
            s.record(&p.id, "error", &path, ms(t0), 0.0);
            match &model {
                // Endpoint URLs and upstream error bodies go to the log, not to the caller.
                Some(m) => {
                    tracing::warn!("prompt {}: System One model '{}' failed: {e}", p.id, m.name);
                    Err(err(
                        StatusCode::BAD_GATEWAY,
                        format!("System One model '{}' failed; see the gateway log", m.name),
                    ))
                }
                None => Err(err(StatusCode::SERVICE_UNAVAILABLE, e)),
            }
        }
    }
}

fn exact_probabilities(spec: &PromptSpec, probabilities: &[f64]) -> Value {
    json!(spec
        .labels
        .iter()
        .zip(probabilities)
        .map(|(l, p)| (l.name.clone(), *p))
        .collect::<HashMap<_, _>>())
}

/// Ask `first`; while the answering model is unsure and names a stronger model to escalate to,
/// ask that one. Returns the answer and the name of the model that gave it. A failed escalation
/// keeps the weaker model's answer.
/// System One's answer, or the answer of the model it escalated to: the model's name, what its
/// calls cost, and whether the last answer came from the cache instead of a call.
struct SystemOneAnswer {
    answer: upstream::Answer,
    answered_by: String,
    usd: f64,
    cached: bool,
}

/// An answer as the cache keeps it.
fn cached_from(a: &upstream::Answer, model: &str) -> CachedAnswer {
    CachedAnswer {
        label: a.label,
        confidence: a.confidence,
        probabilities: a.probabilities.clone(),
        label_mass: a.label_mass,
        model: model.to_string(),
    }
}

/// Ask `first`, and the stronger models it escalates to while each is unsure. With
/// `escalation_cache` (the prompt's id), an escalation takes the stronger model's earlier answer
/// to the text from the cache, and keeps its new ones there.
async fn ask_system_one(
    s: &Shared,
    first: &SystemOneModel,
    spec: &PromptSpec,
    text: &str,
    path: &mut Vec<String>,
    escalation_cache: Option<&str>,
) -> anyhow::Result<SystemOneAnswer> {
    path.push(format!("system_one:{}", first.name));
    let mut answer = upstream::classify(&s.http, first, spec, text).await?;
    // Every answered call costs, the ones an escalation replaced too.
    let mut usd = upstream::call_cost(first, spec, text, &answer);
    let mut cached = false;
    let mut current = first.clone();
    let mut asked = vec![first.name.clone()];
    while let (Some(next), Some(below), Some(c)) = (
        current.escalate_to.clone(),
        current.escalate_below,
        answer.confidence,
    ) {
        if c >= below || asked.contains(&next) {
            break;
        }
        let Some(stronger) = s
            .system_one_models()
            .into_iter()
            .find(|m| m.name == next && m.enabled)
        else {
            tracing::warn!(
                "'{}' escalates to '{next}', which is missing or disabled",
                current.name
            );
            break;
        };
        asked.push(next.clone());
        if let Some(hit) = escalation_cache.and_then(|pid| s.answers.get(pid, &next, text)) {
            path.push("cache".into());
            answer = upstream::Answer {
                label: hit.label,
                confidence: hit.confidence,
                probabilities: hit.probabilities,
                label_mass: hit.label_mass,
                usage: None,
                model: None,
            };
            cached = true;
            current = stronger;
            continue;
        }
        path.push(format!("system_one:{next}"));
        match upstream::classify(&s.http, &stronger, spec, text).await {
            Ok(a) => {
                usd += upstream::call_cost(&stronger, spec, text, &a);
                if let Some(pid) = escalation_cache {
                    s.answers.put(pid, &next, text, cached_from(&a, &next));
                }
                answer = a;
                cached = false;
                current = stronger;
            }
            Err(e) => {
                tracing::warn!(
                    "escalating from '{}' to '{next}' failed ({e:#}); keeping '{}''s answer",
                    current.name,
                    current.name
                );
                break;
            }
        }
    }
    Ok(SystemOneAnswer {
        answer,
        answered_by: current.name,
        usd,
        cached,
    })
}

/// The cache's answer to the request, which `name` gave before, as the response says it: `None`
/// without one, or, for a decisions request, without the full distribution it needs.
fn cached_answer(
    s: &Shared,
    p: &PromptState,
    name: Option<&str>,
    input: &str,
    decisions: bool,
    path: &mut Vec<String>,
    t0: Instant,
) -> Option<Json<Value>> {
    let hit = s.answers.get(&p.id, name?, input)?;
    if decisions && hit.probabilities.is_none() {
        return None;
    }
    path.push("cache".into());
    let mut out = respond(p, hit.label, hit.confidence, "cache", &hit.model, None, t0);
    if decisions {
        out.0["probs"] = exact_probabilities(&p.spec, hit.probabilities.as_deref()?);
        if let Some(mass) = hit.label_mass {
            out.0["label_mass"] = json!(mass);
        }
        out.0["usage"] = json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0});
    }
    s.record(&p.id, "cache", path, ms(t0), 0.0);
    s.count_request(p);
    Some(out)
}

fn ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

pub(crate) fn argmax(p: &[f64]) -> (usize, f64) {
    p.iter()
        .copied()
        .enumerate()
        .fold((0, f64::MIN), |a, (i, x)| if x > a.1 { (i, x) } else { a })
}

fn respond(
    p: &PromptState,
    i: usize,
    conf: Option<f64>,
    source: &str,
    model: &str,
    probs: Option<&[f64]>,
    t0: Instant,
) -> Json<Value> {
    let mut v = json!({
        "prompt_id": p.id,
        "label": p.spec.labels[i].name,
        "confidence": conf.map(|c| (c * 10000.0).round() / 10000.0),
        "source": source,
        "model": model,
        "latency_ms": (ms(t0) * 1000.0).round() / 1000.0,
    });
    if let Some(pr) = probs {
        v["probs"] = json!(p
            .spec
            .labels
            .iter()
            .zip(pr)
            .map(|(l, x)| (l.name.clone(), (x * 10000.0).round() / 10000.0))
            .collect::<HashMap<_, _>>());
    }
    Json(v)
}

// ---- usage ----------------------------------------------------------------------------------

#[derive(Deserialize)]
struct ListQuery {
    limit: Option<usize>,
    days: Option<i64>,
}

struct ModelNames {
    system_one: Option<String>,
    system_one_model_id: Option<String>,
    system_one_pricing: Option<db::Pricing>,
    base_model: String,
    resolved_base_model: String,
}

fn model_names(s: &Shared, o: &PromptOverrides, settings: &Settings) -> ModelNames {
    let system_one = o
        .system_one_model
        .clone()
        .or(settings.default_system_one.clone());
    let model = s
        .system_one_models()
        .into_iter()
        .find(|m| Some(&m.name) == system_one.as_ref());
    let system_one_model_id = model.as_ref().map(|m| m.model.clone());
    let system_one_pricing = model.map(|m| m.pricing);
    let base_model = o
        .base_model
        .clone()
        .unwrap_or(settings.default_base_model.clone());
    let resolved_base_model = if base_model == "auto" {
        Settings::default().default_base_model
    } else {
        base_model.clone()
    };
    ModelNames {
        system_one,
        system_one_model_id,
        system_one_pricing,
        base_model,
        resolved_base_model,
    }
}

fn monitor_view(
    s: &Shared,
    p: &PromptState,
    cfg: &crate::monitor::MonitorSettings,
) -> ApiResult<Option<Value>> {
    if let Some(dep) = p.deployed() {
        return Ok(Some(
            dep.monitor
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .summary(cfg),
        ));
    }
    Ok(s.db.monitor(&p.id)?.map(|m| m.summary(cfg)))
}

/// One prompt's requests over some days, and what System One cost for them.
#[derive(Clone, Copy, Default)]
struct Counts {
    student: u64,
    system_one: u64,
    cache: u64,
    errors: u64,
    latency_ms: f64,
    system_one_usd: f64,
}

impl Counts {
    fn add(&mut self, u: &Counts) {
        self.student += u.student;
        self.system_one += u.system_one;
        self.cache += u.cache;
        self.errors += u.errors;
        self.latency_ms += u.latency_ms;
        self.system_one_usd += u.system_one_usd;
    }
}

impl From<&db::DayUsage> for Counts {
    fn from(u: &db::DayUsage) -> Self {
        Counts {
            student: u.student,
            system_one: u.system_one,
            cache: u.cache,
            errors: u.errors,
            latency_ms: u.latency_ms_sum,
            system_one_usd: u.system_one_usd,
        }
    }
}

/// Each prompt's usage by day since `since_day`, counters not flushed yet included.
fn usage_since(s: &Shared, since_day: &str) -> ApiResult<HashMap<(String, String), Counts>> {
    let mut usage: HashMap<(String, String), Counts> = HashMap::new();
    for (pid, u) in s.db.usage_since(since_day)? {
        let day = u.day.clone();
        usage.entry((pid, day)).or_default().add(&(&u).into());
    }
    for ((pid, day), u) in s
        .usage
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
    {
        usage
            .entry((pid.clone(), day.clone()))
            .or_default()
            .add(&u.into());
    }
    Ok(usage)
}

/// Prompt rows with usage, most used first.
///
/// Costs are estimates: what System One's answers cost at each model's pricing, and what the
/// student's answers saved, at the average cost of a System One answer for the prompt in the last
/// 7 days (or the whole period, when System One answered none of the last 7 days' requests).
#[expect(
    clippy::cast_precision_loss,
    reason = "shares and averages of request counts, which stay far below 2^52"
)]
fn prompt_views(s: &Shared, days: i64) -> ApiResult<Vec<Value>> {
    let days = days.clamp(1, 90);
    let today = chrono::Utc::now().date_naive();
    let day_list: Vec<String> = (0..days)
        .rev()
        .map(|d| (today - chrono::Duration::days(d)).to_string())
        .collect();
    let usage = usage_since(s, &day_list[0])?;
    // The latency histograms are read from SQLite, so the last second's samples go there first.
    if let Err(e) = s.flush_latencies() {
        tracing::warn!("flushing request latencies: {e:#}");
    }
    let settings = s.settings();
    let deployments: HashMap<String, Deployment> =
        s.db.deployments()?
            .into_iter()
            .map(|d| (d.prompt_id.clone(), d))
            .collect();
    let prompts: Vec<Arc<PromptState>> = s
        .prompts
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .values()
        .cloned()
        .collect();
    let last_7_days: Vec<&String> = day_list.iter().rev().take(7).collect();
    let mut out = vec![];
    for p in prompts {
        let on = |d: &String| {
            usage
                .get(&(p.id.clone(), d.clone()))
                .copied()
                .unwrap_or_default()
        };
        let daily: Vec<Value> = day_list
            .iter()
            .map(|d| {
                let u = on(d);
                json!({"day": d, "student": u.student, "system_one": u.system_one, "cache": u.cache,
                       "errors": u.errors, "system_one_usd": u.system_one_usd})
            })
            .collect();
        let (mut week, mut period) = (Counts::default(), Counts::default());
        for d in &last_7_days {
            week.add(&on(d));
        }
        for d in &day_list {
            period.add(&on(d));
        }
        let all7 = week.student + week.system_one + week.cache;
        let cost_per_answer = [week, period]
            .into_iter()
            .find(|c| c.system_one > 0)
            .map(|c| c.system_one_usd / c.system_one as f64);
        let o = p.overrides();
        let dep = deployments.get(&p.id);
        let models = model_names(s, &o, &settings);
        out.push(json!({
            "id": p.id,
            "title": p.spec.title(),
            "instructions": p.spec.instructions,
            "labels": p.spec.label_names(),
            "total": p.total.load(Ordering::Relaxed),
            "today": day_list.last().map_or(0, |d| { let u = on(d); u.student + u.system_one + u.cache }),
            "last_7_days": all7,
            "errors_7_days": week.errors,
            "student_share_7_days": if all7 > 0 { week.student as f64 / all7 as f64 } else { 0.0 },
            "cache_7_days": week.cache,
            "cache_share_7_days": if all7 > 0 { week.cache as f64 / all7 as f64 } else { 0.0 },
            "model_latencies_7_days": s.db.answering_model_latencies(&p.id, &last_7_days.last().unwrap()[..])?,
            "path_latencies_7_days": s.db.path_latencies(&p.id, &last_7_days.last().unwrap()[..])?,
            "avg_latency_ms_7_days": if all7 + week.errors > 0 { week.latency_ms / (all7 + week.errors) as f64 } else { 0.0 },
            "system_one_cost_usd_7_days": week.system_one_usd,
            "system_one_cost_per_answer_usd": cost_per_answer,
            "student_savings_usd_7_days": cost_per_answer.map(|c| c * week.student as f64),
            "cache_savings_usd_7_days": cost_per_answer.map(|c| c * week.cache as f64),
            "status": p.status(),
            "training_job": s.db.progress_job_for(&p.id)?.as_ref().map(|j| job_progress_view(s, j)),
            "threshold": s.threshold_for(&p, &settings),
            "system_one_model": models.system_one,
            "system_one_model_id": models.system_one_model_id,
            "system_one_pricing": models.system_one_pricing,
            "judge": o.judge.clone().or(settings.default_judge.clone()),
            "judge_model_id": s.judge_for(&p).map(|judge| judge.model),
            "base_model": models.base_model,
            "resolved_base_model": models.resolved_base_model,
            "target_accuracy": o.target_accuracy.unwrap_or(settings.default_target_accuracy),
            "overrides": o,
            "monitor": monitor_view(s, &p, &settings.monitor)?,
            "deployment": dep.map(|d| json!({"job_id": d.job_id, "threshold": d.threshold, "active": d.active,
                                             "deployed_at": d.deployed_at, "metrics": d.metrics})),
            "daily": daily,
        }));
    }
    out.sort_by(|a, b| b["total"].as_u64().cmp(&a["total"].as_u64()));
    Ok(out)
}

async fn list_prompts(
    State(s): State<Shared>,
    h: HeaderMap,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<Value>> {
    viewer(&s, &h)?;
    let mut v = prompt_views(&s, q.days.unwrap_or(14))?;
    v.truncate(q.limit.unwrap_or(50));
    Ok(Json(json!(v)))
}

async fn get_prompt(
    State(s): State<Shared>,
    h: HeaderMap,
    Path(id): Path<String>,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<Value>> {
    viewer(&s, &h)?;
    prompt_views(&s, q.days.unwrap_or(14))?
        .into_iter()
        .find(|p| p["id"] == id)
        .map(Json)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "unknown prompt"))
}

#[expect(
    clippy::cast_precision_loss,
    reason = "a share of request counts, which stay far below 2^52"
)]
async fn summary(State(s): State<Shared>, h: HeaderMap) -> ApiResult<Json<Value>> {
    viewer(&s, &h)?;
    let v = prompt_views(&s, 7)?;
    let sum = |k: &str| v.iter().map(|p| p[k].as_u64().unwrap_or(0)).sum::<u64>();
    let week = sum("last_7_days");
    let student_week: f64 = v
        .iter()
        .map(|p| {
            p["student_share_7_days"].as_f64().unwrap_or(0.0)
                * p["last_7_days"].as_f64().unwrap_or(0.0)
        })
        .sum();
    let usd = |k: &str| v.iter().filter_map(|p| p[k].as_f64()).sum::<f64>();
    Ok(Json(json!({
        "prompts": v.len(),
        "students": v.iter().filter(|p| p["status"] == "deployed").count(),
        "requests_total": sum("total"),
        "requests_today": sum("today"),
        "requests_7_days": week,
        "student_share_7_days": if week > 0 { student_week / week as f64 } else { 0.0 },
        "system_one_cost_usd_7_days": usd("system_one_cost_usd_7_days"),
        "student_savings_usd_7_days": usd("student_savings_usd_7_days"),
        "cache_7_days": sum("cache_7_days"),
        "cache_share_7_days": if week > 0 { sum("cache_7_days") as f64 / week as f64 } else { 0.0 },
        "cache_savings_usd_7_days": usd("cache_savings_usd_7_days"),
    })))
}

// ---- training progress ----------------------------------------------------------------------

/// The newest jobs for the usage page's training history, to whoever may view usage (anyone, unless there is a
/// viewer token): how far each job has got and which prompt it trains a student for, and none
/// of its spec (judge, store, settings) or its message, which can quote an upstream error body.
#[derive(Deserialize)]
struct JobsQuery {
    prompt_id: Option<String>,
}

async fn list_jobs(
    State(s): State<Shared>,
    h: HeaderMap,
    Query(q): Query<JobsQuery>,
) -> ApiResult<Json<Value>> {
    viewer(&s, &h)?;
    let attempts = match q.prompt_id {
        Some(id) => s.db.jobs_for(&id, 50)?,
        None => s.db.jobs(50)?,
    };
    let jobs: Vec<Value> = attempts.iter().map(|j| job_progress_view(&s, j)).collect();
    Ok(Json(json!(jobs)))
}

/// A job's prompt is named as `/api/prompts` names it to the same viewers: the first line of its
/// instructions (`title`) and its labels' names, without their descriptions. It is null for a
/// prompt the gateway doesn't know, and the page then shows the prompt's id alone.
fn job_progress_view(s: &Shared, j: &db::Job) -> Value {
    let prompt = s.prompt(&j.prompt_id).map(|p| {
        json!({
            "title": p.spec.title(),
            "labels": p.spec.label_names(),
        })
    });
    json!({
        "id": j.id,
        "prompt_id": j.prompt_id,
        "prompt": prompt,
        "status": j.status,
        "phase": job_phase(j),
        "done": j.done,
        "total": j.total,
        "created_at": j.created_at,
        "started_at": j.started_at,
        "finished_at": j.finished_at,
        "last_stage": j.last_stage,
        "stage_done": j.stage_done,
        "stage_total": j.stage_total,
    })
}

/// A running job's stage, from the start of the worker's message ("judging N inputs with …",
/// "training a student on …"): judging, training, calibration, export or validation. A
/// worker has phase `starting` before its first stage. A
/// job an admin cancelled is `cancelled`; any other job's phase is its status.
fn job_phase(j: &db::Job) -> &str {
    let message = j.message.as_deref().unwrap_or_default();
    match j.status.as_str() {
        "running" => db::progress_stage(message).unwrap_or("starting"),
        "failed" if j.cancelled_by_admin() => "cancelled",
        status => status,
    }
}

// ---- admin ----------------------------------------------------------------------------------

async fn admin_state(State(s): State<Shared>, h: HeaderMap) -> ApiResult<Json<Value>> {
    admin(&s, &h)?;
    Ok(Json(json!({
        "settings": s.settings(),
        "system_one_models": s.db.system_one_models()?,
        "judges": s.db.judges()?,
        "prompts": prompt_views(&s, 14)?,
        "jobs": s.db.jobs(50)?.iter().map(admin_job_view).collect::<Vec<_>>(),
        "store": s.store.url,
        "cache": s.answers.stats(),
    })))
}

/// The cache's settings, size, hits, misses and evictions since the gateway started.
async fn cache_stats(State(s): State<Shared>, h: HeaderMap) -> ApiResult<Json<Value>> {
    admin(&s, &h)?;
    Ok(Json(s.answers.stats()))
}

/// Empty the cache: the next request for each text asks System One again.
async fn clear_cache(State(s): State<Shared>, h: HeaderMap) -> ApiResult<Json<Value>> {
    admin(&s, &h)?;
    s.answers.clear();
    Ok(Json(s.answers.stats()))
}

/// A job as the admin page lists it: all of it, and its phase (`job_phase`), which tells a job
/// an admin cancelled, and may retry, from one that failed.
fn admin_job_view(j: &db::Job) -> Value {
    let mut v = json!(j);
    v["phase"] = json!(job_phase(j));
    v
}

fn merge(base: &mut Value, patch: &Value) {
    if let (Some(b), Some(p)) = (base.as_object_mut(), patch.as_object()) {
        for (k, v) in p {
            b.insert(k.clone(), v.clone());
        }
    }
}

async fn put_settings(
    State(s): State<Shared>,
    h: HeaderMap,
    Json(patch): Json<Value>,
) -> ApiResult<Json<Value>> {
    admin(&s, &h)?;
    let mut v = serde_json::to_value(s.settings())?;
    merge(&mut v, &patch);
    let new: Settings =
        serde_json::from_value(v).map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?;
    new.monitor
        .check()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    if let Some(n) = &new.default_system_one {
        if !s.db.system_one_models()?.iter().any(|m| &m.name == n) {
            return Err(err(
                StatusCode::BAD_REQUEST,
                format!("no System One model named '{n}'"),
            ));
        }
    }
    if let Some(n) = &new.default_judge {
        let j =
            s.db.judges()?
                .into_iter()
                .find(|j| &j.name == n)
                .ok_or_else(|| err(StatusCode::BAD_REQUEST, format!("no judge named '{n}'")))?;
        j.check().map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    }
    new.calibration
        .check()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    new.cache
        .check()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    new.per_language
        .check()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    if new.default_threshold == 0
        || !(0.5..=1.0).contains(&new.default_target_accuracy)
        || !new.train.is_object()
    {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "threshold must be > 0, target accuracy in [0.5, 1], train an object",
        ));
    }
    if new.max_attempts == 0
        || new.retry_secs > new.retry_max_secs
        || new.retry_max_secs > db::MAX_RETRY_SECS
    {
        return Err(err(
            StatusCode::BAD_REQUEST,
            format!(
                "max_attempts must be > 0, and retry_secs <= retry_max_secs <= {} (a week)",
                db::MAX_RETRY_SECS
            ),
        ));
    }
    s.db.put_settings(&new)?;
    s.reload_config()?;
    let queued = requeue_ready(&s)?;
    Ok(Json(json!({"settings": new, "queued": queued})))
}

/// After a threshold change, prompts that already qualify are queued right away.
fn requeue_ready(s: &Shared) -> ApiResult<usize> {
    let settings = s.settings();
    let prompts: Vec<Arc<PromptState>> = s
        .prompts
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .values()
        .cloned()
        .collect();
    let mut n = 0;
    for p in prompts {
        if p.status() == "collecting"
            && p.total.load(Ordering::Relaxed) >= s.threshold_for(&p, &settings)
            && s.queue_job(&p, "threshold")?.is_some()
        {
            n += 1;
        }
    }
    Ok(n)
}

async fn put_system_one(
    State(s): State<Shared>,
    h: HeaderMap,
    Path(name): Path<String>,
    Json(mut m): Json<SystemOneModel>,
) -> ApiResult<Json<Value>> {
    admin(&s, &h)?;
    m.name = name;
    m.check().map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    url::Url::parse(&m.base_url)
        .map_err(|e| err(StatusCode::BAD_REQUEST, format!("base_url: {e}")))?;
    m.escalate_to = m.escalate_to.filter(|n| !n.is_empty());
    match (&m.escalate_to, m.escalate_below) {
        (None, None) => {}
        (Some(n), Some(below)) if below > 0.0 && below <= 1.0 => {
            if n == &m.name {
                return Err(err(
                    StatusCode::BAD_REQUEST,
                    "a model can't escalate to itself",
                ));
            }
            if !s.db.system_one_models()?.iter().any(|o| &o.name == n) {
                return Err(err(
                    StatusCode::BAD_REQUEST,
                    format!("no System One model named '{n}' to escalate to"),
                ));
            }
        }
        _ => {
            return Err(err(
                StatusCode::BAD_REQUEST,
                "escalation needs both escalate_to and escalate_below (a confidence in (0, 1])",
            ))
        }
    }
    s.db.put_system_one_model(&m)?;
    s.reload_config()?;
    // Its endpoint or model may have changed under the same name: its answers may be stale.
    s.answers.clear();
    Ok(Json(json!(m)))
}

async fn delete_system_one(
    State(s): State<Shared>,
    h: HeaderMap,
    Path(name): Path<String>,
) -> ApiResult<StatusCode> {
    admin(&s, &h)?;
    let deleted = s.db.delete_system_one_model(&name)?;
    s.reload_config()?;
    s.answers.clear();
    Ok(if deleted {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    })
}

async fn put_judge(
    State(s): State<Shared>,
    h: HeaderMap,
    Path(name): Path<String>,
    Json(mut j): Json<JudgeModel>,
) -> ApiResult<Json<Value>> {
    admin(&s, &h)?;
    j.name = name;
    j.check().map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    url::Url::parse(&j.base_url)
        .map_err(|e| err(StatusCode::BAD_REQUEST, format!("base_url: {e}")))?;
    s.db.put_judge(&j)?;
    s.reload_config()?;
    Ok(Json(json!(j)))
}

async fn delete_judge(
    State(s): State<Shared>,
    h: HeaderMap,
    Path(name): Path<String>,
) -> ApiResult<StatusCode> {
    admin(&s, &h)?;
    let deleted = s.db.delete_judge(&name)?;
    s.reload_config()?;
    Ok(if deleted {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    })
}

async fn put_prompt_overrides(
    State(s): State<Shared>,
    h: HeaderMap,
    Path(id): Path<String>,
    Json(o): Json<PromptOverrides>,
) -> ApiResult<Json<Value>> {
    admin(&s, &h)?;
    let p = s
        .prompt(&id)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "unknown prompt"))?;
    if let Some(n) = &o.system_one_model {
        if !s.db.system_one_models()?.iter().any(|m| &m.name == n) {
            return Err(err(
                StatusCode::BAD_REQUEST,
                format!("no System One model named '{n}'"),
            ));
        }
    }
    if let Some(n) = &o.judge {
        let j =
            s.db.judges()?
                .into_iter()
                .find(|j| &j.name == n)
                .ok_or_else(|| err(StatusCode::BAD_REQUEST, format!("no judge named '{n}'")))?;
        j.check().map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    }
    if o.threshold == Some(0)
        || o.target_accuracy.is_some_and(|t| !(0.5..=1.0).contains(&t))
        || o.base_model.as_deref() == Some("")
    {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "threshold must be > 0, target accuracy in [0.5, 1], base model non-empty",
        ));
    }
    // Under the write lock, so two saves at once leave in memory what SQLite keeps.
    let mut overrides = p.overrides.write().unwrap_or_else(PoisonError::into_inner);
    s.db.set_overrides(&id, &o)?;
    *overrides = o.clone();
    drop(overrides);
    let queued = requeue_ready(&s)?;
    Ok(Json(json!({"overrides": o, "queued": queued})))
}

/// Queue a job, or run a retry that waits for its time now.
async fn train_now(
    State(s): State<Shared>,
    h: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    admin(&s, &h)?;
    let p = s
        .prompt(&id)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "unknown prompt"))?;
    match s.queue_job(&p, "manual")? {
        Some(j) => Ok(Json(json!(j))),
        None => Err(err(
            StatusCode::CONFLICT,
            "a training job for this prompt is already queued or running",
        )),
    }
}

#[derive(Deserialize)]
struct LabelsReq {
    prompt: PromptSpec,
    /// The file's name under `human_labels/{prompt_id}/`; uploading the same name replaces it.
    name: String,
    labels: Vec<LabelRow>,
}

#[derive(Deserialize, serde::Serialize)]
struct LabelRow {
    text: String,
    label: String,
    /// `train` (trains the student), `audit` (held out of training, the answer the student is
    /// audited against) or none (the worker's `human_holdout` share decides).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    split: Option<String>,
}

/// Reference labels: answers a person (or a labeled dataset) gave, for the worker to train on
/// and to audit the student against. Registers the prompt if it is new.
async fn put_labels(
    State(s): State<Shared>,
    h: HeaderMap,
    Json(req): Json<LabelsReq>,
) -> ApiResult<Json<Value>> {
    admin(&s, &h)?;
    req.prompt
        .validate()
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
    let name = req.name.trim();
    if name.is_empty()
        || name.len() > 64
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(err(
            StatusCode::BAD_REQUEST,
            "name: 1 to 64 letters, digits, '-' or '_'",
        ));
    }
    let names = req.prompt.label_names();
    let mut body = String::new();
    let (mut train, mut audit, mut unsplit) = (0u64, 0u64, 0u64);
    let mut unknown = std::collections::BTreeMap::<String, u64>::new();
    for row in &req.labels {
        if row.text.trim().is_empty() {
            continue;
        }
        if !names.contains(&row.label) {
            *unknown.entry(row.label.clone()).or_default() += 1;
            continue;
        }
        match row.split.as_deref() {
            Some("train") => train += 1,
            Some("audit") => audit += 1,
            None => unsplit += 1,
            Some(other) => {
                return Err(err(
                    StatusCode::BAD_REQUEST,
                    format!("split '{other}': use train, audit or leave it out"),
                ))
            }
        }
        body.push_str(&serde_json::to_string(row).map_err(anyhow::Error::from)?);
        body.push('\n');
    }
    if !unknown.is_empty() {
        return Err(err(
            StatusCode::BAD_REQUEST,
            format!("labels the prompt doesn't have: {unknown:?}"),
        ));
    }
    let p = s.resolve_prompt(req.prompt).await?;
    s.store
        .put(
            &format!("human_labels/{}/{name}.jsonl", p.id),
            body.into_bytes().into(),
        )
        .await?;
    tracing::info!(
        "prompt {}: reference labels '{name}': {train} train, {audit} audit, {unsplit} unsplit",
        p.id
    );
    Ok(Json(
        json!({"prompt_id": p.id, "name": name, "train": train, "audit": audit, "unsplit": unsplit}),
    ))
}

/// Run the retained student's shadow audit again, from scratch, under the current settings.
async fn restart_audit(
    State(s): State<Shared>,
    h: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    admin(&s, &h)?;
    let p = s
        .prompt(&id)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "unknown prompt"))?;
    let deployment =
        s.db.deployments()?
            .into_iter()
            .find(|d| d.prompt_id == id)
            .ok_or_else(|| err(StatusCode::CONFLICT, "no retained student to audit"))?;
    let (student, references, languages) = s
        .load_student(
            &id,
            &deployment.job_id,
            &deployment.model_prefix,
            p.spec.labels.len(),
        )
        .await?;
    let _guard = job_lock(&s);
    let current = s.db.deployments()?.into_iter().find(|d| d.prompt_id == id);
    if current
        .as_ref()
        .is_none_or(|d| d.job_id != deployment.job_id || d.deployed_at != deployment.deployed_at)
    {
        return Err(err(
            StatusCode::CONFLICT,
            "student changed while loading; try again",
        ));
    }
    let settings = s.settings();
    if !settings.monitor.enabled {
        return Err(err(
            StatusCode::CONFLICT,
            "enable monitoring before restarting audits",
        ));
    }
    let judge = s.judge_for(&p);
    if judge.is_none() && references.audit_count() == 0 {
        return Err(err(
            StatusCode::CONFLICT,
            "nothing to audit against: configure a judge, or retrain with reference labels",
        ));
    }
    let job = s.db.open_job_for(&id)?;
    if job.as_ref().is_some_and(|j| j.status == "running") {
        return Err(err(
            StatusCode::CONFLICT,
            "cancel the running training job before restarting audits",
        ));
    }
    let baseline = p
        .overrides()
        .target_accuracy
        .unwrap_or(settings.default_target_accuracy);
    let monitor = crate::monitor::MonitorState::new(&deployment.job_id, baseline, judge, true);
    if let Some(job) = job {
        s.db.finish_job(&job.id, false, Some(db::CANCELLED_BY_ADMIN))?;
    }
    s.db.put_monitor(&id, &monitor)?;
    s.db.set_deployment_active(&id, true)?;
    s.db.set_status(&id, "shadow")?;
    *p.deployed.write().unwrap_or_else(PoisonError::into_inner) = Some(Arc::new(Deployed {
        student,
        references,
        threshold: deployment.threshold,
        job_id: deployment.job_id,
        monitor: std::sync::Mutex::new(monitor),
        languages,
    }));
    p.set_status("shadow");
    Ok(Json(
        json!({"prompt_id":id,"status":"shadow","required_agreement":baseline-settings.monitor.tolerance}),
    ))
}

async fn undeploy(
    State(s): State<Shared>,
    h: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    admin(&s, &h)?;
    let p = s
        .prompt(&id)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "unknown prompt"))?;
    let _g = job_lock(&s);
    *p.deployed.write().unwrap_or_else(PoisonError::into_inner) = None;
    s.db.set_deployment_active(&id, false)?;
    s.db.set_status(&id, "collecting")?;
    p.set_status("collecting");
    Ok(Json(json!({"prompt_id": id, "status": "collecting"})))
}

async fn cancel_job(
    State(s): State<Shared>,
    h: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    admin(&s, &h)?;
    let job =
        s.db.job(&id)?
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "unknown job"))?;
    if job.status != "queued" && job.status != "running" {
        return Err(err(
            StatusCode::CONFLICT,
            format!("job is already {}", job.status),
        ));
    }
    // Its completion may finish it first; then the cancel is refused and changes nothing. A
    // cancelled job isn't retried unless an admin asks (`retry_job`), and cancelling a retry that
    // waits ends the retries.
    let _g = job_lock(&s);
    if !s.db.finish_job(&id, false, Some(db::CANCELLED_BY_ADMIN))? {
        return Err(err(StatusCode::CONFLICT, "job has finished already"));
    }
    fail_prompt(&s, &job.prompt_id)?;
    Ok(Json(json!({"job_id": id, "status": "failed"})))
}

/// Queue a job an admin cancelled again: a new job for its prompt, which starts over as "Train
/// now" would, at attempt 1 and claimable at once. The cancelled job stays as it is. A job that
/// failed on its own is left to the automatic retries and "Train now".
async fn retry_job(
    State(s): State<Shared>,
    h: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    admin(&s, &h)?;
    let _g = job_lock(&s);
    let job =
        s.db.job(&id)?
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "unknown job"))?;
    if !job.cancelled_by_admin() {
        let this = match job.status.as_str() {
            "failed" => "failed on its own".to_string(),
            status => format!("is {status}"),
        };
        return Err(err(
            StatusCode::CONFLICT,
            format!("only a job an admin cancelled can be retried; this one {this}"),
        ));
    }
    let p = s
        .prompt(&job.prompt_id)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "the job's prompt no longer exists"))?;
    if s.db.open_job_for(&p.id)?.is_some() {
        return Err(err(
            StatusCode::CONFLICT,
            "a training job for this prompt is already queued or running",
        ));
    }
    // What the worker gets (judge, base model, settings) is built when it claims the job, from
    // the prompt's settings then: this spec only says why the job was queued.
    let spec = json!({"reason": "manual_retry", "retry_of": job.id, "requests": p.total.load(Ordering::Relaxed)});
    let retry = s.db.create_job(&p.id, &spec)?;
    s.db.set_status(&p.id, "queued")?;
    p.set_status("queued");
    tracing::info!(
        "prompt {}: queued training job {} (retry of {}, which an admin cancelled)",
        p.id,
        retry.id,
        job.id
    );
    Ok(Json(json!(retry)))
}

/// Queueing and every change to a job's status, a prompt's status or its student happen under
/// `queue_lock`, so memory says what SQLite says (verification/JobLifecycle.tla).
fn job_lock(s: &Shared) -> std::sync::MutexGuard<'_, ()> {
    s.queue_lock.lock().unwrap_or_else(PoisonError::into_inner)
}

fn fail_prompt(s: &Shared, prompt_id: &str) -> ApiResult<()> {
    if let Some(p) = s.prompt(prompt_id) {
        let status = p.deployed().map_or("failed", |d| {
            if d.monitor
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .phase
                == "shadow"
            {
                "shadow"
            } else {
                "deployed"
            }
        });
        s.db.set_status(prompt_id, status)?;
        p.set_status(status);
    }
    Ok(())
}

/// After the worker reports a job failed: queue the prompt's next attempt, which no worker
/// claims before its time (`Settings::retry_delay`), or after the last attempt, fail the prompt.
/// Under `queue_lock`, like `fail_prompt`. An admin's cancel, a student that can't be loaded and
/// a job that can't start are not retried: they call `fail_prompt` directly.
fn retry_or_fail(s: &Shared, failed: &db::Job) -> ApiResult<Option<db::Job>> {
    let settings = s.settings();
    let (Some(p), Some(secs)) = (
        s.prompt(&failed.prompt_id),
        settings.retry_delay(failed.attempt),
    ) else {
        fail_prompt(s, &failed.prompt_id)?;
        return Ok(None);
    };
    let not_before = db::secs_from_now(secs);
    let spec = json!({"reason": "retry", "retry_of": failed.id, "requests": p.total.load(Ordering::Relaxed)});
    let retry = s.db.create_retry(failed, &spec, &not_before)?;
    s.db.set_status(&p.id, "queued")?;
    p.set_status("queued");
    tracing::info!(
        "prompt {}: job {} failed; attempt {} of {} ({}) waits until {not_before}",
        p.id,
        failed.id,
        retry.attempt,
        settings.max_attempts,
        retry.id
    );
    Ok(Some(retry))
}

// ---- worker ---------------------------------------------------------------------------------

async fn claim_job(State(s): State<Shared>, h: HeaderMap) -> ApiResult<Response> {
    require(&h, &s.cfg.worker_token)?;
    s.flush_all().await?; // the worker must see every logged input
    let _g = job_lock(&s);
    while let Some(job) = s.db.claim_job()? {
        if let Some(p) = s.prompt(&job.prompt_id) {
            p.set_status("training");
        }
        match s.job_spec(&job) {
            Ok(spec) => return Ok(Json(spec).into_response()),
            Err(e) => {
                tracing::error!("job {} can't start: {e}", job.id);
                s.db.finish_job(&job.id, false, Some(&e))?;
                fail_prompt(&s, &job.prompt_id)?;
            }
        }
    }
    Ok(StatusCode::NO_CONTENT.into_response())
}

/// `{"message"}`, with `"done"` and `"total"` while the worker counts its stage (inputs judged,
/// training steps). A message without them clears the last ones.
#[derive(Deserialize)]
struct Progress {
    message: String,
    done: Option<u64>,
    total: Option<u64>,
}

async fn job_progress(
    State(s): State<Shared>,
    h: HeaderMap,
    Path(id): Path<String>,
    Json(p): Json<Progress>,
) -> ApiResult<StatusCode> {
    require(&h, &s.cfg.worker_token)?;
    let counts = match (p.done, p.total) {
        (None, None) => None,
        (Some(done), Some(total)) if done <= total => Some((done, total)),
        _ => {
            return Err(err(
                StatusCode::BAD_REQUEST,
                "done and total go together, with done at most total",
            ))
        }
    };
    if !s.db.job_message(&id, &p.message, counts)? {
        return Err(err(StatusCode::CONFLICT, "job is not running"));
    }
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct Completion {
    ok: bool,
    message: Option<String>,
    threshold: Option<f64>,
    #[serde(default)]
    metrics: Value,
}

async fn complete_job(
    State(s): State<Shared>,
    h: HeaderMap,
    Path(id): Path<String>,
    Json(c): Json<Completion>,
) -> ApiResult<Json<Value>> {
    require(&h, &s.cfg.worker_token)?;
    let job =
        s.db.job(&id)?
            .ok_or_else(|| err(StatusCode::NOT_FOUND, "unknown job"))?;
    if job.status != "running" {
        return Err(err(StatusCode::CONFLICT, format!("job is {}", job.status)));
    }
    if !c.ok {
        let _g = job_lock(&s);
        let retry = if s.db.finish_job(&id, false, c.message.as_deref())? {
            retry_or_fail(&s, &job)?
        } else {
            None
        };
        let retry = retry
            .map(|r| json!({"job_id": r.id, "attempt": r.attempt, "not_before": r.not_before}));
        return Ok(Json(json!({"deployed": false, "retry": retry})));
    }
    let prefix = format!("models/{}/{}", job.prompt_id, id);
    let threshold = c.threshold.unwrap_or(1.01);
    let p = s
        .prompt(&job.prompt_id)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "prompt vanished"))?;
    let (student, references, languages) = match s
        .load_student(&job.prompt_id, &id, &prefix, p.spec.labels.len())
        .await
    {
        Ok(loaded) => loaded,
        Err(e) => {
            let msg = format!("the published student could not be loaded: {e:#}");
            let _g = job_lock(&s);
            if s.db.finish_job(&id, false, Some(&msg))? {
                fail_prompt(&s, &job.prompt_id)?;
            }
            return Err(err(StatusCode::UNPROCESSABLE_ENTITY, msg));
        }
    };
    // An admin may have cancelled the job while its student loaded, or a repeated call may have
    // completed it: the student is deployed only by the call that finishes the job.
    let _g = job_lock(&s);
    if !s.db.finish_job(&id, true, c.message.as_deref())? {
        return Err(err(StatusCode::CONFLICT, "job has finished already"));
    }
    let dep = Deployment {
        prompt_id: p.id.clone(),
        job_id: id.clone(),
        model_prefix: prefix,
        threshold,
        metrics: c.metrics.clone(),
        deployed_at: db::now(),
        active: true,
    };
    let settings = s.settings();
    let baseline = c.metrics["cascade"]["target_accuracy"]
        .as_f64()
        .filter(|a| a.is_finite() && (0.0..=1.0).contains(a))
        .unwrap_or(
            p.overrides()
                .target_accuracy
                .unwrap_or(settings.default_target_accuracy),
        );
    let monitor = crate::monitor::MonitorState::for_deployment(
        &dep,
        baseline,
        s.judge_for(&p),
        settings.monitor.enabled,
    );
    let status = if settings.monitor.enabled {
        "shadow"
    } else {
        "deployed"
    };
    s.db.put_deployment(&dep)?;
    s.db.put_monitor(&p.id, &monitor)?;
    s.db.set_status(&p.id, status)?;
    *p.deployed.write().unwrap_or_else(PoisonError::into_inner) = Some(Arc::new(Deployed {
        student,
        references,
        threshold,
        job_id: id.clone(),
        monitor: std::sync::Mutex::new(monitor),
        languages,
    }));
    p.set_status(status);
    tracing::info!(
        "prompt {}: student from job {id} deployed (threshold {threshold:.3})",
        p.id
    );
    Ok(Json(
        json!({"deployed": !settings.monitor.enabled, "shadow": settings.monitor.enabled, "threshold": threshold}),
    ))
}
