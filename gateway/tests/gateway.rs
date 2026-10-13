//! Drives the gateway over HTTP against a mock System One endpoint: classify, usage, admin,
//! threshold -> job -> worker claim -> completion -> student serving (when a model is available).

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use system_none_gateway::db::Job;
use system_none_gateway::{api, AppState, GatewayConfig};

const ADMIN: &str = "admin-secret";
const WORKER: &str = "worker-secret";

#[tokio::test]
async fn audit_judge_supports_vote_sampling_and_systemone() {
    use system_none_gateway::{db::JudgeModel, upstream};
    let calls = Arc::new(AtomicUsize::new(0));
    let s1 = mock_system_one(calls.clone()).await;
    let spec = serde_json::from_value(prompt()).unwrap();
    let mut judge: JudgeModel = serde_json::from_value(json!({
        "name": "audit-judge", "model": "teacher", "base_url": format!("http://{s1}/v1"),
        "mode": "sample", "samples": 3, "license_ack": true,
    }))
    .unwrap();
    let client = reqwest::Client::new();
    assert_eq!(
        upstream::judge(&client, &judge, &spec, "refund")
            .await
            .unwrap()
            .label,
        0
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "an endpoint ignoring n is sampled again"
    );
    let bodies = Arc::new(Mutex::new(vec![]));
    let decider = mock_decider(bodies.clone()).await;
    judge.mode = "systemone".into();
    judge.base_url = format!("http://{decider}");
    assert_eq!(
        upstream::judge(&client, &judge, &spec, "refund")
            .await
            .unwrap()
            .label,
        0
    );
    assert_eq!(bodies.lock().unwrap().len(), 1);
}

/// A System One stand-in: answers the lettered prompt by keyword, with logprobs.
async fn mock_system_one(calls: Arc<AtomicUsize>) -> SocketAddr {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(body): Json<Value>| {
            let calls = calls.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                let text = body["messages"][1]["content"].as_str().unwrap_or("").to_lowercase();
                let text = text.split("\"\"\"").nth(1).unwrap_or("");
                let letter = if text.contains("refund") { "A" } else if text.contains("cancel") { "B" } else { "C" };
                Json(json!({"choices": [{"message": {"content": letter},
                    "logprobs": {"content": [{"token": letter, "logprob": 0.0,
                        "top_logprobs": [{"token": letter, "logprob": -0.05}, {"token": "C", "logprob": -3.0}]}]}}]}))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

async fn start_gateway(dir: &std::path::Path, viewer: Option<&str>) -> (String, Arc<AppState>) {
    let cfg = GatewayConfig {
        db_path: dir.join("gw.db"),
        store_url: format!("file://{}", dir.join("store").display()),
        cache_dir: dir.join("cache"),
        admin_token: ADMIN.into(),
        worker_token: WORKER.into(),
        viewer_token: viewer.map(String::from),
        onnxruntime: None,
        threads: Some(1),
        flush_secs: 1,
        log_batch: 1000,
    };
    let state = AppState::start(cfg).await.unwrap();
    state.spawn_background();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = api::router(state.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, state)
}

fn prompt() -> Value {
    json!({"instructions": "What does the customer want?", "labels": ["refund", "cancel", "other"]})
}

/// Polls `done` for up to two seconds: background audits finish after the response.
async fn wait_until(mut done: impl FnMut() -> bool) {
    for _ in 0..200 {
        if done() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    clippy::too_many_lines,
    reason = "exercise deployment, promotion, drift, a restarted test and a failed one end to end"
)]
async fn the_judge_audits_students_and_a_failed_shadow_test_stops_without_retraining() {
    // A constant refund predictor makes both agreement and disagreement deterministic.
    // Run this integration scenario when the ONNX runtime library is configured.
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        return;
    }
    let model_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audit-student");
    let dir = tempfile::tempdir().unwrap();
    let (s1_calls, judge_calls) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let s1 = mock_system_one(s1_calls.clone()).await;
    let judge = mock_system_one(judge_calls.clone()).await;
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let r = http.put(format!("{url}/admin/api/system-one-models/reference")).bearer_auth(ADMIN)
        .json(&json!({"kind":"openai_chat", "base_url":format!("http://{s1}/v1"), "model":"reference"})).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let r = http
        .put(format!("{url}/admin/api/judges/trainer"))
        .bearer_auth(ADMIN)
        .json(&json!({"base_url":format!("http://{judge}/v1"), "model":"trainer", "license_ack":true}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let r = http.put(format!("{url}/admin/api/settings")).bearer_auth(ADMIN)
        .json(&json!({"default_system_one":"reference", "default_judge":"trainer", "default_threshold":5,
            "monitor":{"min_samples":2,"window":2,"sample_rate":1.0,"tolerance":0.0}})).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let classify = |input: &str| {
        let (http, url, input) = (http.clone(), url.clone(), input.to_string());
        async move {
            let v: Value = http
                .post(format!("{url}/v1/classify"))
                .json(&json!({"prompt":prompt(),"input":input}))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            v
        }
    };
    let pid = classify("refund first").await["prompt_id"]
        .as_str()
        .unwrap()
        .to_string();
    let p = state.prompt(&pid).unwrap();
    state.queue_job(&p, "manual").unwrap();
    let spec: Value = http
        .post(format!("{url}/internal/jobs/claim"))
        .bearer_auth(WORKER)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    for name in ["model.onnx", "tokenizer.json", "manifest.json"] {
        state
            .store
            .put(
                &format!("{}/{name}", spec["models_prefix"].as_str().unwrap()),
                std::fs::read(model_dir.join(name)).unwrap().into(),
            )
            .await
            .unwrap();
    }
    let r = http.post(format!("{url}/internal/jobs/{}/complete",spec["job_id"].as_str().unwrap())).bearer_auth(WORKER)
        .json(&json!({"ok":true,"threshold":0.0,"metrics":{"inputs":10,"labeled":10,"cascade":{"target_accuracy":0.95}}})).send().await.unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let monitor = |p: &system_none_gateway::PromptState| {
        p.deployed()
            .map(|d| d.monitor.lock().unwrap().clone())
            .unwrap()
    };

    // With every audit slot busy, the check is skipped rather than awaited.
    let slots = state
        .audit_slots
        .clone()
        .acquire_many_owned(32)
        .await
        .unwrap();
    assert_eq!(
        classify("refund while audits are busy").await["source"],
        "system_one"
    );
    assert_eq!((monitor(&p).audits, monitor(&p).skipped), (0, 1));
    drop(slots);

    // In shadow, System One answers each request once, and the judge (the student's teacher, not
    // System One) checks the student's prediction in the background.
    let (s1_before, judge_before) = (
        s1_calls.load(Ordering::SeqCst),
        judge_calls.load(Ordering::SeqCst),
    );
    for input in ["refund second", "refund third"] {
        assert_eq!(classify(input).await["source"], "system_one");
    }
    assert_eq!(s1_calls.load(Ordering::SeqCst), s1_before + 2);
    wait_until(|| monitor(&p).phase == "live").await;
    assert_eq!(monitor(&p).phase, "live");
    assert_eq!(judge_calls.load(Ordering::SeqCst), judge_before + 2);
    state.flush_usage().unwrap();
    let routes = state
        .db
        .path_latencies(&pid, &system_none_gateway::db::today())
        .unwrap();
    assert_eq!(
        routes.iter().map(|row| &row["path"]).collect::<Vec<_>>(),
        [&json!(["system_one:reference"])],
        "shadow predictions are outside the response path"
    );

    // Live, the student answers. The judge disagrees on cancellations, clearly enough to stop it,
    // and one retraining job is queued at once: no waiting for a larger corpus.
    for input in ["cancel fourth", "cancel fifth"] {
        assert_eq!(classify(input).await["source"], "student");
    }
    wait_until(|| p.deployed().is_none()).await;
    assert!(
        p.deployed().is_none(),
        "clear disagreement stops student answers"
    );
    let saved = state.db.monitor(&pid).unwrap().unwrap();
    assert_eq!(saved.phase, "demoted");
    assert_eq!(saved.failure.unwrap().phase, "live");
    let retry = state.db.open_job_for(&pid).unwrap().unwrap();
    assert_eq!(retry.spec["reason"], "drift");
    let retry_spec = state.job_spec(&retry).unwrap();
    assert_eq!(retry_spec["judge"]["model"], "trainer");
    assert!(retry_spec.get("minimum_inputs").is_none());

    // Re-running the test is an admin action, uses the current settings and cancels the queued
    // retraining. It never lets the student answer at once.
    let audit_url = format!("{url}/admin/api/prompts/{pid}/audit");
    assert_eq!(http.post(&audit_url).send().await.unwrap().status(), 401);
    assert_eq!(
        http.put(format!("{url}/admin/api/settings"))
            .bearer_auth(ADMIN)
            .json(&json!({"default_target_accuracy":0.90}))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    let restarted = http
        .post(&audit_url)
        .bearer_auth(ADMIN)
        .send()
        .await
        .unwrap();
    assert_eq!(restarted.status(), 200);
    let restarted: Value = restarted.json().await.unwrap();
    assert_eq!(restarted["status"], "shadow");
    assert_eq!(restarted["required_agreement"], 0.90);
    assert!(state.db.open_job_for(&pid).unwrap().is_none());
    assert!(state
        .db
        .job(&retry.id)
        .unwrap()
        .unwrap()
        .cancelled_by_admin());
    assert_eq!(monitor(&p).phase, "shadow");
    assert_eq!(monitor(&p).audits, 0);
    assert!(state.db.deployments().unwrap()[0].active);
    assert_eq!(
        classify("refund after restarting the test").await["source"],
        "system_one"
    );

    // A student that fails its shadow test stops there: retraining on the same data would train
    // the same student, so no job is queued and an admin decides what changes.
    wait_until(|| monitor(&p).audits == 1).await;
    assert_eq!(classify("cancel sixth").await["source"], "system_one");
    wait_until(|| p.deployed().is_none()).await;
    assert!(p.deployed().is_none());
    assert!(state.db.open_job_for(&pid).unwrap().is_none());
    assert_eq!(p.status(), "rejected");
    assert_eq!(state.db.monitor(&pid).unwrap().unwrap().phase, "rejected");

    // While a training job runs, the test can't be restarted.
    state.queue_job(&p, "manual").unwrap();
    state.db.claim_job().unwrap().unwrap();
    assert_eq!(
        http.post(&audit_url)
            .bearer_auth(ADMIN)
            .send()
            .await
            .unwrap()
            .status(),
        409
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    clippy::too_many_lines,
    reason = "one scenario, step by step, against one gateway"
)]
async fn full_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let s1 = mock_system_one(calls.clone()).await;
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();

    // No System One configured yet: a clear 503.
    let r = http
        .post(format!("{url}/v1/classify"))
        .json(&json!({"prompt": prompt(), "input": "refund please"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);

    // Admin endpoints need the token.
    let r = http
        .get(format!("{url}/admin/api/state"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);
    let admin = |req: reqwest::RequestBuilder| req.bearer_auth(ADMIN);

    let r = admin(http.put(format!("{url}/admin/api/system-one-models/fast-s1")))
        .json(&json!({"kind": "openai_chat", "base_url": format!("http://{s1}/v1"), "model": "s1"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    // TypeSafe Jev may serve but never judge.
    let r = admin(http.put(format!("{url}/admin/api/judges/jev")))
        .json(&json!({"base_url": "https://api.typesafe.example/v1", "model": "typesafe/jev-2", "license_ack": true}))
        .send().await.unwrap();
    assert_eq!(r.status(), 400);
    assert!(r.text().await.unwrap().contains("2.3(b)"));
    // An endpoint's key never comes from the gateway's own credentials.
    let r = admin(http.put(format!("{url}/admin/api/system-one-models/leaky")))
        .json(&json!({"kind": "openai_chat", "base_url": "https://attacker.example/v1", "api_key_env": "SN_WORKER_TOKEN"}))
        .send().await.unwrap();
    assert_eq!(r.status(), 400);
    let r = admin(http.put(format!("{url}/admin/api/judges/leaky")))
        .json(&json!({"base_url": "https://attacker.example/v1", "model": "m", "license_ack": true, "api_key_env": "AWS_SECRET_ACCESS_KEY"}))
        .send().await.unwrap();
    assert_eq!(r.status(), 400);
    let r = admin(http.put(format!("{url}/admin/api/judges/open-judge")))
        .json(&json!({"base_url": format!("http://{s1}/v1"), "model": "qwen3-32b", "license_ack": true}))
        .send().await.unwrap();
    assert_eq!(r.status(), 200);
    let r = admin(http.put(format!("{url}/admin/api/settings")))
        .json(&json!({"default_system_one": "fast-s1", "default_judge": "open-judge", "default_threshold": 5,
                      "default_base_model": "microsoft/mdeberta-v3-base", "monitor": {"enabled": false}}))
        .send().await.unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());

    // Requests are answered by System One; inputs are logged; the threshold queues a job.
    let mut pid = String::new();
    for text in [
        "I want a refund",
        "cancel my plan",
        "opening hours?",
        "refund me now",
        "please cancel",
        "hello",
    ] {
        let v: Value = http
            .post(format!("{url}/v1/classify"))
            .json(&json!({"prompt": prompt(), "input": text}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(v["source"], "system_one", "{v}");
        pid = v["prompt_id"].as_str().unwrap().to_string();
        if text.contains("refund") {
            assert_eq!(v["label"], "refund");
            assert!(v["confidence"].as_f64().unwrap() > 0.9);
        }
    }
    // A known prompt can be referenced by id alone.
    let v: Value = http
        .post(format!("{url}/v1/classify"))
        .json(&json!({"prompt_id": pid, "input": "cancel"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["label"], "cancel");
    assert_eq!(calls.load(Ordering::SeqCst), 7);

    state.flush_all().await.unwrap();
    let list: Value = http
        .get(format!("{url}/api/prompts"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list[0]["id"], pid.as_str());
    assert_eq!(list[0]["total"], 7);
    assert_eq!(list[0]["status"], "queued");
    assert_eq!(
        list[0]["daily"].as_array().unwrap().last().unwrap()["system_one"],
        7
    );
    let summary: Value = http
        .get(format!("{url}/api/summary"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        summary["requests_today"], 7,
        "answered requests; the 503 counts as an error"
    );
    assert_eq!(list[0]["errors_7_days"], 1);

    let keys = state.store.list(&format!("inputs/{pid}/")).await.unwrap();
    let logged: usize = futures_count(&state, &keys).await;
    assert_eq!(
        logged, 8,
        "every input is logged, including the one that got a 503"
    );

    // The worker claims the job and gets everything it needs.
    let r = http
        .post(format!("{url}/internal/jobs/claim"))
        .bearer_auth("wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);
    let spec: Value = http
        .post(format!("{url}/internal/jobs/claim"))
        .bearer_auth(WORKER)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(spec["prompt_id"], pid.as_str());
    assert_eq!(spec["judge"]["model"], "qwen3-32b");
    assert_eq!(spec["base_model"], "microsoft/mdeberta-v3-base");
    assert_eq!(spec["inputs_prefix"], format!("inputs/{pid}/"));
    let r = http
        .post(format!("{url}/internal/jobs/claim"))
        .bearer_auth(WORKER)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 204, "one job per prompt");
    let job = spec["job_id"].as_str().unwrap().to_string();

    if let Ok(model_dir) = std::env::var("SN_TEST_MODEL_DIR") {
        // Publish a real student and check that confident answers now come from it.
        for f in ["model.onnx", "tokenizer.json", "manifest.json"] {
            let data = std::fs::read(std::path::Path::new(&model_dir).join(f)).unwrap();
            state
                .store
                .put(
                    &format!("{}/{f}", spec["models_prefix"].as_str().unwrap()),
                    data.into(),
                )
                .await
                .unwrap();
        }
        let r: Value = http
            .post(format!("{url}/internal/jobs/{job}/complete"))
            .bearer_auth(WORKER)
            .json(&json!({"ok": true, "threshold": 0.0, "metrics": {"accuracy": 1.0}}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(r["deployed"], true, "{r}");
        let before = calls.load(Ordering::SeqCst);
        let v: Value = http
            .post(format!("{url}/v1/classify"))
            .json(&json!({"prompt_id": pid, "input": "refund refund money back"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(v["source"], "student", "{v}");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            before,
            "System One was not called"
        );
        // Undeploy: back to System One.
        admin(http.post(format!("{url}/admin/api/prompts/{pid}/undeploy")))
            .send()
            .await
            .unwrap();
        let v: Value = http
            .post(format!("{url}/v1/classify"))
            .json(&json!({"prompt_id": pid, "input": "refund"}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(v["source"], "system_one");
    } else {
        // Without a model, completing with a missing student fails cleanly.
        let r = http
            .post(format!("{url}/internal/jobs/{job}/complete"))
            .bearer_auth(WORKER)
            .json(&json!({"ok": true, "threshold": 0.9}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 422);
        let st: Value = admin(http.get(format!("{url}/admin/api/state")))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(st["jobs"][0]["status"], "failed");
        assert_eq!(
            st["jobs"].as_array().unwrap().len(),
            1,
            "retraining wouldn't make the student load: no retry"
        );
        assert_eq!(st["prompts"][0]["status"], "failed");
    }

    // Per-prompt overrides are validated.
    let r = admin(http.put(format!("{url}/admin/api/prompts/{pid}")))
        .json(&json!({"judge": "nope"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    let r = admin(http.put(format!("{url}/admin/api/prompts/{pid}")))
        .json(&json!({"base_model": "microsoft/deberta-v3-small", "threshold": 100}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    // A failing System One model is named to the caller, but its URL stays in the log.
    let closed = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap();
    let r = admin(http.put(format!("{url}/admin/api/system-one-models/down")))
        .json(&json!({"kind": "openai_chat", "base_url": format!("http://{closed}/v1"), "model": "s1", "timeout_ms": 5000}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let r = http
        .post(format!("{url}/v1/classify"))
        .json(&json!({"prompt": prompt(), "input": "hello", "model": "down"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);
    let body = r.text().await.unwrap();
    assert!(
        body.contains("'down'") && !body.contains(&closed.to_string()),
        "{body}"
    );

    // The UIs are served.
    assert!(http
        .get(format!("{url}/"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap()
        .contains("<title>"));
    assert!(http
        .get(format!("{url}/admin"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap()
        .contains("<title>"));
}

#[tokio::test]
async fn tokens_and_security_headers() {
    let dir = tempfile::tempdir().unwrap();
    let (url, _state) = start_gateway(dir.path(), Some("viewer-secret")).await;
    let http = reqwest::Client::new();

    // Usage takes the viewer token or the admin token; admin takes only its own.
    for (token, usage, admin) in [
        (None, 401, 401),
        (Some("viewer-secreT"), 401, 401),
        (Some("viewer-secret"), 200, 401),
        (Some(ADMIN), 200, 200),
        (Some(WORKER), 401, 401),
    ] {
        let auth = |req: reqwest::RequestBuilder| match token {
            Some(t) => req.bearer_auth(t),
            None => req,
        };
        let r = auth(http.get(format!("{url}/api/summary")))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), usage, "usage with {token:?}");
        let r = auth(http.get(format!("{url}/admin/api/state")))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), admin, "admin with {token:?}");
    }

    for page in ["/", "/admin", "/api/summary"] {
        let r = http.get(format!("{url}{page}")).send().await.unwrap();
        let h = r.headers();
        assert!(h["content-security-policy"]
            .to_str()
            .unwrap()
            .contains("frame-ancestors 'none'"));
        assert_eq!(h["x-frame-options"], "DENY");
        assert_eq!(h["x-content-type-options"], "nosniff");
    }
}

#[test]
fn debug_output_never_shows_a_token() {
    let cfg = GatewayConfig {
        admin_token: ADMIN.into(),
        worker_token: WORKER.into(),
        viewer_token: Some("viewer-secret".into()),
        ..GatewayConfig::default()
    };
    let shown = format!("{cfg:?}");
    for token in [ADMIN, WORKER, "viewer-secret"] {
        assert!(!shown.contains(token), "{token} in {shown}");
    }
    assert!(shown.contains(r#"admin_token: "<set>""#), "{shown}");
    let cfg = GatewayConfig::default();
    assert!(format!("{cfg:?}").contains(r#"viewer_token: "<not set>""#));
}

/// verification/JobLifecycle.tla, `FinishedJobsStay`: a cancel doesn't stop the worker, which
/// keeps reporting progress; the job must keep the reason it was stopped, and its student
/// must never deploy.
#[tokio::test]
async fn a_cancelled_job_stays_cancelled() {
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let admin = |req: reqwest::RequestBuilder| req.bearer_auth(ADMIN);
    let r = admin(http.put(format!("{url}/admin/api/judges/open-judge")))
        .json(&json!({"base_url": "http://127.0.0.1:9/v1", "model": "qwen3-32b", "license_ack": true}))
        .send().await.unwrap();
    assert_eq!(r.status(), 200);
    let r = admin(http.put(format!("{url}/admin/api/settings")))
        .json(&json!({"default_judge": "open-judge"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let p = state
        .resolve_prompt(serde_json::from_value(prompt()).unwrap())
        .await
        .unwrap();
    let r = admin(http.post(format!("{url}/admin/api/prompts/{}/train", p.id)))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let spec: Value = http
        .post(format!("{url}/internal/jobs/claim"))
        .bearer_auth(WORKER)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let job = spec["job_id"].as_str().unwrap();
    let r = admin(http.post(format!("{url}/admin/api/jobs/{job}/cancel")))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    let r = http
        .post(format!("{url}/internal/jobs/{job}/progress"))
        .bearer_auth(WORKER)
        .json(&json!({"message": "fine-tuning"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 409, "the worker can tell its job was stopped");
    let r = http
        .post(format!("{url}/internal/jobs/{job}/complete"))
        .bearer_auth(WORKER)
        .json(&json!({"ok": true, "threshold": 0.9}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 409);
    let j = state.db.job(job).unwrap().unwrap();
    assert_eq!(j.status, "failed");
    assert_eq!(j.message.as_deref(), Some("cancelled by an admin"));
    assert!(p.deployed().is_none());
    assert_eq!(p.status(), "failed");
}

/// The admin page's jobs, newest first.
async fn listed_jobs(http: &reqwest::Client, url: &str) -> Vec<Value> {
    let st: Value = http
        .get(format!("{url}/admin/api/state"))
        .bearer_auth(ADMIN)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    st["jobs"].as_array().unwrap().clone()
}

/// The admin page's newest job.
async fn newest_job(http: &reqwest::Client, url: &str) -> Value {
    listed_jobs(http, url).await.remove(0)
}

/// The worker's counts reach the admin page's job list with their message. A message without
/// counts (the next stage) clears them, and so does the job's end.
#[tokio::test]
async fn progress_counts_reach_the_admin_page() {
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let r = http
        .put(format!("{url}/admin/api/judges/open-judge"))
        .bearer_auth(ADMIN)
        .json(&json!({"base_url": "http://127.0.0.1:9/v1", "model": "qwen3-32b", "license_ack": true}))
        .send().await.unwrap();
    assert_eq!(r.status(), 200);
    let r = http
        .put(format!("{url}/admin/api/settings"))
        .bearer_auth(ADMIN)
        .json(&json!({"default_judge": "open-judge"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let p = state
        .resolve_prompt(serde_json::from_value(prompt()).unwrap())
        .await
        .unwrap();
    let job = state.queue_job(&p, "manual").unwrap().unwrap().id;
    let progress = |body: Value| {
        http.post(format!("{url}/internal/jobs/{job}/progress"))
            .bearer_auth(WORKER)
            .json(&body)
            .send()
    };
    let judging = "judging 20 inputs with qwen3-32b";
    let counted = json!({"message": judging, "done": 5, "total": 20});
    assert_eq!(
        progress(counted.clone()).await.unwrap().status(),
        409,
        "queued"
    );
    let r = http
        .post(format!("{url}/internal/jobs/claim"))
        .bearer_auth(WORKER)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    assert_eq!(progress(counted).await.unwrap().status(), 204);
    let j = newest_job(&http, &url).await;
    assert_eq!(
        (&j["status"], &j["message"], &j["done"], &j["total"]),
        (&json!("running"), &json!(judging), &json!(5), &json!(20)),
        "{j}"
    );
    // Counts come in pairs, done at most total; the job keeps the ones it had.
    for body in [
        json!({"message": "x", "done": 21, "total": 20}),
        json!({"message": "x", "done": 5}),
        json!({"message": "x", "total": 20}),
    ] {
        assert_eq!(
            progress(body.clone()).await.unwrap().status(),
            400,
            "{body}"
        );
    }
    assert_eq!(newest_job(&http, &url).await["done"], 5);

    let training = "training a student on microsoft/mdeberta-v3-base with 20 judged inputs";
    assert_eq!(
        progress(json!({"message": training}))
            .await
            .unwrap()
            .status(),
        204
    );
    let j = newest_job(&http, &url).await;
    assert_eq!(
        (&j["message"], &j["done"], &j["total"]),
        (&json!(training), &Value::Null, &Value::Null),
        "{j}"
    );

    let r = progress(json!({"message": training, "done": 20, "total": 20}))
        .await
        .unwrap();
    assert_eq!(r.status(), 204);
    let r = http
        .post(format!("{url}/internal/jobs/{job}/complete"))
        .bearer_auth(WORKER)
        .json(&json!({"ok": false, "message": "not enough labeled data to train"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let listed = listed_jobs(&http, &url).await; // the newest is now its retry
    let j = listed.iter().find(|j| j["id"] == job.as_str()).unwrap();
    assert_eq!(
        (&j["status"], &j["done"], &j["total"]),
        (&json!("failed"), &Value::Null, &Value::Null),
        "{j}"
    );
}

/// A judge for the worker's jobs, and these settings.
async fn set_up_training(http: &reqwest::Client, url: &str, mut settings: Value) {
    let r = http
        .put(format!("{url}/admin/api/judges/open-judge"))
        .bearer_auth(ADMIN)
        .json(&json!({"base_url": "http://127.0.0.1:9/v1", "model": "qwen3-32b", "license_ack": true}))
        .send().await.unwrap();
    assert_eq!(r.status(), 200);
    settings["default_judge"] = json!("open-judge");
    let r = http
        .put(format!("{url}/admin/api/settings"))
        .bearer_auth(ADMIN)
        .json(&settings)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
}

/// The job a worker's claim gets, if any.
async fn claim(http: &reqwest::Client, url: &str) -> Option<String> {
    let r = http
        .post(format!("{url}/internal/jobs/claim"))
        .bearer_auth(WORKER)
        .send()
        .await
        .unwrap();
    if r.status() == 204 {
        return None;
    }
    assert_eq!(r.status(), 200);
    let spec: Value = r.json().await.unwrap();
    Some(spec["job_id"].as_str().unwrap().to_string())
}

/// The worker reports that the job failed.
async fn fail(http: &reqwest::Client, url: &str, job: &str) -> reqwest::Response {
    http.post(format!("{url}/internal/jobs/{job}/complete"))
        .bearer_auth(WORKER)
        .json(&json!({"ok": false, "message": "judge unreachable"}))
        .send()
        .await
        .unwrap()
}

/// The retry queued after the worker reported a failure, as the reply names it.
async fn fail_and_get_retry(state: &AppState, http: &reqwest::Client, url: &str, job: &str) -> Job {
    let r = fail(http, url, job).await;
    assert_eq!(r.status(), 200);
    let reply: Value = r.json().await.unwrap();
    let failed = state.db.job(job).unwrap().unwrap();
    assert_eq!(
        (failed.status.as_str(), failed.message.as_deref()),
        ("failed", Some("judge unreachable"))
    );
    let id = reply["retry"]["job_id"].as_str().expect("a retry");
    state.db.job(id).unwrap().unwrap()
}

/// How long after `failed` finished `retry` may be claimed, in seconds.
fn wait_after(state: &AppState, failed: &str, retry: &Job) -> i64 {
    let time = |t: &str| chrono::DateTime::parse_from_rfc3339(t).unwrap();
    let failed = state.db.job(failed).unwrap().unwrap();
    (time(retry.not_before.as_deref().unwrap()) - time(failed.finished_at.as_deref().unwrap()))
        .num_seconds()
}

async fn train_now(http: &reqwest::Client, url: &str, prompt_id: &str) -> reqwest::Response {
    http.post(format!("{url}/admin/api/prompts/{prompt_id}/train"))
        .bearer_auth(ADMIN)
        .send()
        .await
        .unwrap()
}

/// A job the worker reports failed is queued again, and waits: 60 s after the first failure,
/// twice that after the second. Train now doesn't wait. After the last attempt, the prompt fails
/// as it would without retries.
#[tokio::test]
async fn a_failed_job_is_retried_with_exponential_backoff() {
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    for bad in [
        json!({"max_attempts": 0}),
        json!({"retry_secs": 600, "retry_max_secs": 60}),
        json!({"retry_max_secs": 30 * 24 * 3600}),
    ] {
        let r = http
            .put(format!("{url}/admin/api/settings"))
            .bearer_auth(ADMIN)
            .json(&bad)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400, "{bad}");
    }
    set_up_training(
        &http,
        &url,
        json!({"max_attempts": 3, "retry_secs": 60, "retry_max_secs": 3600}),
    )
    .await;
    let p = state
        .resolve_prompt(serde_json::from_value(prompt()).unwrap())
        .await
        .unwrap();
    let first = state.queue_job(&p, "manual").unwrap().unwrap();
    assert_eq!((first.attempt, first.not_before.as_deref()), (1, None));

    let mut job = first.id;
    for (attempt, wait) in [(2, 60), (3, 120)] {
        assert_eq!(claim(&http, &url).await.as_deref(), Some(job.as_str()));
        let retry = fail_and_get_retry(&state, &http, &url, &job).await;
        assert_eq!((retry.status.as_str(), retry.attempt), ("queued", attempt));
        assert_eq!(retry.spec["retry_of"], job.as_str());
        let waited = wait_after(&state, &job, &retry);
        assert!((wait..=wait + 1).contains(&waited), "waits {waited} s");
        assert_eq!(p.status(), "queued");
        assert_eq!(claim(&http, &url).await, None, "not before its time");
        let r = train_now(&http, &url, &p.id).await;
        assert_eq!(r.status(), 200);
        let hurried: Value = r.json().await.unwrap();
        assert_eq!(
            (&hurried["id"], &hurried["not_before"]),
            (&json!(retry.id), &Value::Null)
        );
        job = retry.id;
    }
    assert_eq!(claim(&http, &url).await.as_deref(), Some(job.as_str()));
    let r = fail(&http, &url, &job).await;
    assert_eq!(r.status(), 200);
    assert_eq!(
        r.json::<Value>().await.unwrap()["retry"],
        Value::Null,
        "the third attempt was the last"
    );
    assert_eq!(p.status(), "failed");
    assert!(state.db.open_job_for(&p.id).unwrap().is_none());
    let jobs = state.db.jobs(10).unwrap();
    assert_eq!(
        jobs.iter().map(|j| j.attempt).collect::<Vec<_>>(),
        [3, 2, 1]
    );
}

/// The attempts count failures in a row: after a success, the next training starts again at
/// attempt 1, and its retry waits the shortest time again.
#[tokio::test]
async fn a_success_resets_the_backoff() {
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    set_up_training(&http, &url, json!({})).await; // 5 attempts, 60 s, doubling
    let p = state
        .resolve_prompt(serde_json::from_value(prompt()).unwrap())
        .await
        .unwrap();
    let mut job = state.queue_job(&p, "manual").unwrap().unwrap().id;
    for _ in 0..2 {
        claim(&http, &url).await.unwrap();
        job = fail_and_get_retry(&state, &http, &url, &job).await.id;
        assert_eq!(train_now(&http, &url, &p.id).await.status(), 200);
    }
    assert_eq!(state.db.job(&job).unwrap().unwrap().attempt, 3);
    assert_eq!(claim(&http, &url).await.as_deref(), Some(job.as_str()));
    // The third attempt succeeds. Without a model its student can't load, so this is what
    // complete_job writes once it has.
    assert!(state
        .db
        .finish_job(&job, true, Some("test agreement 0.97"))
        .unwrap());
    state.db.set_status(&p.id, "deployed").unwrap();
    p.set_status("deployed");

    let r = train_now(&http, &url, &p.id).await;
    assert_eq!(r.status(), 200);
    let next: Value = r.json().await.unwrap();
    assert_eq!(next["attempt"], 1);
    let next = next["id"].as_str().unwrap();
    assert_eq!(claim(&http, &url).await.as_deref(), Some(next));
    let retry = fail_and_get_retry(&state, &http, &url, next).await;
    assert_eq!(retry.attempt, 2);
    let waited = wait_after(&state, next, &retry);
    assert!((60..=61).contains(&waited), "waits {waited} s, not 240");
}

/// verification/JobLifecycle.tla, `CancelledNeverRetried`: an admin's cancel ends the prompt's
/// training, whether its job was running (the worker's failure then comes too late) or was a
/// retry waiting for its time.
#[tokio::test]
async fn a_cancelled_job_is_not_retried() {
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    set_up_training(&http, &url, json!({})).await;
    let p = state
        .resolve_prompt(serde_json::from_value(prompt()).unwrap())
        .await
        .unwrap();
    let cancel = |job: String| {
        http.post(format!("{url}/admin/api/jobs/{job}/cancel"))
            .bearer_auth(ADMIN)
            .send()
    };

    let job = state.queue_job(&p, "manual").unwrap().unwrap().id;
    claim(&http, &url).await.unwrap();
    assert_eq!(cancel(job.clone()).await.unwrap().status(), 200);
    assert_eq!(fail(&http, &url, &job).await.status(), 409);
    assert_eq!(state.db.jobs(10).unwrap().len(), 1, "no retry");
    assert_eq!(p.status(), "failed");

    let job = state.queue_job(&p, "manual").unwrap().unwrap().id;
    claim(&http, &url).await.unwrap();
    let retry = fail_and_get_retry(&state, &http, &url, &job).await;
    assert_eq!(cancel(retry.id.clone()).await.unwrap().status(), 200);
    let retry = state.db.job(&retry.id).unwrap().unwrap();
    assert_eq!(
        (retry.status.as_str(), retry.message.as_deref()),
        ("failed", Some("cancelled by an admin"))
    );
    assert_eq!(p.status(), "failed");
    assert!(state.db.open_job_for(&p.id).unwrap().is_none());
    assert_eq!(claim(&http, &url).await, None);
    assert_eq!(state.db.jobs(10).unwrap().len(), 3);
}

async fn cancel(http: &reqwest::Client, url: &str, job: &str) -> reqwest::Response {
    http.post(format!("{url}/admin/api/jobs/{job}/cancel"))
        .bearer_auth(ADMIN)
        .send()
        .await
        .unwrap()
}

async fn retry(http: &reqwest::Client, url: &str, job: &str) -> reqwest::Response {
    http.post(format!("{url}/admin/api/jobs/{job}/retry"))
        .bearer_auth(ADMIN)
        .send()
        .await
        .unwrap()
}

/// The admin's Retry is refused with this status, and says why.
async fn retry_refused(http: &reqwest::Client, url: &str, job: &str, status: u16) -> String {
    let r = retry(http, url, job).await;
    assert_eq!(r.status(), status, "retry of {job}");
    let reply: Value = r.json().await.unwrap();
    reply["error"].as_str().unwrap().to_string()
}

/// verification/JobLifecycle.tla, `AdminRetriesFollowCancels`: an admin's Retry queues a job an
/// admin cancelled again, as a new job for its prompt that starts over like Train now: attempt 1,
/// claimable at once, and its failure waits the shortest time again. The cancelled job stays
/// cancelled (`FinishedJobsStay`). Only the admin token may ask.
#[tokio::test]
async fn an_admin_retries_a_cancelled_job() {
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    set_up_training(&http, &url, json!({})).await; // 5 attempts, 60 s, doubling
    let p = state
        .resolve_prompt(serde_json::from_value(prompt()).unwrap())
        .await
        .unwrap();
    // The second attempt is cancelled while it waits for its time.
    let first = state.queue_job(&p, "manual").unwrap().unwrap().id;
    claim(&http, &url).await.unwrap();
    let waiting = fail_and_get_retry(&state, &http, &url, &first).await;
    assert_eq!(waiting.attempt, 2);
    assert_eq!(cancel(&http, &url, &waiting.id).await.status(), 200);
    let cancelled = state.db.job(&waiting.id).unwrap().unwrap();
    assert!(cancelled.cancelled_by_admin());
    assert_eq!(p.status(), "failed");
    let listed = listed_jobs(&http, &url).await;
    assert_eq!(
        (&listed[0]["id"], &listed[0]["phase"]),
        (&json!(waiting.id), &json!("cancelled")),
        "the admin page can tell it from a failure"
    );
    assert_eq!(listed[1]["phase"], "failed");

    for token in [None, Some(WORKER), Some("admin-secreT")] {
        let req = http.post(format!("{url}/admin/api/jobs/{}/retry", waiting.id));
        let req = match token {
            Some(t) => req.bearer_auth(t),
            None => req,
        };
        assert_eq!(req.send().await.unwrap().status(), 401, "with {token:?}");
    }
    assert!(state.db.open_job_for(&p.id).unwrap().is_none());

    let r = retry(&http, &url, &waiting.id).await;
    assert_eq!(r.status(), 200);
    let again: Job = r.json().await.unwrap();
    assert_eq!(
        (
            again.status.as_str(),
            again.attempt,
            again.not_before.as_deref()
        ),
        ("queued", 1, None)
    );
    assert_eq!(
        (&again.spec["reason"], &again.spec["retry_of"]),
        (&json!("manual_retry"), &json!(waiting.id))
    );
    assert_eq!(p.status(), "queued");
    assert_eq!(state.db.prompts().unwrap()[0].status, "queued");
    let still = state.db.job(&waiting.id).unwrap().unwrap();
    assert_eq!(
        (
            still.status,
            still.message,
            still.finished_at,
            still.attempt
        ),
        (
            cancelled.status,
            cancelled.message,
            cancelled.finished_at,
            cancelled.attempt
        ),
        "the cancelled job stays as it was"
    );
    assert_eq!(
        retry_refused(&http, &url, &waiting.id, 409).await,
        "a training job for this prompt is already queued or running",
        "one Retry queues one job"
    );

    assert_eq!(claim(&http, &url).await, Some(again.id.clone()), "no wait");
    let next = fail_and_get_retry(&state, &http, &url, &again.id).await;
    assert_eq!(next.attempt, 2);
    let waited = wait_after(&state, &again.id, &next);
    assert!((60..=61).contains(&waited), "waits {waited} s, not 120");
    assert_eq!(state.db.jobs(10).unwrap().len(), 4);
}

/// Retry is only for a job an admin cancelled, whose prompt has no job queued or running: a job
/// that is queued, running, done or failed on its own gets a 409, and so does a cancelled job
/// while its prompt has another; an unknown job, or one whose prompt is gone, gets a 404. None of
/// them queues anything.
#[tokio::test]
async fn only_a_cancelled_job_can_be_retried() {
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    set_up_training(&http, &url, json!({})).await;
    let p = state
        .resolve_prompt(serde_json::from_value(prompt()).unwrap())
        .await
        .unwrap();
    let not_cancelled =
        |this: &str| format!("only a job an admin cancelled can be retried; this one {this}");

    let done = state.queue_job(&p, "manual").unwrap().unwrap().id;
    claim(&http, &url).await.unwrap();
    // It succeeds. Without a model its student can't load, so this is what complete_job writes
    // once it has.
    assert!(state
        .db
        .finish_job(&done, true, Some("test agreement 0.97"))
        .unwrap());
    state.db.set_status(&p.id, "deployed").unwrap();
    p.set_status("deployed");
    assert_eq!(
        retry_refused(&http, &url, &done, 409).await,
        not_cancelled("is done")
    );

    let r = train_now(&http, &url, &p.id).await;
    assert_eq!(r.status(), 200);
    let job = r.json::<Value>().await.unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(
        retry_refused(&http, &url, &job, 409).await,
        not_cancelled("is queued")
    );
    claim(&http, &url).await.unwrap();
    assert_eq!(
        retry_refused(&http, &url, &job, 409).await,
        not_cancelled("is running")
    );
    let waiting = fail_and_get_retry(&state, &http, &url, &job).await;
    assert_eq!(
        retry_refused(&http, &url, &job, 409).await,
        not_cancelled("failed on its own"),
        "the automatic retries and Train now are for that"
    );

    // Cancelled, while its prompt has a job queued, then running.
    assert_eq!(cancel(&http, &url, &waiting.id).await.status(), 200);
    let r = train_now(&http, &url, &p.id).await;
    assert_eq!(r.status(), 200);
    let busy = "a training job for this prompt is already queued or running";
    assert_eq!(retry_refused(&http, &url, &waiting.id, 409).await, busy);
    claim(&http, &url).await.unwrap();
    assert_eq!(retry_refused(&http, &url, &waiting.id, 409).await, busy);

    assert_eq!(
        retry_refused(&http, &url, "0123456789ab", 404).await,
        "unknown job"
    );
    let orphan = state.db.create_job("gone", &json!({})).unwrap().id;
    assert_eq!(cancel(&http, &url, &orphan).await.status(), 200);
    assert_eq!(
        retry_refused(&http, &url, &orphan, 404).await,
        "the job's prompt no longer exists"
    );

    assert_eq!(state.db.jobs(10).unwrap().len(), 5, "nothing was queued");
    assert_eq!(p.status(), "training");
}

/// verification/JobLifecycle.tla, `AtMostOneOpenJob`: while a retry waits, the threshold queues
/// nothing, even for a prompt that is collecting again, and Train now runs that retry instead of
/// queueing another job.
#[tokio::test]
async fn a_waiting_retry_is_the_only_open_job() {
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    set_up_training(&http, &url, json!({"default_threshold": 1})).await;
    let p = state
        .resolve_prompt(serde_json::from_value(prompt()).unwrap())
        .await
        .unwrap();
    let job = state.queue_job(&p, "manual").unwrap().unwrap().id;
    claim(&http, &url).await.unwrap();
    let retry = fail_and_get_retry(&state, &http, &url, &job).await;
    let open = || {
        state
            .db
            .jobs(10)
            .unwrap()
            .into_iter()
            .filter(|j| j.status == "queued" || j.status == "running")
            .map(|j| j.id)
            .collect::<Vec<_>>()
    };

    state.count_request(&p); // past the threshold, but the prompt is queued
    assert!(state.queue_job(&p, "threshold").unwrap().is_none());
    // "Stop using student" sets the prompt collecting, whatever its jobs.
    let r = http
        .post(format!("{url}/admin/api/prompts/{}/undeploy", p.id))
        .bearer_auth(ADMIN)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    state.count_request(&p);
    assert!(state.queue_job(&p, "threshold").unwrap().is_none());
    assert_eq!(open(), std::slice::from_ref(&retry.id));
    assert!(
        state
            .db
            .job(&retry.id)
            .unwrap()
            .unwrap()
            .not_before
            .is_some(),
        "the threshold doesn't hurry a retry"
    );

    let r = train_now(&http, &url, &p.id).await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<Value>().await.unwrap()["id"], retry.id.as_str());
    assert_eq!(
        train_now(&http, &url, &p.id).await.status(),
        409,
        "it no longer waits: it is queued"
    );
    assert_eq!(open(), std::slice::from_ref(&retry.id));
    assert_eq!(claim(&http, &url).await, Some(retry.id));
}

/// verification/JobLifecycle.tla, `ThresholdOnlyWhileCollecting`: `count_request` sees a collecting
/// prompt, then waits for `queue_lock` while an admin queues a job and cancels it. With the lock,
/// it must not queue another: the prompt isn't collecting anymore.
#[tokio::test]
async fn the_threshold_queues_only_collecting_prompts() {
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let p = state
        .resolve_prompt(serde_json::from_value(prompt()).unwrap())
        .await
        .unwrap();
    assert_eq!(p.status(), "collecting"); // what count_request saw
    let job = state.queue_job(&p, "manual").unwrap().unwrap();
    let r = http
        .post(format!("{url}/admin/api/jobs/{}/cancel", job.id))
        .bearer_auth(ADMIN)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert!(state.queue_job(&p, "threshold").unwrap().is_none());
    assert_eq!(p.status(), "failed");
    assert!(
        state.queue_job(&p, "manual").unwrap().is_some(),
        "Train now still works"
    );
}

async fn futures_count(state: &AppState, keys: &[String]) -> usize {
    let mut n = 0;
    for k in keys {
        let body = state.store.get(k).await.unwrap();
        for line in String::from_utf8(body.to_vec()).unwrap().lines() {
            let v: Value = serde_json::from_str(line).unwrap();
            // Inputs only, and the router's view of their language: never an answer.
            let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
            assert!(
                keys.iter()
                    .all(|k| ["id", "text", "ts", "lang", "lang_confidence"].contains(&k.as_str())),
                "{keys:?}"
            );
            n += 1;
        }
    }
    n
}

/// A small System One model (`decision_http`): unsure about anything that says "maybe".
async fn mock_weak_system_one() -> SocketAddr {
    let app = Router::new().route(
        "/decide",
        post(|Json(body): Json<Value>| async move {
            let text = body["input"].as_str().unwrap_or("");
            Json(if text.contains("maybe") {
                json!({"label": "other", "confidence": 0.4})
            } else {
                json!({"label": "refund", "confidence": 0.9})
            })
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    clippy::too_many_lines,
    reason = "exercise escalation, route metrics and failure in one scenario"
)]
async fn unsure_system_one_escalates_to_a_stronger_model() {
    let dir = tempfile::tempdir().unwrap();
    let llm_calls = Arc::new(AtomicUsize::new(0));
    let llm = mock_system_one(llm_calls.clone()).await;
    let weak = mock_weak_system_one().await;
    let (url, _state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let put = |name: &str, body: Value| {
        http.put(format!("{url}/admin/api/system-one-models/{name}"))
            .bearer_auth(ADMIN)
            .json(&body)
            .send()
    };

    let llm_model =
        json!({"kind": "openai_chat", "base_url": format!("http://{llm}/v1"), "model": "big-llm"});
    assert_eq!(put("llm", llm_model).await.unwrap().status(), 200);
    let weak_model = |to: Value, below: Value| {
        json!({"kind": "decision_http", "base_url": format!("http://{weak}/decide"), "model": "jev",
               "escalate_to": to, "escalate_below": below})
    };
    // Escalation needs a known target that isn't the model itself, and a threshold in (0, 1].
    for (to, below) in [
        (json!("llm"), json!(null)),
        (json!("nowhere"), json!(0.6)),
        (json!("weak"), json!(0.6)),
        (json!("llm"), json!(1.5)),
    ] {
        let r = put("weak", weak_model(to.clone(), below.clone()))
            .await
            .unwrap();
        assert_eq!(r.status(), 400, "escalate_to {to}, escalate_below {below}");
    }
    assert_eq!(
        put("weak", weak_model(json!("llm"), json!(0.6)))
            .await
            .unwrap()
            .status(),
        200
    );
    let r = http
        .put(format!("{url}/admin/api/settings"))
        .bearer_auth(ADMIN)
        .json(&json!({"default_system_one": "weak", "default_threshold": 1000}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());

    let classify = |input: &'static str| {
        let http = http.clone();
        let url = url.clone();
        async move {
            http.post(format!("{url}/v1/classify"))
                .json(&json!({"prompt": prompt(), "input": input}))
                .send()
                .await
                .unwrap()
                .json::<Value>()
                .await
                .unwrap()
        }
    };
    // Confident: the small model answers alone.
    let v = classify("refund please").await;
    assert_eq!(
        (v["model"].as_str(), v["label"].as_str()),
        (Some("weak"), Some("refund")),
        "{v}"
    );
    assert!(v.get("escalated_from").is_none(), "{v}");
    assert_eq!(llm_calls.load(Ordering::SeqCst), 0);
    // Unsure: the LLM answers instead.
    let v = classify("maybe cancel it?").await;
    assert_eq!(v["source"], "system_one", "{v}");
    assert_eq!(
        (v["model"].as_str(), v["label"].as_str()),
        (Some("llm"), Some("cancel")),
        "{v}"
    );
    assert_eq!(v["escalated_from"], "weak", "{v}");
    assert_eq!(llm_calls.load(Ordering::SeqCst), 1);
    let routes: Value = http
        .get(format!("{url}/api/prompts"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let paths = routes[0]["path_latencies_7_days"].as_array().unwrap();
    assert_eq!(paths.len(), 2);
    assert!(paths
        .iter()
        .any(|r| r["path"] == json!(["system_one:weak", "system_one:llm"])));
    assert!(paths
        .iter()
        .all(|r| r["p50_ms"].as_f64().unwrap() <= r["max_ms"].as_f64().unwrap()));

    // When the stronger model is down, the small model's answer stands.
    let down =
        json!({"kind": "openai_chat", "base_url": "http://127.0.0.1:9/v1", "model": "big-llm"});
    assert_eq!(put("llm", down).await.unwrap().status(), 200);
    let v = classify("maybe cancel it?").await;
    assert_eq!(
        (v["model"].as_str(), v["label"].as_str()),
        (Some("weak"), Some("other")),
        "{v}"
    );
    assert!(v.get("escalated_from").is_none(), "{v}");
}

/// A Decider stand-in (TypeSafe's POST /v1/systemone): chooses the label the input names, with
/// probability 0.9, or 0.4 when it says "maybe". Keeps every raw request body.
async fn mock_decider(bodies: Arc<Mutex<Vec<String>>>) -> SocketAddr {
    let app = Router::new().route(
        "/v1/systemone",
        post(move |body: String| {
            let bodies = bodies.clone();
            async move {
                bodies.lock().unwrap().push(body.clone());
                let req: Value = serde_json::from_str(&body).unwrap();
                let text = req["state"].as_str().unwrap_or("");
                let names: Vec<String> = req["questions"]["label"]["criteria"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .cloned()
                    .collect();
                let choice = ["surprise"]
                    .into_iter()
                    .chain(names.iter().map(String::as_str))
                    .find(|n| text.contains(n))
                    .unwrap_or("other")
                    .to_string();
                let p = if text.contains("maybe") { 0.4 } else { 0.9 };
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a prompt has at most 255 labels"
                )]
                let k = names.len() as f64;
                let prob = |n: &String| {
                    if *n == choice {
                        p
                    } else {
                        (1.0 - p) / (k - 1.0)
                    }
                };
                let probabilities: serde_json::Map<String, Value> =
                    names.iter().map(|n| (n.clone(), json!(prob(n)))).collect();
                // `confidence` is TypeSafe's (n·p − 1)/(n − 1), not the probability.
                let answer = json!({"type": "choice", "choice": choice, "x_p_max": p,
                    "confidence": (k * p - 1.0) / (k - 1.0), "probabilities": probabilities});
                Json(json!({"model": "decider-2b-v11", "answers": {"label": answer}}))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

/// What `make` sets up: Decider answers as the default System One model (kind `systemone`) and
/// is the judge the worker trains students from (mode `systemone`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    clippy::too_many_lines,
    reason = "one scenario, step by step, against one gateway"
)]
async fn a_decider_server_answers_and_judges() {
    let dir = tempfile::tempdir().unwrap();
    let bodies = Arc::new(Mutex::new(vec![]));
    let decider = mock_decider(bodies.clone()).await;
    let llm_calls = Arc::new(AtomicUsize::new(0));
    let llm = mock_system_one(llm_calls.clone()).await;
    let (url, _state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let put = |path: &str, body: Value| {
        http.put(format!("{url}/admin/api/{path}"))
            .bearer_auth(ADMIN)
            .json(&body)
            .send()
    };
    let decider_model = json!({"kind": "systemone", "base_url": format!("http://{decider}/"),
                               "model": "Mapika/decider-2b"});
    let r = put("system-one-models/decider", decider_model.clone())
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let judge = json!({"base_url": format!("http://{decider}"), "model": "Mapika/decider-2b",
                       "mode": "systemone", "license_ack": true});
    let r = put("judges/decider", judge).await.unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let r = put(
        "settings",
        json!({"default_system_one": "decider", "default_judge": "decider"}),
    )
    .await
    .unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());

    let spec = json!({"instructions": "  What does the customer want?\n",
                      "labels": ["refund", {"name": "cancel", "description": "stop the order"}, "other"]});
    let classify = |input: &'static str| {
        let (http, url, spec) = (http.clone(), url.clone(), spec.clone());
        async move {
            http.post(format!("{url}/v1/classify"))
                .json(&json!({"prompt": spec, "input": input}))
                .send()
                .await
                .unwrap()
        }
    };
    let v: Value = classify("refund please").await.json().await.unwrap();
    assert_eq!(
        (
            v["source"].as_str(),
            v["model"].as_str(),
            v["label"].as_str()
        ),
        (Some("system_one"), Some("decider"), Some("refund")),
        "{v}"
    );
    assert_eq!(v["confidence"], 0.9, "the answer's probability: {v}");
    // One Choice question over the input, its labels in the prompt's order with their descriptions.
    let body = bodies.lock().unwrap()[0].clone();
    let sent: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(sent["state"], "refund please");
    assert_eq!(sent["model"], "Mapika/decider-2b");
    assert_eq!(sent["questions"]["label"]["type"], "choice");
    assert_eq!(
        sent["questions"]["label"]["instructions"],
        "What does the customer want?"
    );
    assert!(
        body.contains(r#""criteria":{"refund":"","cancel":"stop the order","other":""}"#),
        "{body}"
    );
    // An answer outside the prompt's labels is an error.
    assert_eq!(classify("surprise!").await.status(), 502);

    // Unsure answers escalate like any other System One model's.
    let llm_model =
        json!({"kind": "openai_chat", "base_url": format!("http://{llm}/v1"), "model": "big-llm"});
    assert_eq!(
        put("system-one-models/llm", llm_model)
            .await
            .unwrap()
            .status(),
        200
    );
    let mut escalating = decider_model;
    escalating["escalate_to"] = json!("llm");
    escalating["escalate_below"] = json!(0.6);
    assert_eq!(
        put("system-one-models/decider", escalating)
            .await
            .unwrap()
            .status(),
        200
    );
    let v: Value = classify("maybe cancel it").await.json().await.unwrap();
    assert_eq!(
        (
            v["model"].as_str(),
            v["label"].as_str(),
            v["escalated_from"].as_str()
        ),
        (Some("llm"), Some("cancel"), Some("decider")),
        "{v}"
    );
    assert_eq!(llm_calls.load(Ordering::SeqCst), 1);

    // The worker's job names Decider as its judge.
    let id = v["prompt_id"].as_str().unwrap();
    let r = http
        .post(format!("{url}/admin/api/prompts/{id}/train"))
        .bearer_auth(ADMIN)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let job: Value = http
        .post(format!("{url}/internal/jobs/claim"))
        .bearer_auth(WORKER)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        (
            job["judge"]["mode"].as_str(),
            job["judge"]["model"].as_str()
        ),
        (Some("systemone"), Some("Mapika/decider-2b")),
        "{job}"
    );
    assert_eq!(job["judge"]["base_url"], format!("http://{decider}"));
    assert_eq!(job["judge"]["license_ack"], true);
}

/// A stand-in for Fastino's GLiDE (`POST https://api.fastino.ai/v1/systemone`), answering in the
/// Choice shape docs.fastino.ai documents: probability 0.8 on the label the input names, and a
/// `confidence` that is the margin over the runner-up. As Fastino documents, it answers 401
/// without its key as a bearer token, and 422 to a top-level field other than `model`, `state`
/// and `questions`, or to a criterion that isn't a string. Keeps every raw request body.
async fn mock_fastino(key: &'static str, bodies: Arc<Mutex<Vec<String>>>) -> SocketAddr {
    let app = Router::new().route(
        "/v1/systemone",
        post(move |headers: HeaderMap, body: String| {
            let bodies = bodies.clone();
            async move {
                bodies.lock().unwrap().push(body.clone());
                let bearer = format!("Bearer {key}");
                let auth = headers.get("authorization").and_then(|v| v.to_str().ok());
                if auth != Some(bearer.as_str()) {
                    return (
                        StatusCode::UNAUTHORIZED,
                        Json(json!({"detail": "Invalid API key"})),
                    );
                }
                let req: Value = serde_json::from_str(&body).unwrap();
                let question = &req["questions"]["label"];
                let criteria = question["criteria"].as_object();
                let known = |k: &String| ["model", "state", "questions"].contains(&k.as_str());
                let valid = req.as_object().is_some_and(|o| o.keys().all(known))
                    && req["model"].as_str().is_some_and(|m| !m.is_empty())
                    && req["state"].as_str().is_some_and(|s| !s.is_empty())
                    && question["type"] == "choice"
                    && question["instructions"]
                        .as_str()
                        .is_some_and(|s| !s.is_empty())
                    && criteria.is_some_and(|c| {
                        (2..=255).contains(&c.len()) && c.values().all(Value::is_string)
                    });
                if !valid {
                    let detail = json!({"detail": "validation error"});
                    return (StatusCode::UNPROCESSABLE_ENTITY, Json(detail));
                }
                let text = req["state"].as_str().unwrap_or("");
                let names: Vec<&String> = criteria.unwrap().keys().collect();
                let choice = names
                    .iter()
                    .find(|n| text.contains(n.as_str()))
                    .map_or("other", |n| n.as_str());
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a prompt has at most 255 labels"
                )]
                let rest = 0.2 / (names.len() - 1) as f64;
                let probabilities: serde_json::Map<String, Value> = names
                    .iter()
                    .map(|n| ((*n).clone(), json!(if *n == choice { 0.8 } else { rest })))
                    .collect();
                let answer = json!({"type": "choice", "choice": choice,
                    "confidence": 0.8 - rest, "probabilities": probabilities});
                let usage = json!({"input_tokens": 89, "output_tokens": 1});
                let v = json!({"model": "glide", "answers": {"label": answer}, "usage": usage,
                               "token_usage": 90});
                (StatusCode::OK, Json(v))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

/// Fastino's GLiDE serves the same `POST /v1/systemone` as Decider, so it is a `systemone`
/// System One model: Fastino's root URL, model `fastino/glide` and its key from `api_key_env`.
/// Fastino's terms forbid training competing models on its outputs, so it never judges.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fastino_glide_answers_but_never_judges() {
    // A variable only this test reads.
    const KEY_ENV: &str = "GATEWAY_TEST_FASTINO_API_KEY";
    std::env::set_var(KEY_ENV, "fast_sk_test");
    let dir = tempfile::tempdir().unwrap();
    let bodies = Arc::new(Mutex::new(vec![]));
    let fastino = mock_fastino("fast_sk_test", bodies.clone()).await;
    let (url, _state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let put = |path: &str, body: Value| {
        http.put(format!("{url}/admin/api/{path}"))
            .bearer_auth(ADMIN)
            .json(&body)
            .send()
    };
    let glide = json!({"kind": "systemone", "base_url": format!("http://{fastino}/"),
                       "model": "fastino/glide", "api_key_env": KEY_ENV});
    let r = put("system-one-models/glide", glide.clone()).await.unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());

    let spec = json!({"instructions": "What does the customer want?",
                      "labels": ["refund", {"name": "cancel", "description": "stop the order"}, "other"]});
    let classify = |input: &'static str, model: &'static str| {
        let (http, url, spec) = (http.clone(), url.clone(), spec.clone());
        async move {
            http.post(format!("{url}/v1/classify"))
                .json(&json!({"prompt": spec, "input": input, "model": model}))
                .send()
                .await
                .unwrap()
        }
    };
    let v: Value = classify("refund please", "glide")
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(
        (
            v["source"].as_str(),
            v["model"].as_str(),
            v["label"].as_str()
        ),
        (Some("system_one"), Some("glide"), Some("refund")),
        "{v}"
    );
    assert_eq!(
        v["confidence"], 0.8,
        "the answer's probability, not the margin: {v}"
    );
    // Every label has a string description, empty when the prompt gives none.
    let body = bodies.lock().unwrap()[0].clone();
    let sent: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        (sent["model"].as_str(), sent["state"].as_str()),
        (Some("fastino/glide"), Some("refund please"))
    );
    assert!(
        body.contains(r#""criteria":{"refund":"","cancel":"stop the order","other":""}"#),
        "{body}"
    );

    // Without its key, Fastino answers 401, and the caller gets the gateway's 502.
    let mut keyless = glide;
    keyless["api_key_env"] = Value::Null;
    let r = put("system-one-models/keyless", keyless).await.unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    assert_eq!(classify("refund please", "keyless").await.status(), 502);

    // Acknowledging a license doesn't make it a judge.
    let judge = json!({"base_url": "https://api.fastino.ai", "model": "fastino/glide",
                       "mode": "systemone", "license_ack": true, "api_key_env": "FASTINO_API_KEY"});
    let r = put("judges/glide", judge).await.unwrap();
    assert_eq!(r.status(), 400);
    assert!(r.text().await.unwrap().contains("§4.A"));
}

/// Every System One answer is costed at its model's pricing (TypeSafe Jev's rates unless set),
/// with the tokens the model reports, and the usage page shows the cost and the cost per answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn system_one_answers_are_costed_at_each_models_pricing() {
    // A variable only this test reads.
    const KEY_ENV: &str = "GATEWAY_TEST_PRICED_API_KEY";
    std::env::set_var(KEY_ENV, "sk_priced");
    let dir = tempfile::tempdir().unwrap();
    // It reports 89 input tokens and 1 output token for every call.
    let server = mock_fastino("sk_priced", Arc::new(Mutex::new(vec![]))).await;
    let (url, _state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let put = |path: &str, body: Value| {
        http.put(format!("{url}/admin/api/{path}"))
            .bearer_auth(ADMIN)
            .json(&body)
            .send()
    };
    let model = json!({"kind": "systemone", "base_url": format!("http://{server}/"),
                       "model": "fastino/glide", "api_key_env": KEY_ENV});
    assert_eq!(
        put("system-one-models/jev-priced", model.clone())
            .await
            .unwrap()
            .status(),
        200
    );
    let mut priced = model.clone();
    priced["pricing"] = json!({"usd_per_call": 0.01, "usd_per_million_input_tokens": 1.0,
                               "usd_per_million_output_tokens": 2.0});
    assert_eq!(
        put("system-one-models/priced", priced.clone())
            .await
            .unwrap()
            .status(),
        200
    );
    priced["pricing"]["usd_per_call"] = json!(-1);
    let r = put("system-one-models/negative", priced).await.unwrap();
    assert_eq!(r.status(), 400);
    assert!(r.text().await.unwrap().contains("pricing"));
    let r = put("settings", json!({"default_system_one": "jev-priced"}))
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    let spec = json!({"instructions": "What does the customer want?", "labels": ["refund", "cancel", "other"]});
    for body in [
        json!({"prompt": spec, "input": "refund please"}),
        json!({"prompt": spec, "input": "cancel it", "model": "priced"}),
    ] {
        let r = http
            .post(format!("{url}/v1/classify"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    }
    let jev = 89.0 * 0.042 / 1e6;
    let custom = 0.01 + 89.0 * 1.0 / 1e6 + 2.0 / 1e6;
    let close = |v: &Value, want: f64| (v.as_f64().unwrap() - want).abs() < 1e-12;

    let prompts: Value = http
        .get(format!("{url}/api/prompts"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let p = &prompts[0];
    assert!(close(&p["system_one_cost_usd_7_days"], jev + custom), "{p}");
    assert!(
        close(
            &p["system_one_cost_per_answer_usd"],
            f64::midpoint(jev, custom)
        ),
        "{p}"
    );
    assert!(close(&p["student_savings_usd_7_days"], 0.0), "{p}");
    let today = p["daily"].as_array().unwrap().last().unwrap();
    assert!(close(&today["system_one_usd"], jev + custom), "{today}");
    assert_eq!(
        p["system_one_pricing"],
        json!({"usd_per_call": 0.0, "usd_per_million_input_tokens": 0.042,
               "usd_per_million_output_tokens": 0.0})
    );
    let summary: Value = http
        .get(format!("{url}/api/summary"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        close(&summary["system_one_cost_usd_7_days"], jev + custom),
        "{summary}"
    );
}

/// The cache (issue #95): System One's answer for a text comes back without a call, counts as
/// the cache's, and goes when the cache is emptied, its model changes or it is turned off.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    clippy::too_many_lines,
    reason = "hits, usage, admin numbers, emptying and turning it off, in order"
)]
async fn the_cache_answers_what_system_one_answered_before() {
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let s1 = mock_system_one(calls.clone()).await;
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let put = |path: &str, body: Value| {
        http.put(format!("{url}/admin/api/{path}"))
            .bearer_auth(ADMIN)
            .json(&body)
            .send()
    };
    let model = json!({"kind": "openai_chat", "base_url": format!("http://{s1}/v1"), "model": "m"});
    assert_eq!(
        put("system-one-models/s1", model.clone())
            .await
            .unwrap()
            .status(),
        200
    );
    let r = put(
        "settings",
        json!({"default_system_one": "s1", "cache": {"enabled": true, "policy": "fifo"}}),
    )
    .await
    .unwrap();
    assert_eq!(r.status(), 400, "an unknown policy is refused");
    let r = put(
        "settings",
        json!({"default_system_one": "s1", "cache": {"enabled": true}}),
    )
    .await
    .unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let classify = |input: &str| {
        let (http, url, input) = (http.clone(), url.clone(), input.to_string());
        async move {
            let v: Value = http
                .post(format!("{url}/v1/classify"))
                .json(&json!({"prompt": prompt(), "input": input}))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            v
        }
    };
    let first = classify("cancel my order").await;
    assert_eq!(
        (first["source"].as_str(), first["label"].as_str()),
        (Some("system_one"), Some("cancel"))
    );
    let again = classify("  cancel my order\n").await;
    assert_eq!(
        (
            again["source"].as_str(),
            again["model"].as_str(),
            again["label"].as_str()
        ),
        (Some("cache"), Some("s1"), Some("cancel")),
        "{again}"
    );
    assert_eq!(again["confidence"], first["confidence"]);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the cache answered without a call"
    );

    let numbers: Value = http
        .get(format!("{url}/admin/api/cache"))
        .bearer_auth(ADMIN)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        (numbers["entries"].as_u64(), numbers["hits"].as_u64()),
        (Some(1), Some(1)),
        "{numbers}"
    );
    let prompts: Value = http
        .get(format!("{url}/api/prompts"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let p = &prompts[0];
    assert_eq!(
        (p["last_7_days"].as_u64(), p["cache_7_days"].as_u64()),
        (Some(2), Some(1)),
        "{p}"
    );
    assert!(p["model_latencies_7_days"]
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r["model"] == "cache"));
    // An answer from the cache costs nothing, and saves what a System One answer costs.
    let cost = p["system_one_cost_per_answer_usd"].as_f64().unwrap();
    assert!(
        (p["cache_savings_usd_7_days"].as_f64().unwrap() - cost).abs() < 1e-15,
        "{p}"
    );

    // A changed System One model may answer differently: its answers go.
    assert_eq!(
        put("system-one-models/s1", model).await.unwrap().status(),
        200
    );
    assert_eq!(classify("cancel my order").await["source"], "system_one");
    assert_eq!(classify("cancel my order").await["source"], "cache");
    let r = http
        .delete(format!("{url}/admin/api/cache"))
        .bearer_auth(ADMIN)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(state.answers.stats()["entries"], 0);
    assert_eq!(classify("cancel my order").await["source"], "system_one");
    // Off, every request asks System One.
    let r = put("settings", json!({"cache": {"enabled": false}}))
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(classify("cancel my order").await["source"], "system_one");
    assert_eq!(classify("cancel my order").await["source"], "system_one");
    assert_eq!(calls.load(Ordering::SeqCst), 5);
}

/// After the student, the cache answers only what no confident student does; before it, the
/// cache's answer comes first. A student's own answers are never cached.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_cache_goes_after_the_student_or_before_it() {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        return;
    }
    let model_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audit-student");
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let s1 = mock_system_one(calls.clone()).await;
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let put = |path: &str, body: Value| {
        http.put(format!("{url}/admin/api/{path}"))
            .bearer_auth(ADMIN)
            .json(&body)
            .send()
    };
    let model = json!({"kind": "openai_chat", "base_url": format!("http://{s1}/v1"), "model": "m"});
    assert_eq!(
        put("system-one-models/s1", model).await.unwrap().status(),
        200
    );
    // Training needs a judge; with audits off, it is never asked.
    let judge =
        json!({"base_url": format!("http://{s1}/v1"), "model": "trainer", "license_ack": true});
    assert_eq!(put("judges/trainer", judge).await.unwrap().status(), 200);
    let settings = |position: &str| {
        json!({"default_system_one": "s1", "default_judge": "trainer", "monitor": {"enabled": false},
               "cache": {"enabled": true, "position": position}})
    };
    assert_eq!(
        put("settings", settings("after_student"))
            .await
            .unwrap()
            .status(),
        200
    );
    let classify = |input: &str| {
        let (http, url, input) = (http.clone(), url.clone(), input.to_string());
        async move {
            let v: Value = http
                .post(format!("{url}/v1/classify"))
                .json(&json!({"prompt": prompt(), "input": input}))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            (
                v["source"].as_str().unwrap().to_string(),
                v["label"].as_str().unwrap().to_string(),
            )
        }
    };
    let answer = |source: &str, label: &str| (source.to_string(), label.to_string());
    assert_eq!(classify("cancel it").await, answer("system_one", "cancel"));
    // Without a student, the cache is the alternative to one.
    assert_eq!(classify("cancel it").await, answer("cache", "cancel"));
    let pid = state.prompts.read().unwrap().keys().next().unwrap().clone();
    // A student that answers "refund", always confidently (threshold 0).
    let p = state.prompt(&pid).unwrap();
    state.queue_job(&p, "manual").unwrap();
    let spec: Value = http
        .post(format!("{url}/internal/jobs/claim"))
        .bearer_auth(WORKER)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    for name in ["model.onnx", "tokenizer.json", "manifest.json"] {
        state
            .store
            .put(
                &format!("{}/{name}", spec["models_prefix"].as_str().unwrap()),
                std::fs::read(model_dir.join(name)).unwrap().into(),
            )
            .await
            .unwrap();
    }
    let r = http.post(format!("{url}/internal/jobs/{}/complete", spec["job_id"].as_str().unwrap())).bearer_auth(WORKER)
        .json(&json!({"ok":true,"threshold":0.0,"metrics":{"inputs":10,"labeled":10,"cascade":{"target_accuracy":0.95}}})).send().await.unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());

    assert_eq!(classify("cancel it").await, answer("student", "refund"));
    assert_eq!(
        put("settings", settings("before_student"))
            .await
            .unwrap()
            .status(),
        200
    );
    assert_eq!(classify("cancel it").await, answer("cache", "cancel"));
    assert_eq!(classify("something new").await, answer("student", "refund"));
    assert_eq!(classify("something new").await, answer("student", "refund"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// A judge that answers like `mock_system_one` and names itself `version` (what its alias
/// resolves to now), as OpenAI-compatible servers do in `model`.
async fn mock_versioned_judge(version: Arc<Mutex<String>>) -> SocketAddr {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(body): Json<Value>| {
            let version = version.clone();
            async move {
                let text = body["messages"][1]["content"].as_str().unwrap_or("").to_lowercase();
                let text = text.split("\"\"\"").nth(1).unwrap_or("");
                let letter = if text.contains("refund") { "A" } else if text.contains("cancel") { "B" } else { "C" };
                let model = version.lock().unwrap().clone();
                Json(json!({"model": model, "choices": [{"message": {"content": letter},
                    "logprobs": {"content": [{"token": letter, "logprob": 0.0,
                        "top_logprobs": [{"token": letter, "logprob": -0.05}, {"token": "C", "logprob": -3.0}]}]}}]}))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

/// Issue #87: when the judge's alias resolves to another version, the deployed student's
/// evidence was against a training target that is gone, so it is tested again, in shadow.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    clippy::too_many_lines,
    reason = "deploy, go live, move the judge's alias and go live again, end to end"
)]
async fn a_new_judge_version_tests_the_live_student_again() {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        return;
    }
    let model_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audit-student");
    let dir = tempfile::tempdir().unwrap();
    let s1 = mock_system_one(Arc::new(AtomicUsize::new(0))).await;
    let version = Arc::new(Mutex::new("teacher-2026-09".to_string()));
    let judge = mock_versioned_judge(version.clone()).await;
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let put = |path: &str, body: Value| {
        http.put(format!("{url}/admin/api/{path}"))
            .bearer_auth(ADMIN)
            .json(&body)
            .send()
    };
    let s1 =
        json!({"kind": "openai_chat", "base_url": format!("http://{s1}/v1"), "model": "reference"});
    assert_eq!(
        put("system-one-models/reference", s1)
            .await
            .unwrap()
            .status(),
        200
    );
    let teacher = json!({"base_url": format!("http://{judge}/v1"), "model": "teacher-latest", "license_ack": true});
    assert_eq!(put("judges/teacher", teacher).await.unwrap().status(), 200);
    let r = put(
        "settings",
        json!({"default_system_one": "reference", "default_judge": "teacher",
        "monitor": {"min_samples": 2, "window": 2, "sample_rate": 1.0, "tolerance": 0.0}}),
    )
    .await
    .unwrap();
    assert_eq!(r.status(), 200);
    let classify = |input: String| {
        let (http, url) = (http.clone(), url.clone());
        async move {
            let v: Value = http
                .post(format!("{url}/v1/classify"))
                .json(&json!({"prompt": prompt(), "input": input}))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            v
        }
    };
    let pid = classify("refund 0".into()).await["prompt_id"]
        .as_str()
        .unwrap()
        .to_string();
    let p = state.prompt(&pid).unwrap();
    state.queue_job(&p, "manual").unwrap();
    let spec: Value = http
        .post(format!("{url}/internal/jobs/claim"))
        .bearer_auth(WORKER)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    for name in ["model.onnx", "tokenizer.json", "manifest.json"] {
        state
            .store
            .put(
                &format!("{}/{name}", spec["models_prefix"].as_str().unwrap()),
                std::fs::read(model_dir.join(name)).unwrap().into(),
            )
            .await
            .unwrap();
    }
    // The worker trained it on what the judge answered as then.
    let r = http.post(format!("{url}/internal/jobs/{}/complete", spec["job_id"].as_str().unwrap())).bearer_auth(WORKER)
        .json(&json!({"ok": true, "threshold": 0.0, "metrics": {"inputs": 10, "labeled": 10,
            "judge": "teacher-latest", "judge_version": "teacher-2026-09", "cascade": {"target_accuracy": 0.95}}}))
        .send().await.unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let monitor = || {
        p.deployed()
            .map(|d| d.monitor.lock().unwrap().clone())
            .unwrap()
    };
    assert_eq!(monitor().judge_version.as_deref(), Some("teacher-2026-09"));

    // Two agreeing audits against the same version: live.
    for i in 1..=2 {
        classify(format!("refund {i}")).await;
        wait_until(|| monitor().audits >= i).await;
    }
    wait_until(|| monitor().phase == "live").await;
    assert_eq!(
        (monitor().phase.as_str(), p.status()),
        ("live", "deployed".to_string())
    );
    assert_eq!(classify("refund 3".into()).await["source"], "student");
    wait_until(|| monitor().audits >= 3).await;

    // The alias moves: the next audit starts the test again, and System One answers meanwhile.
    *version.lock().unwrap() = "teacher-2026-10".into();
    classify("refund 4".into()).await;
    wait_until(|| monitor().phase == "shadow").await;
    let m = monitor();
    assert_eq!(
        (m.phase.as_str(), p.status()),
        ("shadow", "shadow".to_string())
    );
    assert_eq!(m.judge_version.as_deref(), Some("teacher-2026-10"));
    let change = m.judge_change.unwrap();
    assert_eq!(
        (change.from.as_str(), change.to.as_str()),
        ("teacher-2026-09", "teacher-2026-10")
    );
    assert_eq!(
        m.audits, 1,
        "the audit that saw the new version is the first evidence against it"
    );
    assert_eq!(classify("refund 5".into()).await["source"], "system_one");
    // Agreeing with the new version, it goes live again.
    wait_until(|| monitor().phase == "live").await;
    assert_eq!(monitor().phase, "live");
    let prompts: Value = http
        .get(format!("{url}/api/prompts"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        prompts[0]["monitor"]["judge_change"]["to"],
        "teacher-2026-10"
    );
}

/// Issue #96: a prompt with a student per language asks the worker for them, and the language
/// router sends a text the router is sure of to its language's student, the rest to the prompt's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    clippy::too_many_lines,
    reason = "turn it on, train, publish both students and route, end to end"
)]
async fn the_language_router_picks_each_languages_student() {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        return;
    }
    let model_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audit-student");
    let dir = tempfile::tempdir().unwrap();
    let s1 = mock_system_one(Arc::new(AtomicUsize::new(0))).await;
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let put = |path: &str, body: Value| {
        http.put(format!("{url}/admin/api/{path}"))
            .bearer_auth(ADMIN)
            .json(&body)
            .send()
    };
    let model = json!({"kind": "openai_chat", "base_url": format!("http://{s1}/v1"), "model": "m"});
    assert_eq!(
        put("system-one-models/s1", model).await.unwrap().status(),
        200
    );
    let judge =
        json!({"base_url": format!("http://{s1}/v1"), "model": "trainer", "license_ack": true});
    assert_eq!(put("judges/trainer", judge).await.unwrap().status(), 200);
    let r = put("settings", json!({"default_system_one": "s1", "default_judge": "trainer",
        "monitor": {"enabled": false}, "per_language": {"min_confidence": 0.0, "min_examples": 10}}))
    .await
    .unwrap();
    assert_eq!(r.status(), 400, "a confidence of 0 is refused");
    let r = put("settings", json!({"default_system_one": "s1", "default_judge": "trainer",
        "monitor": {"enabled": false}, "per_language": {"min_confidence": 0.7, "min_examples": 50}}))
    .await
    .unwrap();
    assert_eq!(r.status(), 200);
    let classify = |input: &str| {
        let (http, url, input) = (http.clone(), url.clone(), input.to_string());
        async move {
            let v: Value = http
                .post(format!("{url}/v1/classify"))
                .json(&json!({"prompt": prompt(), "input": input}))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            v
        }
    };
    let pid = classify("refund please").await["prompt_id"]
        .as_str()
        .unwrap()
        .to_string();
    let r = put(&format!("prompts/{pid}"), json!({"per_language": true}))
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let p = state.prompt(&pid).unwrap();
    state.queue_job(&p, "manual").unwrap();
    let spec: Value = http
        .post(format!("{url}/internal/jobs/claim"))
        .bearer_auth(WORKER)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        spec["per_language"],
        json!({"enabled": true, "min_confidence": 0.7, "min_examples": 50,
               "base_model": "jhu-clsp/ettin-encoder-17m"})
    );
    // The worker publishes the prompt's student, a Portuguese one, and languages.json.
    let prefix = spec["models_prefix"].as_str().unwrap();
    for name in ["model.onnx", "tokenizer.json", "manifest.json"] {
        let data = std::fs::read(model_dir.join(name)).unwrap();
        state
            .store
            .put(&format!("{prefix}/{name}"), data.clone().into())
            .await
            .unwrap();
        state
            .store
            .put(&format!("{prefix}/lang/por/{name}"), data.into())
            .await
            .unwrap();
    }
    let languages =
        json!({"min_confidence": 0.7, "languages": {"por": {"threshold": 0.0, "examples": 60}}});
    state
        .store
        .put(
            &format!("{prefix}/languages.json"),
            languages.to_string().into_bytes().into(),
        )
        .await
        .unwrap();
    let r = http.post(format!("{url}/internal/jobs/{}/complete", spec["job_id"].as_str().unwrap())).bearer_auth(WORKER)
        .json(&json!({"ok": true, "threshold": 0.0, "metrics": {"inputs": 10, "labeled": 10,
            "languages": {"por": {"threshold": 0.0, "examples": 60}}, "cascade": {"target_accuracy": 0.95}}}))
        .send().await.unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    assert_eq!(p.deployed().unwrap().languages.by_language.len(), 1);

    let pt =
        classify("Quero cancelar a minha assinatura e receber o dinheiro de volta, por favor.")
            .await;
    assert_eq!(
        (
            pt["source"].as_str(),
            pt["model"].as_str(),
            pt["language"].as_str()
        ),
        (Some("student"), Some("audit-fixture@por"), Some("por")),
        "{pt}"
    );
    let en =
        classify("I would like to cancel my subscription and get my money back, please.").await;
    assert_eq!(
        (en["source"].as_str(), en["model"].as_str()),
        (Some("student"), Some("audit-fixture")),
        "{en}"
    );
    assert!(en.get("language").is_none());
    let prompts: Value = http
        .get(format!("{url}/api/prompts"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let models: Vec<&str> = prompts[0]["model_latencies_7_days"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|r| r["model"].as_str())
        .collect();
    assert!(
        models.contains(&"student:audit-fixture@por") && models.contains(&"student:audit-fixture"),
        "{models:?}"
    );
    // A restart loads both students again, from the cache folder.
    drop(state);
    let (_url, restarted) = start_gateway(dir.path(), None).await;
    let p = restarted.prompt(&pid).unwrap();
    assert_eq!(p.deployed().unwrap().languages.by_language.len(), 1);
}

/// With the cache between System One and the model it escalates to, System One answers every time,
/// and an escalation the stronger model answered before comes from the cache: only that call is
/// saved. System One's own answers are not kept.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_cache_can_sit_between_system_one_and_the_model_it_escalates_to() {
    let dir = tempfile::tempdir().unwrap();
    let llm_calls = Arc::new(AtomicUsize::new(0));
    let llm = mock_system_one(llm_calls.clone()).await;
    let weak = mock_weak_system_one().await;
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let put = |path: &str, body: Value| {
        http.put(format!("{url}/admin/api/{path}"))
            .bearer_auth(ADMIN)
            .json(&body)
            .send()
    };
    let llm_model =
        json!({"kind": "openai_chat", "base_url": format!("http://{llm}/v1"), "model": "big-llm"});
    assert_eq!(
        put("system-one-models/llm", llm_model)
            .await
            .unwrap()
            .status(),
        200
    );
    let weak_model = json!({"kind": "decision_http", "base_url": format!("http://{weak}/decide"), "model": "jev",
                            "escalate_to": "llm", "escalate_below": 0.6});
    assert_eq!(
        put("system-one-models/weak", weak_model)
            .await
            .unwrap()
            .status(),
        200
    );
    let r = put(
        "settings",
        json!({"default_system_one": "weak",
        "cache": {"enabled": true, "position": "before_escalation"}}),
    )
    .await
    .unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let classify = |input: &'static str| {
        let (http, url) = (http.clone(), url.clone());
        async move {
            let v: Value = http
                .post(format!("{url}/v1/classify"))
                .json(&json!({"prompt": prompt(), "input": input}))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            v
        }
    };
    let first = classify("maybe cancel it?").await;
    assert_eq!(
        (
            first["source"].as_str(),
            first["model"].as_str(),
            first["label"].as_str(),
            first["escalated_from"].as_str()
        ),
        (
            Some("system_one"),
            Some("llm"),
            Some("cancel"),
            Some("weak")
        ),
        "{first}"
    );
    let again = classify("maybe cancel it?").await;
    assert_eq!(
        (
            again["source"].as_str(),
            again["model"].as_str(),
            again["label"].as_str(),
            again["escalated_from"].as_str()
        ),
        (Some("cache"), Some("llm"), Some("cancel"), Some("weak")),
        "{again}"
    );
    assert_eq!(
        llm_calls.load(Ordering::SeqCst),
        1,
        "the escalation came from the cache"
    );
    // System One's own confident answers aren't kept: the cache holds the one escalation.
    assert_eq!(classify("refund please").await["source"], "system_one");
    assert_eq!(classify("refund please").await["source"], "system_one");
    assert_eq!(state.answers.stats()["entries"], 1);
}

/// `POST /admin/api/labels`: reference labels reach the store for the prompt's next job, under
/// the admin token, and labels the prompt doesn't have or unknown splits are refused.
#[tokio::test]
async fn reference_labels_are_stored_for_the_worker() {
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let body = json!({"prompt": prompt(), "name": "dataset-1", "labels": [
        {"text": "please refund me", "label": "refund", "split": "train"},
        {"text": "where is my parcel", "label": "other", "split": "audit"},
        {"text": "cancel it", "label": "cancel"},
        {"text": "  ", "label": "refund"}]});
    let post = |b: &Value, token: &str| {
        http.post(format!("{url}/admin/api/labels"))
            .bearer_auth(token)
            .json(b)
            .send()
    };
    assert_eq!(post(&body, WORKER).await.unwrap().status(), 401);
    let r = post(&body, ADMIN).await.unwrap();
    assert_eq!(r.status(), 200);
    let r: Value = r.json().await.unwrap();
    assert_eq!(
        (&r["train"], &r["audit"], &r["unsplit"]),
        (&json!(1), &json!(1), &json!(1))
    );
    let pid = r["prompt_id"].as_str().unwrap().to_string();
    // The prompt is registered, as its first request would have.
    assert!(state.prompt(&pid).is_some());
    let stored = state
        .store
        .get(&format!("human_labels/{pid}/dataset-1.jsonl"))
        .await
        .unwrap();
    let lines: Vec<Value> = String::from_utf8(stored.to_vec())
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 3);
    assert_eq!(
        lines[1],
        json!({"text": "where is my parcel", "label": "other", "split": "audit"})
    );
    assert!(lines[2].get("split").is_none());
    for (bad, why) in [
        (
            json!({"text": "x", "label": "no-such-label"}),
            "labels the prompt doesn't have",
        ),
        (
            json!({"text": "x", "label": "refund", "split": "test"}),
            "split 'test'",
        ),
    ] {
        let mut b = body.clone();
        b["labels"] = json!([bad]);
        let r = post(&b, ADMIN).await.unwrap();
        assert_eq!(r.status(), 400);
        assert!(r.text().await.unwrap().contains(why));
    }
    let mut b = body.clone();
    b["name"] = json!("../escape");
    assert_eq!(post(&b, ADMIN).await.unwrap().status(), 400);
}
