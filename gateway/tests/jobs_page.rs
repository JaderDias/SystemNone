//! Usage training history and `/api/jobs`: every job's progress and its prompt, as the
//! usage page names it, for whoever may view usage, without the admin token, and nothing of a
//! job's spec or its message.

use std::sync::Arc;

use serde_json::{json, Value};
use system_none_gateway::{api, AppState, GatewayConfig};

const ADMIN: &str = "admin-secret";
const WORKER: &str = "worker-secret";
const VIEWER: &str = "viewer-secret";

/// Everything `/api/jobs` says about a job.
const FIELDS: [&str; 13] = [
    "id",
    "prompt_id",
    "prompt",
    "status",
    "phase",
    "done",
    "total",
    "created_at",
    "started_at",
    "finished_at",
    "last_stage",
    "stage_done",
    "stage_total",
];

#[tokio::test]
async fn usage_resolves_auto_and_names_the_configured_models() {
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let response = http
        .put(format!("{url}/admin/api/system-one-models/typesafe"))
        .bearer_auth(ADMIN)
        .json(
            &json!({"kind": "openai_decisions", "base_url": "https://private.example/v1",
            "model": "jev-latest", "api_key_env": "PRIVATE_KEY"}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let response = http.put(format!("{url}/admin/api/judges/decider"))
        .bearer_auth(ADMIN)
        .json(&json!({"base_url": "https://private-judge.example", "mode": "systemone",
            "model": "vllm-sr/Decision-2.0-Kai-0.6B", "api_key_env": "PRIVATE_JUDGE_KEY", "license_ack": true}))
        .send().await.unwrap();
    assert_eq!(response.status(), 200);
    for calibration in [
        json!({"method": "wilson", "confidence_level": 0.95}),
        json!({"method": "clopper_pearson", "confidence_level": 0}),
        json!({"method": "clopper_pearson", "confidence_level": 1}),
    ] {
        let rejected = http
            .put(format!("{url}/admin/api/settings"))
            .bearer_auth(ADMIN)
            .json(&json!({"calibration": calibration}))
            .send()
            .await
            .unwrap();
        assert_eq!(rejected.status(), 400);
    }
    let response = http
        .put(format!("{url}/admin/api/settings"))
        .bearer_auth(ADMIN)
        .json(&json!({"default_base_model": "auto", "default_system_one": "typesafe", "default_judge": "decider", "calibration": {"method": "clopper_pearson", "confidence_level": 0.99}}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let prompt = state
        .resolve_prompt(
            serde_json::from_value(json!({
                "instructions": "Choose a label", "labels": ["yes", "no"]
            }))
            .unwrap(),
        )
        .await
        .unwrap();
    let job = state.queue_job(&prompt, "manual").unwrap().unwrap();
    assert_eq!(
        state.job_spec(&job).unwrap()["calibration"],
        json!({"method": "clopper_pearson", "confidence_level": 0.99})
    );
    assert_eq!(
        state.db.settings().unwrap().calibration.method,
        "clopper_pearson"
    );
    let response = http.get(format!("{url}/api/prompts")).send().await.unwrap();
    let body = response.text().await.unwrap();
    let views: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(views[0]["base_model"], "auto");
    assert_eq!(
        views[0]["resolved_base_model"],
        "jhu-clsp/ettin-encoder-17m"
    );
    assert_eq!(views[0]["system_one_model"], "typesafe");
    assert_eq!(views[0]["system_one_model_id"], "jev-latest");
    assert_eq!(views[0]["judge"], "decider");
    assert_eq!(views[0]["judge_model_id"], "vllm-sr/Decision-2.0-Kai-0.6B");
    assert!(!body.contains("private-judge.example") && !body.contains("PRIVATE_JUDGE_KEY"));
    assert!(!body.contains("private.example") && !body.contains("PRIVATE_KEY"));
    let response = http
        .put(format!("{url}/admin/api/prompts/{}", prompt.id))
        .bearer_auth(ADMIN)
        .json(&json!({"base_model": "org/custom-student"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let views: Value = http
        .get(format!("{url}/api/prompts"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(views[0]["resolved_base_model"], "org/custom-student");
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

/// `/api/jobs` without a token: its body, and the jobs in it (newest first), each with exactly
/// `FIELDS`.
async fn public_jobs(http: &reqwest::Client, url: &str) -> (String, Vec<Value>) {
    let r = http.get(format!("{url}/api/jobs")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let body = r.text().await.unwrap();
    let jobs: Vec<Value> = serde_json::from_str(&body).unwrap();
    for j in &jobs {
        let mut keys: Vec<&str> = j.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        let mut fields = FIELDS.to_vec();
        fields.sort_unstable();
        assert_eq!(keys, fields, "{j}");
    }
    (body, jobs)
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "one job's life, stage by stage, as the page sees it"
)]
async fn training_progress_needs_no_admin_token_and_shows_only_progress_and_prompts() {
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = start_gateway(dir.path(), None).await;
    let http = reqwest::Client::new();
    let admin = |req: reqwest::RequestBuilder| req.bearer_auth(ADMIN);
    let worker = |path: String, body: Value| {
        http.post(format!("{url}/internal/jobs/{path}"))
            .bearer_auth(WORKER)
            .json(&body)
            .send()
    };
    // A judge and a base model that the page must not name, and a prompt it names as the usage
    // page does: by the first line of its instructions and its labels' names, without their
    // descriptions. One attempt: the failure below is the job's end, with no retry queued after it.
    let judge = json!({"base_url": "http://judge.internal.example:9/v1", "model": "private-judge-7b",
                       "api_key_env": "PRIVATE_JUDGE_KEY", "license_ack": true});
    let r = admin(http.put(format!("{url}/admin/api/judges/private")))
        .json(&judge)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let r = admin(http.put(format!("{url}/admin/api/settings")))
        .json(
            &json!({"default_judge": "private", "default_base_model": "org/secret-base",
                      "train": {"epochs": 7}, "max_attempts": 1}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let spec = json!({"instructions": "Is this ticket about the Zanzibar merger?\nAnswer no if unsure.",
                      "labels": ["yes", {"name": "no", "description": "anything else"}]});
    let p = state
        .resolve_prompt(serde_json::from_value(spec).unwrap())
        .await
        .unwrap();
    let named =
        json!({"title": "Is this ticket about the Zanzibar merger?", "labels": ["yes", "no"]});
    let hidden = [
        "private-judge-7b",
        "judge.internal.example",
        "PRIVATE_JUDGE_KEY",
        "secret-base",
        "epochs",
        "anything else",
        "manual", // the spec's reason
        "judging ",
        "training ",
    ];
    let hides = |body: &str| {
        for h in hidden {
            assert!(!body.contains(h), "{h} in {body}");
        }
    };

    let (body, jobs) = public_jobs(&http, &url).await;
    assert_eq!(body, "[]");
    assert_eq!(jobs, Vec::<Value>::new());

    let job = state.queue_job(&p, "manual").unwrap().unwrap().id;
    let (body, jobs) = public_jobs(&http, &url).await;
    hides(&body);
    assert_eq!(
        (&jobs[0]["id"], &jobs[0]["prompt_id"], &jobs[0]["prompt"]),
        (&json!(job), &json!(p.id), &named)
    );
    assert_eq!(
        (
            &jobs[0]["status"],
            &jobs[0]["phase"],
            &jobs[0]["started_at"]
        ),
        (&json!("queued"), &json!("queued"), &Value::Null)
    );

    let r = http
        .post(format!("{url}/internal/jobs/claim"))
        .bearer_auth(WORKER)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let (body, jobs) = public_jobs(&http, &url).await;
    hides(&body);
    assert_eq!(
        (&jobs[0]["status"], &jobs[0]["phase"]),
        (&json!("running"), &json!("starting"))
    );
    assert!(jobs[0]["started_at"].is_string());

    // The worker's messages become a phase; their counts go through.
    let judging =
        json!({"message": "judging 20 inputs with private-judge-7b", "done": 5, "total": 20});
    assert_eq!(
        worker(format!("{job}/progress"), judging)
            .await
            .unwrap()
            .status(),
        204
    );
    let (body, jobs) = public_jobs(&http, &url).await;
    hides(&body);
    assert_eq!(
        (&jobs[0]["phase"], &jobs[0]["done"], &jobs[0]["total"]),
        (&json!("judging"), &json!(5), &json!(20)),
        "{body}"
    );
    let training = json!({"message": "training a student on org/secret-base with 20 judged inputs",
                          "done": 3, "total": 10});
    assert_eq!(
        worker(format!("{job}/progress"), training)
            .await
            .unwrap()
            .status(),
        204
    );
    let (body, jobs) = public_jobs(&http, &url).await;
    hides(&body);
    assert_eq!(
        (&jobs[0]["phase"], &jobs[0]["done"], &jobs[0]["total"]),
        (&json!("training"), &json!(3), &json!(10)),
        "{body}"
    );

    // Stage transitions clear the previous counts and never expose the worker's raw message.
    for phase in ["calibrating", "exporting", "validating"] {
        let response = worker(
            format!("{job}/progress"),
            json!({"message": format!("{phase} org/secret-base")}),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), 204);
        let (body, jobs) = public_jobs(&http, &url).await;
        hides(&body);
        assert_eq!(jobs[0]["phase"], phase);
        assert!(jobs[0]["done"].is_null() && jobs[0]["total"].is_null());
        let prompts: Value = http
            .get(format!("{url}/api/prompts"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(prompts[0]["training_job"]["phase"], phase);
        assert!(!prompts[0]["training_job"]
            .to_string()
            .contains("secret-base"));
    }

    // A failure's message can quote an upstream error body: only the status shows.
    let upstream = "HTTPStatusError: Unauthorized from http://judge.internal.example:9/v1: \
                    {\"error\": \"invalid api key sk-live-0123\"}";
    let r = worker(
        format!("{job}/complete"),
        json!({"ok": false, "message": upstream}),
    )
    .await
    .unwrap();
    assert_eq!(r.status(), 200);
    let (body, jobs) = public_jobs(&http, &url).await;
    hides(&body);
    for h in [
        "sk-live",
        "invalid api key",
        "Unauthorized",
        "HTTPStatusError",
    ] {
        assert!(!body.contains(h), "{h} in {body}");
    }
    assert_eq!(
        (
            &jobs[0]["status"],
            &jobs[0]["phase"],
            &jobs[0]["done"],
            &jobs[0]["total"]
        ),
        (
            &json!("failed"),
            &json!("failed"),
            &Value::Null,
            &Value::Null
        ),
        "{body}"
    );
    assert!(jobs[0]["finished_at"].is_string());

    let filtered: Vec<Value> = http
        .get(format!("{url}/api/jobs?prompt_id={}", p.id))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0]["id"], job);
    let absent: Vec<Value> = http
        .get(format!("{url}/api/jobs?prompt_id=absent"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(absent, Vec::<Value>::new());
    // /jobs is the usage page's training view, served as itself rather than redirected.
    let redirects = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let jobs_page = redirects.get(format!("{url}/jobs")).send().await.unwrap();
    assert_eq!(jobs_page.status(), 200);

    // An admin's cancel shows as one.
    let r = admin(http.post(format!("{url}/admin/api/prompts/{}/train", p.id)))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let again: Value = r.json().await.unwrap();
    let again = again["id"].as_str().unwrap();
    let r = admin(http.post(format!("{url}/admin/api/jobs/{again}/cancel")))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let (body, jobs) = public_jobs(&http, &url).await;
    hides(&body);
    assert_eq!(jobs.len(), 2);
    assert_eq!(
        (&jobs[0]["id"], &jobs[0]["status"], &jobs[0]["phase"]),
        (&json!(again), &json!("failed"), &json!("cancelled"))
    );
    assert_eq!(jobs[1]["phase"], "failed");
    assert!(jobs.iter().all(|j| j["prompt"] == named), "{body}");

    // The page is served without a token, reads /api/jobs and has no write actions; the usage
    // and admin pages link to it.
    let page = http
        .get(format!("{url}/jobs"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        page.contains("Training jobs")
            && page.contains("\"/api/jobs\"")
            && page.contains("/api/jobs?prompt_id=")
            && page.contains("Training history"),
        "{page}"
    );
    for write in ["method:", "/cancel", "/retry", "/train", "/admin/api"] {
        assert!(!page.contains(write), "{write} in the page");
    }
    for other in ["/", "/admin"] {
        let html = http
            .get(format!("{url}{other}"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(html.contains(r#"<a href="/jobs">Training</a>"#), "{other}");
    }
}

/// A job whose prompt the gateway doesn't know is listed with its prompt's id alone.
#[tokio::test]
async fn a_job_of_an_unknown_prompt_has_only_its_prompt_id() {
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = start_gateway(dir.path(), None).await;
    let job = state
        .db
        .create_job("0123456789abcdef", &json!({"reason": "manual"}))
        .unwrap();
    let (_, jobs) = public_jobs(&reqwest::Client::new(), &url).await;
    assert_eq!(
        (&jobs[0]["id"], &jobs[0]["prompt_id"], &jobs[0]["prompt"]),
        (&json!(job.id), &json!("0123456789abcdef"), &Value::Null)
    );
}

/// With `SN_VIEWER_TOKEN` set, the training progress takes the viewer or the admin token, like
/// the usage page; the page itself asks for it.
#[tokio::test]
async fn a_viewer_token_guards_training_progress() {
    let dir = tempfile::tempdir().unwrap();
    let (url, _state) = start_gateway(dir.path(), Some(VIEWER)).await;
    let http = reqwest::Client::new();
    for (token, status) in [
        (None, 401),
        (Some("viewer-secreT"), 401),
        (Some(VIEWER), 200),
        (Some(ADMIN), 200),
        (Some(WORKER), 401),
    ] {
        let mut req = http.get(format!("{url}/api/jobs"));
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        assert_eq!(req.send().await.unwrap().status(), status, "with {token:?}");
    }
    let r = http.get(format!("{url}/jobs")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let h = r.headers();
    assert!(h["content-security-policy"]
        .to_str()
        .unwrap()
        .contains("frame-ancestors 'none'"));
    assert_eq!(h["x-frame-options"], "DENY");
    assert_eq!(h["x-content-type-options"], "nosniff");
    assert!(r.text().await.unwrap().contains("Viewer token"));
}
