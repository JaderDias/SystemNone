//! Decisions clients and upstreams share the student/logging/escalation pipeline.
use std::sync::{Arc, Mutex};

use axum::{routing::post, Json, Router};
use serde_json::{json, Value};
use system_none_gateway::{api, AppState, GatewayConfig};

async fn serve(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    url
}

async fn gateway(dir: &std::path::Path) -> (String, Arc<AppState>) {
    let state = AppState::start(GatewayConfig {
        db_path: dir.join("gw.db"),
        store_url: url::Url::from_directory_path(dir.join("store"))
            .unwrap()
            .to_string(),
        cache_dir: dir.join("cache"),
        admin_token: "admin".into(),
        worker_token: "worker".into(),
        viewer_token: None,
        onnxruntime: None,
        threads: Some(1),
        flush_secs: 1,
        log_batch: 1000,
    })
    .await
    .unwrap();
    (serve(api::router(state.clone())).await, state)
}

fn choice() -> Value {
    json!({"id": "intent", "type": "choice", "question": "Which intent?",
        "options": [{"name": "refund", "description": "money back"}, {"name": "other"}]})
}

async fn configure(http: &reqwest::Client, url: &str, name: &str, model: Value) {
    let response = http
        .put(format!("{url}/admin/api/system-one-models/{name}"))
        .bearer_auth("admin")
        .json(&model)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "one HTTP compatibility scenario including validation"
)]
async fn decisions_upstream_and_downstream() {
    let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
    let copy = seen.clone();
    let upstream = serve(Router::new().route("/v1/decisions", post(move |Json(body): Json<Value>| {
        let seen = copy.clone();
        async move {
            seen.lock().unwrap().push(body.clone());
            let options = body["questions"][0]["options"].as_array().unwrap();
            assert_eq!(body["questions"][0]["type"], "choice");
            let mut probabilities = serde_json::Map::new();
            #[expect(clippy::cast_precision_loss, reason = "at most 26 options")]
            let rest = 0.2 / (options.len() - 1) as f64;
            for (i, option) in options.iter().enumerate() {
                probabilities.insert(option["name"].as_str().unwrap().into(), json!(if i == 0 { 0.8 } else { rest }));
            }
            Json(json!({"model": "decision-model", "answers": {"label": {
                "type": "choice", "choice": options[0]["name"], "probabilities": probabilities, "label_mass": 0.6
            }}, "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}}))
        }
    }))).await;
    let dir = tempfile::tempdir().unwrap();
    let (url, state) = gateway(dir.path()).await;
    let http = reqwest::Client::new();
    configure(&http, &url, "decisions", json!({"kind": "openai_decisions", "base_url": format!("{upstream}/v1/"), "model": "decision-model"})).await;
    let response = http
        .put(format!("{url}/admin/api/settings"))
        .bearer_auth("admin")
        .json(&json!({"default_system_one": "decisions"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let request = json!({"input": {"message": "refund please"}, "questions": [choice(),
        {"id": "urgent", "type": "yes_no", "question": "Is this urgent?", "yes": "needs attention"},
        {"id": "urgency", "type": "score", "question": "How urgent?", "levels": ["normal", "urgent"]}]});
    let response = http
        .post(format!("{url}/v1/decisions"))
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    let answer: Value = response.json().await.unwrap();
    assert_eq!(answer["object"], "decisions");
    assert_eq!(
        answer["usage"],
        json!({"prompt_tokens": 30, "completion_tokens": 0, "total_tokens": 30})
    );
    assert_eq!(answer["answers"]["intent"]["choice"], "refund");
    assert_eq!(
        answer["answers"]["intent"]["probabilities"],
        json!({"refund": 0.8, "other": 0.2})
    );
    assert_eq!(answer["answers"]["urgent"]["type"], "yes_no");
    assert_eq!(
        answer["answers"]["urgent"]["probabilities"],
        json!({"yes": 0.8, "no": 0.2})
    );
    assert!(answer["answers"]["urgent"].get("choice").is_none());
    assert!((answer["answers"]["urgency"]["score"].as_f64().unwrap() - 0.2).abs() < 1e-8);
    assert_eq!(answer["answers"]["intent"]["label_mass"], 0.6);
    assert_eq!(state.db.prompts().unwrap().len(), 3);
    let sent = seen.lock().unwrap()[0].clone();
    assert_eq!(sent["input"], "{\"message\":\"refund please\"}");
    assert_eq!(
        sent["questions"][0]["options"][0]["description"],
        "money back"
    );
    // The same choice through /classify shares its prompt and reports P(choice), not label mass.
    let result: Value = http.post(format!("{url}/v1/classify")).json(&json!({"input": "refund please",
        "prompt": {"instructions": "Which intent?", "labels": [{"name": "refund", "description": "money back"}, "other"]}}))
        .send().await.unwrap().json().await.unwrap();
    assert_eq!(result["confidence"], 0.8);
    assert_eq!(state.db.prompts().unwrap().len(), 3);
    // Temperature is applied to the distribution, without changing label mass.
    let result: Value = http
        .post(format!("{url}/v1/decisions"))
        .json(&json!({"input": "x", "temperature": 2,
        "questions": [choice()]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        (result["answers"]["intent"]["probabilities"]["refund"]
            .as_f64()
            .unwrap()
            - 2.0 / 3.0)
            .abs()
            < 1e-8
    );
    assert_eq!(result["answers"]["intent"]["label_mass"], 0.6);
    let calls = seen.lock().unwrap().len();
    for body in [
        json!({"input": "x", "questions": [choice(), choice()]}),
        json!({"input": "x", "questions": [choice(), {"id": "bad", "type": "score", "question": "q", "levels": ["one"]}]}),
        json!({"input": "x", "questions": [choice()], "temperature": 0}),
        json!({"input": " ", "questions": [choice()]}),
        json!({"input": "x", "questions": [choice()], "prompt_format_version": 2}),
    ] {
        let response = http
            .post(format!("{url}/v1/decisions"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400, "{body}");
    }
    assert_eq!(
        seen.lock().unwrap().len(),
        calls,
        "validate all questions before any side effects"
    );
    // Worker configurations can select a Decisions judge, subject to the existing license gate.
    let response = http
        .put(format!("{url}/admin/api/judges/decisions"))
        .bearer_auth("admin")
        .json(
            &json!({"base_url": format!("{upstream}/v1"), "model": "decision-model",
            "mode": "openai_decisions", "license_ack": true}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
}

#[tokio::test]
async fn decisions_judge_is_used_by_the_rust_audit_path() {
    let upstream = serve(Router::new().route(
        "/v1/decisions",
        post(|Json(body): Json<Value>| async move {
            assert_eq!(body["model"], "teacher");
            assert_eq!(body["questions"][0]["options"][0]["name"], "refund");
            Json(
                json!({"answers": {"label": {"type": "choice", "choice": "refund",
            "probabilities": {"refund": 0.8, "other": 0.2}, "label_mass": 0.6}}}),
            )
        }),
    ))
    .await;
    let judge = serde_json::from_value(json!({"name": "audit", "model": "teacher",
        "base_url": format!("{upstream}/v1"), "mode": "openai_decisions", "license_ack": true}))
    .unwrap();
    let prompt = serde_json::from_value(
        json!({"instructions": "Which intent?", "labels": ["refund", "other"]}),
    )
    .unwrap();
    let answer =
        system_none_gateway::upstream::judge(&reqwest::Client::new(), &judge, &prompt, "refund me")
            .await
            .unwrap();
    assert_eq!(answer.label, 0);
    assert_eq!(answer.confidence, Some(0.8));
    assert_eq!(answer.probabilities, Some(vec![0.8, 0.2]));
}

#[tokio::test]
async fn malformed_decisions_answers_fail_without_inventing_probabilities() {
    let invalid = Arc::new(Mutex::new(json!({"type": "choice", "choice": "refund"})));
    let copy = invalid.clone();
    let upstream = serve(Router::new().route(
        "/v1/decisions",
        post(move || {
            let copy = copy.clone();
            async move { Json(json!({"answers": {"label": copy.lock().unwrap().clone()}})) }
        }),
    ))
    .await;
    let dir = tempfile::tempdir().unwrap();
    let (url, _) = gateway(dir.path()).await;
    let http = reqwest::Client::new();
    configure(
        &http,
        &url,
        "bad",
        json!({"kind": "openai_decisions", "base_url": format!("{upstream}/v1"), "model": "m"}),
    )
    .await;
    for answer in [
        json!({"type": "choice", "choice": "refund"}),
        json!({"type": "choice", "choice": "refund", "probabilities": {"refund": -0.1, "other": 1.1}}),
        json!({"type": "choice", "choice": "refund", "probabilities": {"refund": 0.1, "other": 0.9}}),
        json!({"type": "choice", "choice": "unknown", "probabilities": {"refund": 0.8, "other": 0.2}}),
    ] {
        *invalid.lock().unwrap() = answer;
        let response = http
            .post(format!("{url}/v1/decisions"))
            .json(&json!({"model": "bad", "input": "x", "questions": [choice()]}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 502);
        assert!(!response.text().await.unwrap().contains(&upstream));
    }
}
