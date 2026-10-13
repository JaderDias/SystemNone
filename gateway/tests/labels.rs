//! Prompts with more than 26 labels: `openai_chat` System One models answer with the label's
//! name instead of a letter, and a prompt takes up to 255 labels.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use system_none_gateway::{api, AppState, GatewayConfig};

const ADMIN: &str = "admin-secret";

/// A System One stand-in for the name prompt: the text names its intent (`intent_17`), and the
/// model answers with that name and a full stop, with the tokens' logprobs when asked for them.
/// Texts about two intents get an ambiguous answer, and other texts one that names no label.
async fn mock_named_system_one(bodies: Arc<Mutex<Vec<Value>>>) -> SocketAddr {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(body): Json<Value>| {
            let bodies = bodies.clone();
            async move {
                bodies.lock().unwrap().push(body.clone());
                let prompt = body["messages"][1]["content"].as_str().unwrap_or("");
                let text = prompt.split("\"\"\"").nth(1).unwrap_or("");
                let intents: Vec<&str> = text
                    .match_indices("intent_")
                    .filter_map(|(i, _)| text.get(i..i + 9))
                    .collect();
                let (content, tokens) = match intents.as_slice() {
                    [name] => (
                        format!("{name}."),
                        json!([{"token": "intent", "logprob": 0.8f64.ln()}, {"token": "_", "logprob": 0.0},
                               {"token": &name[7..], "logprob": 0.0}, {"token": ".", "logprob": 0.5f64.ln()}]),
                    ),
                    [a, b, ..] => (
                        format!("Either {a} or {b}"),
                        json!([{"token": "Either", "logprob": -0.1}]),
                    ),
                    [] => (
                        "I can't tell".to_string(),
                        json!([{"token": "I", "logprob": -0.1}]),
                    ),
                };
                let mut choice = json!({"message": {"content": content}});
                if body["logprobs"] == true {
                    choice["logprobs"] = json!({"content": tokens});
                }
                Json(json!({"choices": [choice]}))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

async fn start_gateway(dir: &std::path::Path) -> String {
    let cfg = GatewayConfig {
        db_path: dir.join("gw.db"),
        store_url: format!("file://{}", dir.join("store").display()),
        cache_dir: dir.join("cache"),
        admin_token: ADMIN.into(),
        worker_token: "worker-secret".into(),
        viewer_token: None,
        onnxruntime: None,
        threads: Some(1),
        flush_secs: 1,
        log_batch: 1000,
    };
    let state = AppState::start(cfg).await.unwrap();
    state.spawn_background();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = api::router(state);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    url
}

/// `intent_00` … `intent_39`, then `extra_40` … up to `n` labels.
fn prompt(n: usize) -> Value {
    let labels: Vec<String> = (0..n)
        .map(|i| {
            if i < 40 {
                format!("intent_{i:02}")
            } else {
                format!("extra_{i}")
            }
        })
        .collect();
    json!({"instructions": "What does the customer want?", "labels": labels})
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    clippy::too_many_lines,
    reason = "one scenario, step by step, against one gateway"
)]
async fn openai_chat_answers_more_than_26_labels_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let bodies = Arc::new(Mutex::new(vec![]));
    let s1 = mock_named_system_one(bodies.clone()).await;
    let url = start_gateway(dir.path()).await;
    let http = reqwest::Client::new();
    let admin = |req: reqwest::RequestBuilder| req.bearer_auth(ADMIN);

    for (name, logprobs) in [("named", true), ("no-logprobs", false)] {
        let r = admin(http.put(format!("{url}/admin/api/system-one-models/{name}")))
            .json(
                &json!({"kind": "openai_chat", "base_url": format!("http://{s1}/v1"), "model": "s1",
                          "logprobs": logprobs}),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
    }
    let r = admin(http.put(format!("{url}/admin/api/settings")))
        .json(&json!({"default_system_one": "named"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let classify = |body: Value| http.post(format!("{url}/v1/classify")).json(&body).send();

    // 40 labels: the name answer maps to its label; the confidence is the probability of the
    // tokens that named it (0.8), without the full stop after the name.
    let v: Value = classify(json!({"prompt": prompt(40), "input": "Where is my intent_17 thing?"}))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["source"], "system_one", "{v}");
    assert_eq!(v["label"], "intent_17");
    assert!(
        (v["confidence"].as_f64().unwrap() - 0.8).abs() < 1e-4,
        "{v}"
    );
    let pid = v["prompt_id"].as_str().unwrap().to_string();
    {
        let sent = bodies.lock().unwrap();
        let body = sent.last().unwrap();
        assert_eq!(
            body["max_tokens"],
            8 + "intent_17".len(),
            "room for the longest name"
        );
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["logprobs"], true);
        assert!(body.get("top_logprobs").is_none(), "{body}");
        let user = body["messages"][1]["content"].as_str().unwrap();
        assert!(
            user.contains("Options:\n- intent_00\n- intent_01\n"),
            "{user}"
        );
        assert!(user.contains("- intent_39\n\nText:"), "{user}");
        assert!(user.ends_with("exactly as written above, and nothing else."));
    }

    // Without logprobs there is an answer but no confidence.
    let v: Value =
        classify(json!({"prompt_id": pid, "input": "intent_05 please", "model": "no-logprobs"}))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
    assert_eq!(v["label"], "intent_05", "{v}");
    assert!(v["confidence"].is_null(), "{v}");
    assert!(bodies
        .lock()
        .unwrap()
        .last()
        .unwrap()
        .get("logprobs")
        .is_none());

    // An answer that names no label, or two, is an error rather than a guess.
    for input in ["hello there", "intent_01 or intent_02?"] {
        let r = classify(json!({"prompt_id": pid, "input": input}))
            .await
            .unwrap();
        assert_eq!(r.status(), 502, "{input}");
        let e: Value = r.json().await.unwrap();
        assert!(
            e["error"].as_str().unwrap().contains("'named' failed"),
            "{e}"
        );
    }

    // Up to 255 labels; one more, or fewer than 2, is refused with the reason.
    let v: Value = classify(json!({"prompt": prompt(255), "input": "about intent_03"}))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["label"], "intent_03", "{v}");
    let sent = bodies.lock().unwrap().last().unwrap().clone();
    let user = sent["messages"][1]["content"].as_str().unwrap();
    assert!(
        user.contains("- extra_254\n\nText:"),
        "every label is listed"
    );
    for (n, reason) in [
        (256, "a prompt takes at most 255 labels, got 256"),
        (1, "a prompt needs at least 2 labels, got 1"),
    ] {
        let r = classify(json!({"prompt": prompt(n), "input": "about intent_00"}))
            .await
            .unwrap();
        assert_eq!(r.status(), 400);
        assert_eq!(r.json::<Value>().await.unwrap()["error"], reason);
    }
}
