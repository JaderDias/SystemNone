use serde_json::json;
use system_none_gateway::db::{today, Db, Settings};

#[test]
fn route_percentiles_survive_restarts_and_keep_models_separate() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("db.sqlite");
    let db = Db::open(&file).unwrap();
    assert_eq!(
        Settings::default().default_base_model,
        "jhu-clsp/ettin-encoder-17m"
    );
    let ettin: Vec<String> = vec!["student:jhu-clsp/ettin-encoder-17m".into()];
    let larger = vec!["student:jhu-clsp/ettin-encoder-150m".into()];
    let fallback = vec![ettin[0].clone(), "system_one:large".into()];
    for ms in [1.0, 1.0, 3.0, 5.0, 9.0] {
        db.record_path("p", &ettin, ms).unwrap();
    }
    db.record_path("p", &larger, 15.0).unwrap();
    db.record_path("p", &fallback, 110.0).unwrap();
    db.record_path("other", &ettin, 999.0).unwrap();
    drop(db);
    let db = Db::open(&file).unwrap();
    let rows = db.path_latencies("p", &today()).unwrap();
    assert_eq!(rows.len(), 3);
    let row = rows.iter().find(|r| r["path"] == json!(ettin)).unwrap();
    assert_eq!(row["count"], 5);
    assert_eq!(row["p50_ms"], 3.0);
    assert_eq!(row["max_ms"], 9.0);
    // Merge original samples, not the medians of routes that ended at the same model.
    db.record_path("p", &["system_one:large".into()], 10.0)
        .unwrap();
    db.record_path(
        "p",
        &["student:another".into(), "system_one:large".into()],
        20.0,
    )
    .unwrap();
    let models = db.answering_model_latencies("p", &today()).unwrap();
    let model = models
        .iter()
        .find(|r| r["model"] == "system_one:large")
        .unwrap();
    assert_eq!(model["count"], 3);
    assert_eq!(model["p50_ms"], 20.0);
    assert_eq!(model["max_ms"], 110.0);
    let model = models.iter().find(|r| r["model"] == ettin[0]).unwrap();
    assert_eq!(
        model["count"], 5,
        "fallback requests do not count as student answers"
    );
    assert_eq!(
        db.path_latencies("p", "9999-01-01").unwrap(),
        Vec::<serde_json::Value>::new()
    );
}
