//! Where request inputs and trained students live: a local folder (`file:///...`) or S3 (`s3://bucket/prefix`).
//!
//! Layout (shared with the Python worker):
//!
//! ```text
//! prompts/{prompt_id}.json                          the prompt definition
//! inputs/{prompt_id}/{YYYY-MM-DD}/{ms}-{rand}.jsonl  logged inputs: {"id", "text", "ts"}; never the answers
//! models/{prompt_id}/{job_id}/{model.onnx,tokenizer.json,manifest.json}
//! ```

use std::collections::HashMap;
use std::path::Path as FsPath;
use std::sync::{Arc, Mutex, PoisonError};

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use futures::TryStreamExt;
use object_store::{path::Path, ObjectStore, ObjectStoreExt, PutPayload};
use tokio::sync::Notify;

use crate::prompt::{text_id, PromptSpec};

pub const MODEL_FILES: [&str; 3] = ["model.onnx", "tokenizer.json", "manifest.json"];

pub struct Store {
    inner: Arc<dyn ObjectStore>,
    prefix: Path,
    pub url: String,
}

impl Store {
    pub fn open(url: &str) -> Result<Store> {
        let parsed = url::Url::parse(url).with_context(|| {
            format!("store URL {url:?} (use file:///abs/path or s3://bucket/prefix)")
        })?;
        let (inner, prefix): (Arc<dyn ObjectStore>, Path) = match parsed.scheme() {
            "file" => {
                let dir = parsed
                    .to_file_path()
                    .map_err(|()| anyhow::anyhow!("bad file URL {url}"))?;
                std::fs::create_dir_all(&dir)
                    .with_context(|| format!("creating {}", dir.display()))?;
                (
                    Arc::new(object_store::local::LocalFileSystem::new_with_prefix(&dir)?),
                    Path::from(""),
                )
            }
            "s3" => {
                let bucket = parsed.host_str().unwrap_or_default();
                let s3 = object_store::aws::AmazonS3Builder::from_env()
                    .with_bucket_name(bucket)
                    .build()?;
                (Arc::new(s3), Path::from(parsed.path().trim_matches('/')))
            }
            "memory" => (
                Arc::new(object_store::memory::InMemory::new()),
                Path::from(""),
            ),
            other => bail!("unsupported store scheme '{other}'"),
        };
        Ok(Store {
            inner,
            prefix,
            url: url.to_string(),
        })
    }

    fn key(&self, rel: &str) -> Path {
        if self.prefix.as_ref().is_empty() {
            Path::from(rel)
        } else {
            Path::from(format!("{}/{}", self.prefix, rel))
        }
    }

    pub async fn put(&self, rel: &str, data: Bytes) -> Result<()> {
        self.inner
            .put(&self.key(rel), PutPayload::from(data))
            .await
            .with_context(|| format!("writing {rel}"))?;
        Ok(())
    }

    pub async fn get(&self, rel: &str) -> Result<Bytes> {
        Ok(self
            .inner
            .get(&self.key(rel))
            .await
            .with_context(|| format!("reading {rel}"))?
            .bytes()
            .await?)
    }

    /// Keys under `rel_prefix`, relative to the store root, sorted.
    pub async fn list(&self, rel_prefix: &str) -> Result<Vec<String>> {
        let base = self.prefix.as_ref().to_string();
        let mut keys: Vec<String> = self
            .inner
            .list(Some(&self.key(rel_prefix)))
            .map_ok(|m| {
                let k = m.location.to_string();
                k.strip_prefix(&base)
                    .unwrap_or(&k)
                    .trim_start_matches('/')
                    .to_string()
            })
            .try_collect()
            .await?;
        keys.sort();
        Ok(keys)
    }

    pub async fn put_prompt(&self, spec: &PromptSpec) -> Result<()> {
        self.put(
            &format!("prompts/{}.json", spec.id()),
            Bytes::from(serde_json::to_vec_pretty(spec)?),
        )
        .await
    }

    /// Copy a published student into a local folder so ONNX Runtime can open it. The files go
    /// to a temporary folder that is renamed into place once complete: `load_student` trusts any
    /// folder with a model.onnx, so `dest` must never hold part of a student, neither after a
    /// failed read nor while another download of it is writing (verification/Serving.tla).
    pub async fn download_model(&self, prefix: &str, dest: &FsPath) -> Result<()> {
        let mut part = dest.as_os_str().to_owned();
        part.push(format!(
            ".part-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        ));
        let part = std::path::PathBuf::from(part);
        let mut result = self.download_files(prefix, &part).await;
        if result.is_ok() {
            if let Err(e) = tokio::fs::rename(&part, dest).await {
                // Unless another download of the same student got there first.
                if !dest.join("model.onnx").exists() {
                    result = Err(e.into());
                }
            }
        }
        tokio::fs::remove_dir_all(&part).await.ok();
        result
    }

    async fn download_files(&self, prefix: &str, dir: &FsPath) -> Result<()> {
        tokio::fs::create_dir_all(dir).await?;
        for f in MODEL_FILES {
            let data = self
                .get(&format!("{}/{f}", prefix.trim_end_matches('/')))
                .await?;
            tokio::fs::write(dir.join(f), &data).await?;
        }
        // Students trained with reference labels publish the known answers to audit them against.
        // (Listing takes a folder: a file's own key lists nothing.)
        let folder = prefix.trim_end_matches('/');
        let listed = self.list(folder).await?;
        let optional = format!("{folder}/{}", crate::references::FILE);
        if listed.contains(&optional) {
            let data = self.get(&optional).await?;
            tokio::fs::write(dir.join(crate::references::FILE), &data).await?;
        }
        // A student per language: each in lang/{code}/, as languages.json lists them.
        let languages = format!("{folder}/{}", crate::language::FILE);
        if listed.contains(&languages) {
            let data = self.get(&languages).await?;
            let manifest: crate::language::Manifest =
                serde_json::from_slice(&data).context("parsing languages.json")?;
            for code in manifest.languages.keys() {
                if code.is_empty() || !code.chars().all(|c| c.is_ascii_alphanumeric()) {
                    bail!("languages.json names a language '{code}' that isn't a code");
                }
                let lang_dir = dir.join("lang").join(code);
                tokio::fs::create_dir_all(&lang_dir).await?;
                for f in MODEL_FILES {
                    let data = self.get(&format!("{folder}/lang/{code}/{f}")).await?;
                    tokio::fs::write(lang_dir.join(f), &data).await?;
                }
            }
            tokio::fs::write(dir.join(crate::language::FILE), &data).await?;
        }
        Ok(())
    }
}

/// Buffers inputs per prompt and writes them as JSONL objects (S3 has no append).
pub struct InputLog {
    store: Arc<Store>,
    buf: Mutex<HashMap<String, Vec<String>>>,
    pending: Mutex<usize>,
    /// One flush at a time, so a flush that returns has written what was pushed before it
    /// started, even the records a flush in progress took first: `claim_job` and shutdown rely on
    /// it (verification/InputLog.tla).
    flushing: tokio::sync::Mutex<()>,
    max_chars: usize,
    pub batch: usize,
    pub wake: Notify,
}

impl InputLog {
    pub fn new(store: Arc<Store>, batch: usize) -> InputLog {
        InputLog {
            store,
            buf: Mutex::new(HashMap::new()),
            pending: Mutex::new(0),
            flushing: tokio::sync::Mutex::new(()),
            max_chars: 8000,
            batch,
            wake: Notify::new(),
        }
    }

    pub fn push(&self, prompt_id: &str, text: &str) {
        let text: String = text.chars().take(self.max_chars).collect();
        #[expect(
            clippy::cast_precision_loss,
            reason = "milliseconds since 1970, which stay below 2^52 for another 140,000 years"
        )]
        let ts = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
        let mut record = serde_json::json!({"id": text_id(&text), "text": text, "ts": ts});
        // The language router's view, for training a student per language on what it routes.
        if let Some(d) = crate::language::detect(&text) {
            record["lang"] = d.code.into();
            record["lang_confidence"] = ((d.confidence * 1000.0).round() / 1000.0).into();
        }
        let line = record.to_string();
        self.buf
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(prompt_id.to_string())
            .or_default()
            .push(line);
        let mut p = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
        *p += 1;
        if *p >= self.batch {
            self.wake.notify_one();
        }
    }

    /// Write everything buffered so far. Returns the number of records written.
    pub async fn flush(&self) -> Result<usize> {
        let _flushing = self.flushing.lock().await;
        let taken: HashMap<String, Vec<String>> =
            std::mem::take(&mut *self.buf.lock().unwrap_or_else(PoisonError::into_inner));
        *self.pending.lock().unwrap_or_else(PoisonError::into_inner) = 0;
        let mut n = 0;
        for (prompt_id, lines) in taken {
            if lines.is_empty() {
                continue;
            }
            let now = chrono::Utc::now();
            let key = format!(
                "inputs/{prompt_id}/{}/{}-{}.jsonl",
                now.format("%Y-%m-%d"),
                now.timestamp_millis(),
                &uuid::Uuid::new_v4().simple().to_string()[..8]
            );
            n += lines.len();
            let body = lines.join("\n") + "\n";
            if let Err(e) = self.store.put(&key, Bytes::from(body)).await {
                tracing::error!("input log write failed, keeping records for retry: {e:#}");
                self.buf
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .entry(prompt_id)
                    .or_default()
                    .extend(lines);
                n = 0;
            }
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn logs_inputs_as_jsonl_objects() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(Store::open(&format!("file://{}", dir.path().display())).unwrap());
        let log = InputLog::new(store.clone(), 100);
        log.push(
            "p1",
            "I would like to cancel my subscription and get my money back, please.",
        );
        log.push("p1", "world");
        log.push("p2", "other");
        assert_eq!(log.flush().await.unwrap(), 3);
        let keys = store.list("inputs/p1").await.unwrap();
        assert_eq!(keys.len(), 1);
        let body = String::from_utf8(store.get(&keys[0]).await.unwrap().to_vec()).unwrap();
        let recs: Vec<serde_json::Value> = body
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert!(recs[0]["text"]
            .as_str()
            .unwrap()
            .starts_with("I would like"));
        // The input, its language as the router sees it, and when: no answer is ever stored.
        let mut keys: Vec<&String> = recs[0].as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(keys, ["id", "lang", "lang_confidence", "text", "ts"]);
        assert_eq!(recs[0]["lang"], "eng");
        assert_eq!(log.flush().await.unwrap(), 0);
    }

    /// verification/InputLog.tla, `ClaimSeesEarlierInputs` and `NoLossOnShutdown`: `claim_job` and
    /// shutdown flush so that the store holds every logged input, while the background flush
    /// may have taken the buffer and still be writing it. Theirs must wait for it.
    #[tokio::test]
    async fn a_flush_returns_once_the_flush_in_progress_is_written() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(Store::open(&format!("file://{}", dir.path().display())).unwrap());
        let log = Arc::new(InputLog::new(store, 100));
        for i in 0..20 {
            log.push(&format!("p{i}"), "hello");
        }
        let background = tokio::spawn({
            let log = log.clone();
            async move { log.flush().await }
        });
        tokio::task::yield_now().await; // the background flush takes the buffer, starts writing
        log.flush().await.unwrap();
        assert_eq!(jsonl_files(&dir.path().join("inputs")), 20);
        assert_eq!(background.await.unwrap().unwrap(), 20);
    }

    /// Counted without yielding to the runtime, so a flush in progress can't finish meanwhile.
    fn jsonl_files(dir: &FsPath) -> usize {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return 0;
        };
        entries
            .flatten()
            .map(|e| match e.path() {
                p if p.is_dir() => jsonl_files(&p),
                p => usize::from(p.extension().is_some_and(|x| x == "jsonl")),
            })
            .sum()
    }

    /// verification/Serving.tla, `LoadFailsOnlyOnItsOwnError`: `load_student` skips the download
    /// when the cache has a model.onnx, so a download that fails half way must leave nothing.
    #[tokio::test]
    async fn a_failed_download_leaves_no_partial_student() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&format!("file://{}", dir.path().join("store").display())).unwrap();
        for (f, body) in [("model.onnx", "onnx"), ("tokenizer.json", "{}")] {
            store
                .put(&format!("models/p/j/{f}"), Bytes::from(body))
                .await
                .unwrap();
        }
        let dest = dir.path().join("cache").join("p").join("j");
        assert!(
            store.download_model("models/p/j", &dest).await.is_err(),
            "manifest.json is missing"
        );
        assert!(!dest.join("model.onnx").exists());
        store
            .put("models/p/j/manifest.json", Bytes::from("{}"))
            .await
            .unwrap();
        store.download_model("models/p/j", &dest).await.unwrap();
        for f in MODEL_FILES {
            assert!(dest.join(f).exists(), "{f}");
        }
        // Nothing else is left in the prompt's cache folder.
        assert_eq!(
            std::fs::read_dir(dir.path().join("cache").join("p"))
                .unwrap()
                .count(),
            1
        );
    }
}
