//! A distilled student exported by `system-none train`: model.onnx + tokenizer.json + manifest.json.
//! Mirrors `system_none.runtime.Classifier`, so both give the same probabilities.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};

use anyhow::{anyhow, Context, Result};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use serde::Deserialize;
use tokenizers::{Tokenizer, TruncationParams};

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub labels: Vec<String>,
    #[serde(default = "one")]
    pub temperature: f64,
    #[serde(default = "default_max_len")]
    pub max_length: usize,
    pub inputs: Vec<String>,
    #[serde(default)]
    pub pad_token_id: i64,
    #[serde(default)]
    pub source_model: Option<String>,
    /// Put in front of every input (EmbeddingGemma's classification prompt, say).
    #[serde(default)]
    pub prefix: String,
}

fn one() -> f64 {
    1.0
}
fn default_max_len() -> usize {
    256
}

static RUNTIME: OnceLock<std::result::Result<(), String>> = OnceLock::new();

/// Load the ONNX Runtime shared library once: `path`, else `ORT_DYLIB_PATH`, else the system library.
pub fn init_runtime(path: Option<&Path>) -> Result<()> {
    RUNTIME
        .get_or_init(|| {
            let builder = match path {
                Some(p) => ort::init_from(p)
                    .map_err(|e| format!("loading ONNX Runtime from {}: {e}", p.display()))?,
                None => ort::init(),
            };
            builder.with_name("system-none-gateway").commit();
            Ok(())
        })
        .clone()
        .map_err(|e| anyhow!(e))
}

pub struct Student {
    pub manifest: Manifest,
    pub dir: PathBuf,
    tokenizer: Tokenizer,
    session: Mutex<Session>,
}

impl Student {
    pub fn load(dir: &Path, threads: Option<usize>) -> Result<Student> {
        let manifest: Manifest = serde_json::from_slice(
            &std::fs::read(dir.join("manifest.json")).context("reading manifest.json")?,
        )
        .context("parsing manifest.json")?;
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: manifest.max_length,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("truncation: {e}"))?;
        let mut builder = Session::builder().map_err(|e| anyhow!("ONNX Runtime: {e}"))?;
        if let Some(n) = threads {
            builder = builder.with_intra_threads(n).map_err(|e| anyhow!("{e}"))?;
        }
        let session = builder
            .commit_from_file(dir.join("model.onnx"))
            .map_err(|e| anyhow!("model.onnx: {e}"))?;
        Ok(Student {
            manifest,
            dir: dir.to_path_buf(),
            tokenizer,
            session: Mutex::new(session),
        })
    }

    pub fn name(&self) -> String {
        self.manifest
            .source_model
            .clone()
            .unwrap_or_else(|| self.dir.display().to_string())
    }

    /// Calibrated probabilities for one text (temperature-scaled softmax).
    pub fn predict_proba(&self, text: &str) -> Result<Vec<f64>> {
        let enc = self
            .tokenizer
            .encode(format!("{}{text}", self.manifest.prefix), true)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        let n = enc.get_ids().len();
        let mut feed: Vec<(String, SessionInputValue<'_>)> =
            Vec::with_capacity(self.manifest.inputs.len());
        for name in &self.manifest.inputs {
            let data: Vec<i64> = match name.as_str() {
                "input_ids" => enc.get_ids().iter().copied().map(i64::from).collect(),
                "attention_mask" => enc
                    .get_attention_mask()
                    .iter()
                    .copied()
                    .map(i64::from)
                    .collect(),
                "token_type_ids" => enc.get_type_ids().iter().copied().map(i64::from).collect(),
                other => return Err(anyhow!("unsupported model input '{other}'")),
            };
            let t = Tensor::from_array(([1usize, n], data)).map_err(|e| anyhow!("{e}"))?;
            feed.push((name.clone(), t.into()));
        }
        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let out = session.run(feed).map_err(|e| anyhow!("inference: {e}"))?;
        let (_shape, logits) = out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("{e}"))?;
        Ok(softmax(logits, self.manifest.temperature))
    }
}

pub fn softmax(logits: &[f32], temperature: f64) -> Vec<f64> {
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    let z: Vec<f64> = logits.iter().map(|&x| f64::from(x) / t).collect();
    let m = z.iter().copied().fold(f64::MIN, f64::max);
    let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_with_temperature() {
        let p = softmax(&[2.0, 0.0], 1.0);
        assert!((p[0] - 0.880_797).abs() < 1e-5);
        let sharp = softmax(&[2.0, 0.0], 0.5);
        assert!(sharp[0] > p[0]);
    }

    /// Runs when `SN_TEST_MODEL_DIR` points at an exported student (CI sets it after the Python tests).
    #[test]
    fn loads_an_exported_student() {
        let Ok(dir) = std::env::var("SN_TEST_MODEL_DIR") else {
            return;
        };
        init_runtime(None).unwrap();
        let s = Student::load(Path::new(&dir), Some(1)).unwrap();
        let p = s
            .predict_proba("please refund refund money back charged twice")
            .unwrap();
        assert_eq!(p.len(), s.manifest.labels.len());
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }
}
