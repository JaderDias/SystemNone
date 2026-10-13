//! Known answers published with a student: the reference labels (human or dataset labels) the
//! worker held out of its training, to audit it against, and the ones it trained on.
//!
//! The worker writes them to `reference_labels.jsonl` next to `model.onnx`, one
//! `{"text", "label": index, "use": "audit" | "train"}` per line. A request whose text has an
//! `audit` answer is checked against that answer instead of the judge: it is free, so every such
//! request is checked. A request whose text has a `train` answer is not audited at all: the
//! student learned that exact answer, so agreeing with it proves nothing, and the judge may
//! disagree with the person who labeled it.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::prompt::text_id;

pub const FILE: &str = "reference_labels.jsonl";
/// The gateway and the worker truncate logged and labeled texts to this many characters.
const MAX_CHARS: usize = 8000;

/// What a student's answer to one request is checked against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reference {
    /// A known answer the student never trained on.
    Known(usize),
    /// A known answer the student trained on: not audited.
    Trained,
    /// No known answer: the judge, the model the student learned from.
    Judge,
}

/// Keyed by `text_id`, as logged inputs are, so whitespace around a text doesn't matter.
#[derive(Default)]
pub struct References {
    audit: HashMap<String, usize>,
    trained: HashSet<String>,
}

fn key(text: &str) -> String {
    if text.chars().nth(MAX_CHARS).is_some() {
        text_id(&text.chars().take(MAX_CHARS).collect::<String>())
    } else {
        text_id(text)
    }
}

#[derive(Deserialize)]
struct Line {
    text: String,
    label: usize,
    #[serde(rename = "use")]
    usage: String,
}

impl References {
    /// The references in `dir`, or none when the student was published without them.
    pub fn load(dir: &Path, num_labels: usize) -> Result<Self> {
        let path = dir.join(FILE);
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {FILE}"))?;
        Self::parse(&text, num_labels)
    }

    pub fn parse(text: &str, num_labels: usize) -> Result<Self> {
        let mut out = Self::default();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let l: Line =
                serde_json::from_str(line).with_context(|| format!("{FILE} line {}", i + 1))?;
            if l.label >= num_labels {
                anyhow::bail!("{FILE} line {}: label {} out of range", i + 1, l.label);
            }
            let id = key(&l.text);
            match l.usage.as_str() {
                "audit" => {
                    out.trained.remove(&id);
                    out.audit.insert(id, l.label);
                }
                "train" => {
                    out.audit.remove(&id);
                    out.trained.insert(id);
                }
                other => anyhow::bail!("{FILE} line {}: unknown use '{other}'", i + 1),
            }
        }
        Ok(out)
    }

    pub fn lookup(&self, text: &str) -> Reference {
        if self.audit.is_empty() && self.trained.is_empty() {
            return Reference::Judge;
        }
        let id = key(text);
        if let Some(&label) = self.audit.get(&id) {
            Reference::Known(label)
        } else if self.trained.contains(&id) {
            Reference::Trained
        } else {
            Reference::Judge
        }
    }

    pub fn audit_count(&self) -> usize {
        self.audit.len()
    }

    pub fn trained_count(&self) -> usize {
        self.trained.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_trained_and_unknown_texts() {
        let r = References::parse(
            concat!(
                r#"{"text": "where is my card?", "label": 2, "use": "audit"}"#,
                "\n",
                r#"{"text": "top up failed", "label": 0, "use": "train"}"#,
                "\n\n"
            ),
            3,
        )
        .unwrap();
        assert_eq!(r.lookup("where is my card?"), Reference::Known(2));
        assert_eq!(r.lookup("  where is my card?\n"), Reference::Known(2));
        assert_eq!(r.lookup("top up failed"), Reference::Trained);
        assert_eq!(r.lookup("something else"), Reference::Judge);
        assert_eq!((r.audit_count(), r.trained_count()), (1, 1));
        assert_eq!(References::default().lookup("x"), Reference::Judge);
    }

    #[test]
    fn rejects_labels_the_prompt_does_not_have() {
        assert!(References::parse(r#"{"text": "a", "label": 3, "use": "audit"}"#, 3).is_err());
        assert!(References::parse(r#"{"text": "a", "label": 0, "use": "maybe"}"#, 3).is_err());
    }

    #[test]
    fn a_later_line_replaces_an_earlier_one() {
        let r = References::parse(
            concat!(
                r#"{"text": "a", "label": 0, "use": "train"}"#,
                "\n",
                r#"{"text": "a", "label": 1, "use": "audit"}"#
            ),
            2,
        )
        .unwrap();
        assert_eq!(r.lookup("a"), Reference::Known(1));
        assert_eq!(r.trained_count(), 0);
    }
}
