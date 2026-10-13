//! The language router: which language a text is in, and how sure that is, for students trained
//! one per language.
//!
//! Every logged input records its language and the router's confidence (`lang`,
//! `lang_confidence`), so the worker splits training data by the same router that serves. A
//! prompt with one student per language has, besides its student for every text, a student for
//! each language with enough confidently detected inputs (`lang/{code}/` next to its files, listed
//! in `languages.json`). A text goes to its language's student when the router is at least
//! `min_confidence` sure of the language, and to the prompt's student otherwise.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::student::Student;

/// What `languages.json` lists, next to a student trained with one student per language.
pub const FILE: &str = "languages.json";

/// The router looks at most this many characters: enough to tell the language, and a bound on
/// the time it takes.
const MAX_CHARS: usize = 2000;

/// The language of a text, as an ISO 639-3 code (`eng`, `por`, `hin`...), and how sure the
/// router is of it, from 0 to 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub code: &'static str,
    pub confidence: f64,
}

/// The language `text` is in, or None when the router can't tell (no letters, say).
pub fn detect(text: &str) -> Option<Detection> {
    let head = match text.char_indices().nth(MAX_CHARS) {
        Some((end, _)) => &text[..end],
        None => text,
    };
    whatlang::detect(head).map(|info| Detection {
        code: info.lang().code(),
        confidence: info.confidence(),
    })
}

/// How the worker trained the students per language: `languages.json`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Manifest {
    /// The router's least confidence for a text to go to its language's student.
    pub min_confidence: f64,
    /// Each language's student: its threshold, and what it was trained on.
    pub languages: HashMap<String, LanguageStudent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LanguageStudent {
    pub threshold: f64,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub examples: Option<u64>,
}

/// A deployment's students per language (none for a prompt with one student for every text).
#[derive(Default)]
pub struct Students {
    pub min_confidence: f64,
    pub by_language: HashMap<String, (Student, f64)>,
}

impl Students {
    /// The students in `dir/lang/{code}/`, as `dir/languages.json` lists them, if it does.
    pub fn load(dir: &Path, threads: Option<usize>) -> Result<Self> {
        let path = dir.join(FILE);
        if !path.exists() {
            return Ok(Self::default());
        }
        let manifest: Manifest = serde_json::from_slice(&std::fs::read(&path)?)
            .with_context(|| format!("parsing {}", path.display()))?;
        let mut by_language = HashMap::new();
        for (code, l) in manifest.languages {
            let student = Student::load(&dir.join("lang").join(&code), threads)
                .with_context(|| format!("loading the {code} student"))?;
            by_language.insert(code, (student, l.threshold));
        }
        Ok(Students {
            min_confidence: manifest.min_confidence,
            by_language,
        })
    }

    /// The language's student for `text`, with its code and threshold, when the router is sure
    /// enough of a language that has one.
    pub fn route(&self, text: &str) -> Option<(&str, &Student, f64)> {
        if self.by_language.is_empty() {
            return None;
        }
        let d = detect(text).filter(|d| d.confidence >= self.min_confidence)?;
        self.by_language
            .get_key_value(d.code)
            .map(|(code, (student, threshold))| (code.as_str(), student, *threshold))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_tells_languages_apart_and_says_how_sure() {
        let pt =
            detect("Quero cancelar a minha assinatura e receber o dinheiro de volta, por favor.")
                .unwrap();
        assert_eq!(pt.code, "por");
        assert!(pt.confidence > 0.7, "{pt:?}");
        let en = detect("I would like to cancel my subscription and get my money back, please.")
            .unwrap();
        assert_eq!(en.code, "eng");
        let hi = detect("मैं अपनी सदस्यता रद्द करना चाहता हूं और अपना पैसा वापस चाहता हूं।").unwrap();
        assert_eq!(hi.code, "hin");
        assert_eq!(detect("12345 !!!"), None);
        // A long text is judged on its beginning.
        let long = format!(
            "{} {}",
            "Quero cancelar a minha assinatura.".repeat(200),
            "x".repeat(10_000)
        );
        assert_eq!(detect(&long).unwrap().code, "por");
    }

    #[test]
    fn without_languages_nothing_is_routed() {
        let none = Students::default();
        assert!(none.route("Quero cancelar a minha assinatura.").is_none());
        let dir = tempfile::tempdir().unwrap();
        assert!(Students::load(dir.path(), None)
            .unwrap()
            .by_language
            .is_empty());
    }
}
