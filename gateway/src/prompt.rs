//! A "prompt" is a classification instruction: the question plus its ordered labels.
//! Every request names one; usage, logging, training and routing are all per prompt.

use serde::{Deserialize, Serialize};
use sha1::Sha1;
use sha2::{Digest, Sha256};

pub const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
/// Labels a prompt may have. `openai_chat` models answer with an option letter up to
/// `LETTERS.len()` labels and with the label's name above (the Python judge's `answer: auto`).
pub const MAX_LABELS: usize = 255;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Label {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
}

/// Labels may be sent as plain strings or as `{name, description}` objects.
#[derive(Deserialize)]
#[serde(untagged)]
enum LabelIn {
    Name(String),
    Full {
        name: String,
        #[serde(default)]
        description: String,
    },
}

fn labels_de<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<Label>, D::Error> {
    let raw: Vec<LabelIn> = Vec::deserialize(d)?;
    Ok(raw
        .into_iter()
        .map(|l| match l {
            LabelIn::Name(name) => Label {
                name,
                description: String::new(),
            },
            LabelIn::Full { name, description } => Label { name, description },
        })
        .collect())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PromptSpec {
    pub instructions: String,
    #[serde(deserialize_with = "labels_de")]
    pub labels: Vec<Label>,
}

impl PromptSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.instructions.trim().is_empty() {
            return Err("prompt.instructions is empty".into());
        }
        let n = self.labels.len();
        if n < 2 {
            return Err(format!("a prompt needs at least 2 labels, got {n}"));
        }
        if n > MAX_LABELS {
            return Err(format!(
                "a prompt takes at most {MAX_LABELS} labels, got {n}"
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for l in &self.labels {
            if l.name.trim().is_empty() || !seen.insert(l.name.as_str()) {
                return Err(format!(
                    "label names must be non-empty and unique ('{}')",
                    l.name
                ));
            }
        }
        Ok(())
    }

    /// Stable id: the same instructions and labels always map to the same prompt.
    pub fn id(&self) -> String {
        let canonical = serde_json::json!({
            "instructions": self.instructions.trim(),
            "labels": self.labels.iter().map(|l| [l.name.as_str(), l.description.as_str()]).collect::<Vec<_>>(),
        });
        let digest = Sha256::digest(canonical.to_string().as_bytes());
        hex::encode(&digest[..8])
    }

    pub fn label_names(&self) -> Vec<String> {
        self.labels.iter().map(|l| l.name.clone()).collect()
    }

    /// First line of the instructions, shortened, for tables.
    pub fn title(&self) -> String {
        let first = self.instructions.trim().lines().next().unwrap_or("").trim();
        if first.chars().count() > 90 {
            format!("{}…", first.chars().take(89).collect::<String>())
        } else {
            first.to_string()
        }
    }

    /// Up to 26 labels, a model answers with the option's letter; above, with the label's name.
    pub fn answers_with_letters(&self) -> bool {
        self.labels.len() <= LETTERS.len()
    }

    /// The prompt the Python judge uses (`judge.build_messages`): lettered up to 26 labels,
    /// asking for the label's name above.
    pub fn messages(&self, text: &str) -> serde_json::Value {
        if !self.answers_with_letters() {
            return self.name_messages(text);
        }
        let options: Vec<String> = self
            .labels
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let letter = &LETTERS[i..=i];
                if l.description.is_empty() {
                    format!("{letter}) {}", l.name)
                } else {
                    format!("{letter}) {}: {}", l.name, l.description)
                }
            })
            .collect();
        let last = &LETTERS[self.labels.len() - 1..self.labels.len()];
        serde_json::json!([
            {"role": "system", "content": "You are a careful, calibrated classifier. Answer with a single capital letter."},
            {"role": "user", "content": format!(
                "{}\n\nOptions:\n{}\n\nText:\n\"\"\"\n{}\n\"\"\"\n\nAnswer with the letter of the best option (A-{}) and nothing else.",
                self.instructions.trim(), options.join("\n"), text, last)}
        ])
    }

    /// `judge.build_messages(..., answer="name")`: the options are listed by name.
    fn name_messages(&self, text: &str) -> serde_json::Value {
        let options: Vec<String> = self
            .labels
            .iter()
            .map(|l| {
                if l.description.is_empty() {
                    format!("- {}", l.name)
                } else {
                    format!("- {}: {}", l.name, l.description)
                }
            })
            .collect();
        serde_json::json!([
            {"role": "system", "content": "You are a careful, calibrated classifier. Answer with the name of one option."},
            {"role": "user", "content": format!(
                "{}\n\nOptions:\n{}\n\nText:\n\"\"\"\n{}\n\"\"\"\n\nAnswer with the name of the best option, exactly as written above, and nothing else.",
                self.instructions.trim(), options.join("\n"), text)}
        ])
    }

    /// Tokens a name answer may take: room for the longest label name, whatever the tokenizer
    /// (about one token per character at worst). The same as the Python judge.
    pub fn name_max_tokens(&self) -> usize {
        let longest = self
            .labels
            .iter()
            .map(|l| l.name.chars().count())
            .max()
            .unwrap_or(0);
        (8 + longest).min(128)
    }
}

/// Same id as the Python side (`shadow.text_id`), so logs written here de-duplicate there.
pub fn text_id(text: &str) -> String {
    let digest = Sha1::digest(text.trim().as_bytes());
    hex::encode(digest)[..16].to_string()
}

/// The first k option letters (letters stop at 26).
fn letters(k: usize) -> &'static str {
    &LETTERS[..k.min(LETTERS.len())]
}

/// Turn the first token's top logprobs into a distribution over the k option letters.
pub fn probs_from_top_logprobs(top: &[(String, f64)], k: usize) -> Option<Vec<f64>> {
    let mut mass = vec![0.0; k];
    for (tok, lp) in top {
        let t = tok
            .trim()
            .trim_matches(|c| c == '(' || c == ')' || c == '.' || c == ':')
            .to_uppercase();
        if t.len() == 1 {
            if let Some(i) = letters(k).find(&t) {
                mass[i] += lp.exp();
            }
        }
    }
    let total: f64 = mass.iter().sum();
    (total > 0.0).then(|| mass.iter().map(|m| m / total).collect())
}

/// The option letter in a free-text answer ("B", "B)", "Answer: B", ...).
pub fn letter_from_text(text: &str, k: usize) -> Option<usize> {
    let t = text.trim();
    let first = t.chars().next()?;
    if first.is_ascii_uppercase() && t.chars().nth(1).is_none_or(|c| !c.is_ascii_alphabetic()) {
        if let Some(i) = letters(k).find(first) {
            return Some(i);
        }
    }
    t.split(|c: char| !c.is_ascii_alphanumeric())
        .find(|w| w.len() == 1 && letters(k).contains(*w))
        .and_then(|w| LETTERS.find(w))
}

// ---- name answers (more than 26 labels): the same rules as the Python judge's `label_from_text` ----

/// Why a name answer maps to no label: it names none of them...
pub const UNKNOWN_NAME: &str = "names none of the prompt's labels";
/// ... or several.
pub const AMBIGUOUS_NAME: &str = "names more than one of the prompt's labels";

/// Case, `_` / `-` versus spaces and repeated whitespace don't tell labels apart (`judge._norm`).
fn norm(s: &str) -> String {
    s.replace(['_', '-'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

const QUOTES: &[char] = &[
    '`', '*', '"', '\'', '\u{201c}', '\u{201d}', '\u{2018}', '\u{2019}', '«', '»', ' ', '\t',
];

/// The rest of `s` after a leading `word` (ASCII, any case).
fn strip_word<'a>(s: &'a str, word: &str) -> Option<&'a str> {
    s.get(..word.len())
        .filter(|head| head.eq_ignore_ascii_case(word))
        .map(|_| &s[word.len()..])
}

fn strip_keyword(s: &str) -> Option<&str> {
    ["answer", "label", "option", "category", "class"]
        .iter()
        .find_map(|w| strip_word(s, w))
}

/// Drop an "Answer:" style prefix: `^(?:the\s+)?(?:answer|label|option|category|class)\s*(?:is)?\s*[:=-]?\s*`.
fn strip_answer_prefix(s: &str) -> &str {
    let after_the = strip_word(s, "the")
        .filter(|r| r.starts_with(char::is_whitespace))
        .and_then(|r| strip_keyword(r.trim_start()));
    let Some(rest) = after_the.or_else(|| strip_keyword(s)) else {
        return s;
    };
    let rest = rest.trim_start();
    let rest = strip_word(rest, "is").unwrap_or(rest).trim_start();
    rest.strip_prefix([':', '=', '-'])
        .unwrap_or(rest)
        .trim_start()
}

/// An answer without quotes, markdown, an "Answer:" prefix or a trailing full stop.
fn clean_answer(s: &str) -> &str {
    strip_answer_prefix(s.trim_matches(QUOTES))
        .trim_matches(QUOTES)
        .trim_end_matches(['.', '!', ';', ','])
        .trim_matches(QUOTES)
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `needle` appears in `hay` as whole words (`(?<!\w)needle(?!\w)`).
fn contains_words(hay: &str, needle: &str) -> bool {
    let mut start = 0;
    while let Some(off) = hay[start..].find(needle) {
        let pos = start + off;
        let before = hay[..pos].chars().next_back();
        let after = hay[pos + needle.len()..].chars().next();
        if !before.is_some_and(is_word) && !after.is_some_and(is_word) {
            return true;
        }
        start = pos + hay[pos..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// The label a free-text answer names.
///
/// An exact match comes first, then one that ignores case, `_` / `-` versus spaces, quotes,
/// markdown, a trailing full stop and an "Answer:" prefix. Otherwise the first line must name
/// exactly one label as whole words ("card arrival", not also "card"). An answer that names
/// none, or several, is an error rather than a guess.
pub fn label_from_name(text: &str, names: &[String]) -> Result<usize, &'static str> {
    let mut by_norm = std::collections::HashMap::<String, Vec<usize>>::new();
    for (i, n) in names.iter().enumerate() {
        let key = norm(n);
        if !key.is_empty() {
            by_norm.entry(key).or_default().push(i);
        }
    }
    let one = |key: &str| match by_norm.get(key).map(Vec::as_slice) {
        Some([i]) => Ok(*i),
        _ => Err(AMBIGUOUS_NAME),
    };
    let whole = text.trim();
    let first = whole
        .split([
            '\n', '\r', '\u{b}', '\u{c}', '\u{1c}', '\u{1d}', '\u{1e}', '\u{85}', '\u{2028}',
            '\u{2029}',
        ])
        .next()
        .unwrap_or("");
    for cand in [whole, first] {
        for c in [cand, clean_answer(cand)] {
            if let Some(i) = names.iter().position(|n| n == c) {
                return Ok(i);
            }
            let key = norm(c);
            if by_norm.contains_key(&key) {
                return one(&key);
            }
        }
    }
    let said = norm(first);
    let hits: Vec<&String> = by_norm
        .keys()
        .filter(|k| contains_words(&said, k))
        .collect();
    // "card arrival" also contains "card": keep the longer name.
    let hits: Vec<&String> = hits
        .iter()
        .filter(|n| !hits.iter().any(|m| m != *n && m.contains(n.as_str())))
        .copied()
        .collect();
    match hits.as_slice() {
        [] => Err(UNKNOWN_NAME),
        [key] => one(key.as_str()),
        _ => Err(AMBIGUOUS_NAME),
    }
}

/// exp(sum of logprobs) of the generated tokens, up to the end of the label's name
/// (`judge.answer_probability`): a full stop after the name doesn't count.
pub fn answer_probability(tokens: &[(String, f64)], name: &str) -> f64 {
    let target = norm(name);
    let mut text = String::new();
    let mut total = 0.0;
    for (tok, lp) in tokens {
        text.push_str(tok);
        total += lp;
        if norm(&text).contains(&target) {
            break;
        }
    }
    total.exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> PromptSpec {
        serde_json::from_value(serde_json::json!({
            "instructions": "What does the customer want?",
            "labels": ["refund", {"name": "cancel", "description": "stop the order"}, "other"]
        }))
        .unwrap()
    }

    #[test]
    fn ids_are_stable_and_content_addressed() {
        let a = spec();
        let mut b = spec();
        b.instructions = "  What does the customer want?\n".into();
        assert_eq!(a.id(), b.id());
        assert_eq!(a.id().len(), 16);
        b.labels.swap(0, 2);
        assert_ne!(a.id(), b.id(), "label order is part of the prompt");
    }

    #[test]
    fn validates_labels() {
        let mut s = spec();
        assert!(s.validate().is_ok());
        s.labels.truncate(1);
        assert_eq!(
            s.validate().unwrap_err(),
            "a prompt needs at least 2 labels, got 1"
        );
        s.labels = (0..MAX_LABELS).map(|i| label(&format!("l{i}"))).collect();
        assert_eq!(s.labels.len(), 255);
        assert!(s.validate().is_ok());
        s.labels.push(label("one too many"));
        assert_eq!(
            s.validate().unwrap_err(),
            "a prompt takes at most 255 labels, got 256"
        );
        let dup: PromptSpec =
            serde_json::from_value(serde_json::json!({"instructions": "x", "labels": ["a", "a"]}))
                .unwrap();
        assert!(dup.validate().is_err());
    }

    #[test]
    fn messages_match_the_python_judge() {
        let m = spec().messages("hi");
        let user = m[1]["content"].as_str().unwrap();
        assert!(user.contains("A) refund\nB) cancel: stop the order\nC) other"));
        assert!(user.contains("Text:\n\"\"\"\nhi\n\"\"\""));
        assert!(user.ends_with("(A-C) and nothing else."));
    }

    #[test]
    fn text_id_matches_python() {
        // python: hashlib.sha1("hello world".encode()).hexdigest()[:16]
        assert_eq!(text_id("  hello world \n"), "2aae6c35c94fcfb4");
    }

    #[test]
    fn parses_letters() {
        let top = vec![
            ("A".to_string(), (0.6f64).ln()),
            (" B".to_string(), (0.3f64).ln()),
            ("The".to_string(), -9.0),
        ];
        let p = probs_from_top_logprobs(&top, 3).unwrap();
        assert!((p[0] - 2.0 / 3.0).abs() < 1e-9 && p[2] == 0.0);
        assert_eq!(letter_from_text("B", 3), Some(1));
        assert_eq!(letter_from_text("Answer: C", 3), Some(2));
        assert_eq!(letter_from_text("Z", 3), None);
    }

    fn label(name: &str) -> Label {
        Label {
            name: name.into(),
            description: String::new(),
        }
    }

    /// The labels of the Python judge's tests (`tests/test_judge_names.py`).
    fn names() -> Vec<String> {
        (0..30)
            .map(|i| format!("intent_{i:02}"))
            .chain(["card_arrival", "card", "Sci/Tech"].map(String::from))
            .collect()
    }

    #[test]
    fn more_than_26_labels_are_asked_for_by_name() {
        let mut s = spec();
        s.labels = names().iter().map(|n| label(n)).collect();
        s.labels[1].description = "the second intent".into();
        assert!(!s.answers_with_letters());
        let m = s.messages("hi");
        assert_eq!(
            m[0]["content"],
            "You are a careful, calibrated classifier. Answer with the name of one option."
        );
        let user = m[1]["content"].as_str().unwrap();
        assert!(user.starts_with("What does the customer want?\n\nOptions:\n- intent_00\n- intent_01: the second intent\n- intent_02\n"));
        assert!(user.contains("- Sci/Tech\n\nText:\n\"\"\"\nhi\n\"\"\"\n\n"));
        assert!(user.ends_with(
            "Answer with the name of the best option, exactly as written above, and nothing else."
        ));
        assert!(!user.contains("letter") && !user.contains("A)"));
        assert_eq!(s.name_max_tokens(), 8 + "card_arrival".len());
        s.labels[0].name = "x".repeat(300);
        assert_eq!(s.name_max_tokens(), 128);
        s.labels.truncate(26);
        assert!(s.answers_with_letters());
        assert!(s.messages("hi")[1]["content"]
            .as_str()
            .unwrap()
            .ends_with("(A-Z) and nothing else."));
    }

    #[test]
    fn parses_names_like_the_python_judge() {
        let names = names();
        let name_of = |answer: &str| label_from_name(answer, &names).map(|i| names[i].as_str());
        for (answer, expected) in [
            ("intent_07", "intent_07"),
            ("INTENT_07", "intent_07"),
            ("intent 07", "intent_07"),
            ("`intent_07`.", "intent_07"),
            ("**intent_07**", "intent_07"),
            ("Answer: intent_07", "intent_07"),
            ("The answer is intent_07", "intent_07"),
            ("intent_07\nbecause the customer asks about it", "intent_07"),
            ("  intent_07  \n", "intent_07"),
            ("card_arrival", "card_arrival"),
            ("It is about card arrival.", "card_arrival"),
            ("card", "card"),
            ("Sci/Tech", "Sci/Tech"),
            ("sci/tech", "Sci/Tech"),
        ] {
            assert_eq!(name_of(answer), Ok(expected), "{answer:?}");
        }
        assert_eq!(
            name_of("either intent_07 or intent_08"),
            Err(AMBIGUOUS_NAME)
        );
        assert_eq!(name_of("no idea"), Err(UNKNOWN_NAME));
        assert_eq!(name_of(""), Err(UNKNOWN_NAME));
        assert_eq!(name_of("intent_99"), Err(UNKNOWN_NAME));
        // Names that differ only by case or `_` / space: an exact answer picks one, others are ambiguous.
        let close: Vec<String> = ["Refund", "refund", "card_arrival", "card arrival", "_"]
            .map(String::from)
            .to_vec();
        let pick = |answer: &str| label_from_name(answer, &close);
        assert_eq!(pick("refund"), Ok(1));
        assert_eq!(pick("**Refund**"), Ok(0));
        assert_eq!(pick("REFUND"), Err(AMBIGUOUS_NAME));
        assert_eq!(pick("card arrival"), Ok(3));
        assert_eq!(pick("It is a card-arrival question"), Err(AMBIGUOUS_NAME));
        assert_eq!(pick("_"), Ok(4));
        assert_eq!(pick(""), Err(UNKNOWN_NAME));
    }

    #[test]
    fn answer_probability_stops_at_the_end_of_the_name() {
        let tokens = [
            ("card".to_string(), 0.5f64.ln()),
            ("_arrival".to_string(), 0.8f64.ln()),
            (".".to_string(), 0.1f64.ln()),
        ];
        assert!((answer_probability(&tokens, "card_arrival") - 0.4).abs() < 1e-9);
        assert!((answer_probability(&tokens, "card") - 0.5).abs() < 1e-9);
    }
}
