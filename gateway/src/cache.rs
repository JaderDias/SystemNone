//! An optional cache of System One's answers, in memory and per gateway instance.
//!
//! A request whose prompt and input (ignoring surrounding whitespace) System One answered before,
//! when the same System One model would answer it again, gets that answer back without a call.
//! `position` puts it anywhere in the cascade: before the student (`before_student`, the first
//! layer), between the student and System One (`after_student`, the default: asked only when no
//! confident student answers), or between System One and the stronger model it escalates to
//! (`before_escalation`: System One answers, and an escalation's answer comes from the cache when
//! the stronger model gave it before, so only that call is saved).
//! When it is full, `policy` says which answer goes: the least recently used (`lru`), or the least
//! frequently used, the least recently used among those (`lfu`). With `ttl_secs`, answers older
//! than that are not used. Only System One's answers are kept: a student's cost nothing to
//! compute again, and change with the student.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// The most answers a cache may hold.
pub const MAX_CAPACITY: usize = 10_000_000;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CacheSettings {
    pub enabled: bool,
    /// `before_student` (the first layer), `after_student` (between the student and System One)
    /// or `before_escalation` (between System One and the model it escalates to).
    pub position: String,
    /// `lru` or `lfu`.
    pub policy: String,
    /// The most answers it holds.
    pub capacity: usize,
    /// Answers older than this many seconds are not used; 0: they never expire.
    pub ttl_secs: u64,
}

impl Default for CacheSettings {
    fn default() -> Self {
        CacheSettings {
            enabled: false,
            position: "after_student".into(),
            policy: "lru".into(),
            capacity: 10_000,
            ttl_secs: 0,
        }
    }
}

impl CacheSettings {
    pub fn check(&self) -> Result<(), String> {
        if !matches!(
            self.position.as_str(),
            "after_student" | "before_student" | "before_escalation"
        ) {
            return Err(
                "cache position must be before_student, after_student or before_escalation".into(),
            );
        }
        if !matches!(self.policy.as_str(), "lru" | "lfu") {
            return Err("cache policy must be lru or lfu".into());
        }
        if !(1..=MAX_CAPACITY).contains(&self.capacity) {
            return Err(format!("cache capacity must be 1 to {MAX_CAPACITY}"));
        }
        Ok(())
    }

    pub fn before_student(&self) -> bool {
        self.enabled && self.position == "before_student"
    }

    pub fn after_student(&self) -> bool {
        self.enabled && self.position == "after_student"
    }

    pub fn before_escalation(&self) -> bool {
        self.enabled && self.position == "before_escalation"
    }
}

/// A System One answer, as the cache gives it back.
#[derive(Clone, Debug, PartialEq)]
pub struct CachedAnswer {
    pub label: usize,
    pub confidence: Option<f64>,
    pub probabilities: Option<Vec<f64>>,
    pub label_mass: Option<f64>,
    /// The model that answered: an escalation's, when the request escalated.
    pub model: String,
}

struct Entry {
    answer: CachedAnswer,
    stored: Instant,
    /// Where it sorts in `Inner::order`: (uses with `lfu`, else 0; when it was last used).
    rank: (u64, u64),
}

#[derive(Default)]
struct Inner {
    settings: CacheSettings,
    entries: HashMap<Arc<str>, Entry>,
    /// The answers by rank: the first one goes when the cache is full.
    order: BTreeMap<(u64, u64), Arc<str>>,
    tick: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl Inner {
    fn touch(&mut self, key: &Arc<str>, now_used: bool) {
        self.tick += 1;
        let (lfu, tick) = (self.settings.policy == "lfu", self.tick);
        let Some(e) = self.entries.get_mut(key) else {
            return;
        };
        self.order.remove(&e.rank);
        let uses = if lfu {
            e.rank.0 + u64::from(now_used)
        } else {
            0
        };
        e.rank = (uses, tick);
        self.order.insert(e.rank, key.clone());
    }

    fn remove(&mut self, key: &str) {
        if let Some(e) = self.entries.remove(key) {
            self.order.remove(&e.rank);
        }
    }

    fn expired(&self, e: &Entry) -> bool {
        self.settings.ttl_secs > 0
            && e.stored.elapsed() > Duration::from_secs(self.settings.ttl_secs)
    }
}

pub struct AnswerCache {
    inner: Mutex<Inner>,
}

impl Default for AnswerCache {
    fn default() -> Self {
        Self::new(&CacheSettings::default())
    }
}

impl AnswerCache {
    pub fn new(settings: &CacheSettings) -> Self {
        AnswerCache {
            inner: Mutex::new(Inner {
                settings: settings.clone(),
                ..Inner::default()
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn key(prompt_id: &str, model: &str, text: &str) -> String {
        format!("{prompt_id}\u{0}{model}\u{0}{}", text.trim())
    }

    /// Take new settings. Turning it off, or changing its policy, capacity or expiry, empties
    /// it; a change of position keeps its answers.
    pub fn configure(&self, settings: &CacheSettings) {
        let mut inner = self.lock();
        let old = std::mem::replace(&mut inner.settings, settings.clone());
        let same = CacheSettings {
            position: settings.position.clone(),
            ..old
        } == *settings;
        if !same || !settings.enabled {
            inner.entries.clear();
            inner.order.clear();
        }
    }

    /// The answer `model` gave for `text` under `prompt_id`, unless there is none, it expired,
    /// or the cache is off.
    pub fn get(&self, prompt_id: &str, model: &str, text: &str) -> Option<CachedAnswer> {
        let mut inner = self.lock();
        if !inner.settings.enabled {
            return None;
        }
        let key = Self::key(prompt_id, model, text);
        let Some((key, expired)) = inner
            .entries
            .get_key_value(key.as_str())
            .map(|(k, e)| (k.clone(), inner.expired(e)))
        else {
            inner.misses += 1;
            return None;
        };
        if expired {
            inner.remove(&key);
            inner.misses += 1;
            return None;
        }
        inner.hits += 1;
        inner.touch(&key, true);
        inner.entries.get(&key).map(|e| e.answer.clone())
    }

    /// Keep `model`'s answer for `text` under `prompt_id`, making room when full.
    pub fn put(&self, prompt_id: &str, model: &str, text: &str, answer: CachedAnswer) {
        let mut inner = self.lock();
        if !inner.settings.enabled {
            return;
        }
        let key: Arc<str> = Self::key(prompt_id, model, text).into();
        if let Some(e) = inner.entries.get_mut(&key) {
            e.answer = answer;
            e.stored = Instant::now();
            inner.touch(&key, false);
            return;
        }
        while inner.entries.len() >= inner.settings.capacity {
            let Some((_, oldest)) = inner.order.pop_first() else {
                break;
            };
            inner.entries.remove(&oldest);
            inner.evictions += 1;
        }
        inner.entries.insert(
            key.clone(),
            Entry {
                answer,
                stored: Instant::now(),
                rank: (0, 0),
            },
        );
        inner.touch(&key, true);
    }

    /// Forget every answer, as when a System One model changes.
    pub fn clear(&self) {
        let mut inner = self.lock();
        inner.entries.clear();
        inner.order.clear();
    }

    /// Its settings, how many answers it holds, and its hits, misses and evictions since the
    /// gateway started.
    pub fn stats(&self) -> Value {
        let inner = self.lock();
        json!({
            "enabled": inner.settings.enabled,
            "position": inner.settings.position,
            "policy": inner.settings.policy,
            "capacity": inner.settings.capacity,
            "ttl_secs": inner.settings.ttl_secs,
            "entries": inner.entries.len(),
            "hits": inner.hits,
            "misses": inner.misses,
            "evictions": inner.evictions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(policy: &str, capacity: usize) -> CacheSettings {
        CacheSettings {
            enabled: true,
            policy: policy.into(),
            capacity,
            ..CacheSettings::default()
        }
    }

    fn answer(label: usize) -> CachedAnswer {
        CachedAnswer {
            label,
            confidence: Some(0.9),
            probabilities: None,
            label_mass: None,
            model: "s1".into(),
        }
    }

    fn has(c: &AnswerCache, text: &str) -> bool {
        c.lock()
            .entries
            .contains_key(AnswerCache::key("p", "s1", text).as_str())
    }

    #[test]
    fn answers_come_back_per_prompt_model_and_trimmed_text() {
        let c = AnswerCache::new(&settings("lru", 10));
        c.put("p", "s1", "refund me", answer(1));
        assert_eq!(c.get("p", "s1", "  refund me\n"), Some(answer(1)));
        assert_eq!(c.get("p", "other-model", "refund me"), None);
        assert_eq!(c.get("q", "s1", "refund me"), None);
        let stats = c.stats();
        assert_eq!(
            (stats["hits"].as_u64(), stats["misses"].as_u64()),
            (Some(1), Some(2))
        );
    }

    #[test]
    fn lru_drops_the_least_recently_used() {
        let c = AnswerCache::new(&settings("lru", 2));
        c.put("p", "s1", "a", answer(0));
        c.put("p", "s1", "b", answer(1));
        assert!(c.get("p", "s1", "a").is_some()); // b is now the least recently used
        c.put("p", "s1", "c", answer(2));
        assert!(has(&c, "a") && !has(&c, "b") && has(&c, "c"));
        assert_eq!(c.stats()["evictions"], 1);
    }

    #[test]
    fn lfu_drops_the_least_frequently_used() {
        let c = AnswerCache::new(&settings("lfu", 2));
        c.put("p", "s1", "a", answer(0));
        c.put("p", "s1", "b", answer(1));
        for _ in 0..3 {
            assert!(c.get("p", "s1", "a").is_some());
        }
        assert!(c.get("p", "s1", "b").is_some()); // b is the most recent, but used less
        c.put("p", "s1", "c", answer(2));
        assert!(has(&c, "a") && !has(&c, "b") && has(&c, "c"));
        // Among the least used, the least recently used goes.
        c.put("p", "s1", "d", answer(3));
        assert!(has(&c, "a") && !has(&c, "c") && has(&c, "d"));
    }

    #[test]
    fn expired_answers_are_not_used() {
        let c = AnswerCache::new(&CacheSettings {
            ttl_secs: 60,
            ..settings("lru", 10)
        });
        c.put("p", "s1", "a", answer(0));
        assert!(c.get("p", "s1", "a").is_some());
        c.lock()
            .entries
            .values_mut()
            .for_each(|e| e.stored -= Duration::from_secs(61));
        assert_eq!(c.get("p", "s1", "a"), None);
        assert!(!has(&c, "a"));
    }

    #[test]
    fn settings_are_checked_and_some_changes_empty_it() {
        let c = AnswerCache::new(&settings("lru", 10));
        c.put("p", "s1", "a", answer(0));
        // Moving it before the student keeps its answers; a new policy doesn't.
        c.configure(&CacheSettings {
            position: "before_student".into(),
            ..settings("lru", 10)
        });
        assert!(has(&c, "a"));
        c.configure(&settings("lfu", 10));
        assert!(!has(&c, "a"));
        // Off, it neither keeps nor gives answers.
        c.configure(&CacheSettings::default());
        c.put("p", "s1", "a", answer(0));
        assert_eq!(c.get("p", "s1", "a"), None);
        assert_eq!(c.stats()["misses"], 0);

        assert!(CacheSettings::default().check().is_ok());
        for position in ["before_student", "after_student", "before_escalation"] {
            let s = CacheSettings {
                position: position.into(),
                ..settings("lru", 10)
            };
            assert!(s.check().is_ok(), "{position}");
            assert_eq!(
                [s.before_student(), s.after_student(), s.before_escalation()]
                    .iter()
                    .filter(|&&b| b)
                    .count(),
                1
            );
        }
        for bad in [
            CacheSettings {
                policy: "fifo".into(),
                ..CacheSettings::default()
            },
            CacheSettings {
                position: "last".into(),
                ..CacheSettings::default()
            },
            CacheSettings {
                capacity: 0,
                ..CacheSettings::default()
            },
        ] {
            assert!(bad.check().is_err(), "{bad:?}");
        }
    }
}
