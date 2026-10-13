//! Calls to the System One model (or LLM) that answers while no student is confident enough.

use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::ser::{SerializeMap, Serializer};
use serde::Serialize;
use serde_json::{json, Value};

use crate::db::{JudgeModel, Pricing, SystemOneModel};
use crate::prompt::{
    answer_probability, label_from_name, letter_from_text, probs_from_top_logprobs, Label,
    PromptSpec,
};

#[derive(Debug, Clone)]
pub struct Answer {
    pub label: usize,
    pub confidence: Option<f64>,
    pub probabilities: Option<Vec<f64>>,
    pub label_mass: Option<f64>,
    pub usage: Option<Value>,
    /// The model as the answer names it: what an alias such as `-latest` resolved to, when the
    /// endpoint says.
    pub model: Option<String>,
}

/// The `model` an endpoint's answer names, when it names one.
fn answered_model(v: &Value) -> Option<String> {
    v["model"]
        .as_str()
        .filter(|m| !m.is_empty())
        .map(String::from)
}

fn with_auth(req: reqwest::RequestBuilder, key_env: Option<&str>) -> reqwest::RequestBuilder {
    match key_env.and_then(|k| std::env::var(k).ok()) {
        Some(key) if !key.is_empty() => req.bearer_auth(key),
        _ => req,
    }
}

pub async fn classify(
    client: &reqwest::Client,
    m: &SystemOneModel,
    spec: &PromptSpec,
    text: &str,
) -> Result<Answer> {
    match m.kind.as_str() {
        "openai_chat" => openai_chat(client, m, spec, text).await,
        "decision_http" => decision_http(client, m, spec, text).await,
        "systemone" => systemone(client, m, spec, text).await,
        "openai_decisions" => openai_decisions(client, m, spec, text).await,
        other => {
            bail!("unknown System One kind '{other}'")
        }
    }
}

/// Audit against the configured training judge, independently of the serving fallback.
pub async fn judge(
    client: &reqwest::Client,
    judge: &JudgeModel,
    spec: &PromptSpec,
    text: &str,
) -> Result<Answer> {
    judge.check().map_err(|e| anyhow!(e))?;
    let model = SystemOneModel {
        name: judge.name.clone(),
        kind: match judge.mode.as_str() {
            "systemone" => "systemone",
            "openai_decisions" => "openai_decisions",
            _ => "openai_chat",
        }
        .into(),
        base_url: judge.base_url.clone(),
        model: judge.model.clone(),
        api_key_env: judge.api_key_env.clone(),
        timeout_ms: 30_000,
        logprobs: true,
        enabled: true,
        escalate_to: None,
        escalate_below: None,
        pricing: Pricing::default(),
    };
    if judge.mode != "sample" {
        return classify(client, &model, spec, text).await;
    }
    // Vote shares, including endpoints that ignore n and return one answer per call.
    let samples = usize::try_from(judge.samples.clamp(1, 100))?;
    let mut votes = vec![0usize; spec.labels.len()];
    let mut seen = 0;
    let mut model = None;
    while seen < samples {
        let body = json!({"model": judge.model, "messages": spec.messages(text),
            "max_tokens": if spec.answers_with_letters() { 4 } else { spec.name_max_tokens() },
            "temperature": 1.0, "n": samples - seen});
        let response: Value = with_auth(
            client
                .post(format!(
                    "{}/chat/completions",
                    judge.base_url.trim_end_matches('/')
                ))
                .json(&body)
                .timeout(Duration::from_secs(30)),
            judge.api_key_env.as_deref(),
        )
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
        let choices = response["choices"]
            .as_array()
            .filter(|c| !c.is_empty())
            .ok_or_else(|| anyhow!("judge returned no choices"))?;
        model = model.or_else(|| answered_model(&response));
        for choice in choices.iter().take(samples - seen) {
            let text = choice["message"]["content"].as_str().unwrap_or_default();
            let label = if spec.answers_with_letters() {
                letter_from_text(text, spec.labels.len())
            } else {
                label_from_name(text, &spec.label_names()).ok()
            };
            if let Some(label) = label {
                votes[label] += 1;
            }
            seen += 1;
        }
    }
    let total: usize = votes.iter().sum();
    if total == 0 {
        bail!("judge returned no label");
    }
    let (label, _) =
        votes.iter().copied().enumerate().fold(
            (0, 0),
            |best, item| if item.1 > best.1 { item } else { best },
        );
    Ok(Answer {
        label,
        confidence: None,
        probabilities: None,
        label_mass: None,
        usage: None,
        model,
    })
}

async fn openai_chat(
    client: &reqwest::Client,
    m: &SystemOneModel,
    spec: &PromptSpec,
    text: &str,
) -> Result<Answer> {
    let k = spec.labels.len();
    // Up to 26 labels: one letter, and its top logprobs. Above: the label's name.
    let letters = spec.answers_with_letters();
    let max_tokens = if letters { 1 } else { spec.name_max_tokens() };
    let mut body = json!({"model": m.model, "messages": spec.messages(text), "max_tokens": max_tokens, "temperature": 0});
    if m.logprobs {
        body["logprobs"] = json!(true);
        if letters {
            body["top_logprobs"] = json!(20.min(k + 5));
        }
    }
    let url = format!("{}/chat/completions", m.base_url.trim_end_matches('/'));
    let req = with_auth(
        client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_millis(m.timeout_ms)),
        m.api_key_env.as_deref(),
    );
    let resp = req.send().await.with_context(|| format!("calling {url}"))?;
    let status = resp.status();
    let v: Value = resp
        .json()
        .await
        .with_context(|| format!("{url} answered {status} with non-JSON"))?;
    if !status.is_success() {
        bail!(
            "{url} answered {status}: {}",
            v.to_string().chars().take(300).collect::<String>()
        );
    }
    let choice = &v["choices"][0];
    let usage = openai_usage(&v["usage"]);
    let model = answered_model(&v);
    if !letters {
        return name_answer(spec, choice).map(|a| Answer { usage, model, ..a });
    }
    if let Some(top) = choice["logprobs"]["content"][0]["top_logprobs"].as_array() {
        let pairs: Vec<(String, f64)> = top
            .iter()
            .filter_map(|t| Some((t["token"].as_str()?.to_string(), t["logprob"].as_f64()?)))
            .collect();
        if let Some(p) = probs_from_top_logprobs(&pairs, k) {
            let (i, conf) = p
                .iter()
                .copied()
                .enumerate()
                .fold((0, f64::MIN), |a, (i, x)| if x > a.1 { (i, x) } else { a });
            return Ok(Answer {
                label: i,
                confidence: Some(conf),
                probabilities: Some(p),
                label_mass: None,
                usage,
                model,
            });
        }
    }
    let content = choice["message"]["content"].as_str().unwrap_or("");
    let label = letter_from_text(content, k)
        .ok_or_else(|| anyhow!("no option letter in answer {content:?}"))?;
    Ok(Answer {
        label,
        confidence: None,
        probabilities: None,
        label_mass: None,
        usage,
        model,
    })
}

/// An answer that names a label: that label, and as confidence the probability of the tokens
/// that named it, when the endpoint returned logprobs.
fn name_answer(spec: &PromptSpec, choice: &Value) -> Result<Answer> {
    let content = choice["message"]["content"].as_str().unwrap_or("");
    let names = spec.label_names();
    let label = label_from_name(content, &names).map_err(|why| {
        let shown: String = content.chars().take(200).collect();
        anyhow!("the answer {shown:?} {why}")
    })?;
    let confidence = choice["logprobs"]["content"]
        .as_array()
        .filter(|tokens| !tokens.is_empty())
        .map(|tokens| {
            let pairs: Vec<(String, f64)> = tokens
                .iter()
                .map(|t| {
                    let token = t["token"].as_str().unwrap_or("").to_string();
                    (token, t["logprob"].as_f64().unwrap_or(0.0))
                })
                .collect();
            answer_probability(&pairs, &names[label]).min(1.0)
        });
    Ok(Answer {
        label,
        confidence,
        probabilities: None,
        label_mass: None,
        usage: None,
        model: None,
    })
}

async fn decision_http(
    client: &reqwest::Client,
    m: &SystemOneModel,
    spec: &PromptSpec,
    text: &str,
) -> Result<Answer> {
    let body = json!({"model": m.model, "instructions": spec.instructions, "labels": spec.label_names(), "input": text});
    let req = with_auth(
        client
            .post(&m.base_url)
            .json(&body)
            .timeout(Duration::from_millis(m.timeout_ms)),
        m.api_key_env.as_deref(),
    );
    let resp = req
        .send()
        .await
        .with_context(|| format!("calling {}", m.base_url))?;
    let status = resp.status();
    let v: Value = resp.json().await.context("non-JSON answer")?;
    if !status.is_success() {
        bail!("{} answered {status}", m.base_url);
    }
    let name = v["label"]
        .as_str()
        .ok_or_else(|| anyhow!("answer has no 'label'"))?;
    let label = spec
        .labels
        .iter()
        .position(|l| l.name == name)
        .ok_or_else(|| anyhow!("answer label '{name}' is not one of the prompt's labels"))?;
    Ok(Answer {
        label,
        confidence: v["confidence"].as_f64(),
        probabilities: None,
        label_mass: None,
        usage: None,
        model: answered_model(&v),
    })
}

/// A `systemone` request: the input is the state, and the prompt one Choice question, `label`.
#[derive(Serialize)]
struct SystemOneRequest<'a> {
    state: &'a str,
    questions: Questions<'a>,
    #[serde(skip_serializing_if = "str::is_empty")]
    model: &'a str,
}

#[derive(Serialize)]
struct Questions<'a> {
    label: Choice<'a>,
}

#[derive(Serialize)]
struct Choice<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    instructions: &'a str,
    criteria: Criteria<'a>,
}

/// The labels as `{name: description}`, in the prompt's order: the model sees the options in the
/// order they are listed, and a `serde_json::Value` object would sort them. A label without a
/// description gets `""`, not null: Fastino's GLiDE documents only strings there, and Decider
/// shows the name alone for either.
struct Criteria<'a>(&'a [Label]);

impl Serialize for Criteria<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(self.0.len()))?;
        for l in self.0 {
            map.serialize_entry(&l.name, &l.description)?;
        }
        map.end()
    }
}

/// TypeSafe's `POST /v1/systemone`, which Decider (huggingface.co/Mapika/decider-2b) and Fastino's
/// GLiDE (`https://api.fastino.ai`, model `fastino/glide`) serve:
/// {state, questions: {id: {type: "choice", instructions, criteria}}} ->
/// {answers: {id: {choice, confidence, probabilities: {name: p}}}}. The confidence is the
/// probability of the chosen label, as with `openai_chat`, or the answer's `confidence` when it
/// has no probabilities (TypeSafe's is rescaled, GLiDE's the margin over the runner-up).
async fn systemone(
    client: &reqwest::Client,
    m: &SystemOneModel,
    spec: &PromptSpec,
    text: &str,
) -> Result<Answer> {
    let body = SystemOneRequest {
        state: text,
        questions: Questions {
            label: Choice {
                kind: "choice",
                instructions: spec.instructions.trim(),
                criteria: Criteria(&spec.labels),
            },
        },
        model: &m.model,
    };
    let url = format!("{}/v1/systemone", m.base_url.trim_end_matches('/'));
    let req = with_auth(
        client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_millis(m.timeout_ms)),
        m.api_key_env.as_deref(),
    );
    let resp = req.send().await.with_context(|| format!("calling {url}"))?;
    let status = resp.status();
    let v: Value = resp
        .json()
        .await
        .with_context(|| format!("{url} answered {status} with non-JSON"))?;
    if !status.is_success() {
        bail!(
            "{url} answered {status}: {}",
            v.to_string().chars().take(300).collect::<String>()
        );
    }
    let answer = &v["answers"]["label"];
    let name = answer["choice"]
        .as_str()
        .ok_or_else(|| anyhow!("answer has no 'answers.label.choice'"))?;
    let label = spec
        .labels
        .iter()
        .position(|l| l.name == name)
        .ok_or_else(|| anyhow!("answer label '{name}' is not one of the prompt's labels"))?;
    Ok(Answer {
        label,
        confidence: answer["probabilities"][name]
            .as_f64()
            .or_else(|| answer["confidence"].as_f64()),
        probabilities: read_probabilities(answer, spec).ok(),
        label_mass: answer["x_label_mass"].as_f64(),
        usage: systemone_usage(&v["usage"]),
        model: answered_model(&v),
    })
}

/// A complete distribution in prompt order. Missing or invalid values must not become soft labels.
pub(crate) fn read_probabilities(answer: &Value, spec: &PromptSpec) -> Result<Vec<f64>> {
    let map = answer["probabilities"]
        .as_object()
        .context("missing probabilities")?;
    if map.len() != spec.labels.len() {
        bail!("probabilities do not match the prompt's labels");
    }
    let p: Vec<f64> = spec
        .labels
        .iter()
        .map(|l| {
            map.get(&l.name)
                .and_then(Value::as_f64)
                .filter(|p| p.is_finite() && (0.0..=1.0).contains(p))
                .ok_or_else(|| anyhow!("invalid probability for '{}'", l.name))
        })
        .collect::<Result<_>>()?;
    let total: f64 = p.iter().sum();
    if (total - 1.0).abs() > 1e-4 {
        bail!("probabilities must sum to one");
    }
    Ok(p.into_iter().map(|p| p / total).collect())
}

async fn openai_decisions(
    client: &reqwest::Client,
    m: &SystemOneModel,
    spec: &PromptSpec,
    text: &str,
) -> Result<Answer> {
    if spec.labels.len() > 26 {
        bail!("openai_decisions supports at most 26 choice options");
    }
    let url = format!("{}/decisions", m.base_url.trim_end_matches('/'));
    let model = if m.model.is_empty() {
        "default"
    } else {
        &m.model
    };
    let body = json!({"model": model, "input": text, "questions": [{
        "id": "label", "type": "choice", "question": spec.instructions.trim(), "options": spec.labels
    }]});
    let response = with_auth(
        client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_millis(m.timeout_ms)),
        m.api_key_env.as_deref(),
    )
    .send()
    .await
    .with_context(|| format!("calling {url}"))?
    .error_for_status()?;
    let data: Value = response.json().await.context("non-JSON Decisions answer")?;
    let answer = &data["answers"]["label"];
    if answer["type"] != "choice" {
        bail!("Decisions answer is not a choice");
    }
    let name = answer["choice"]
        .as_str()
        .context("missing Decisions choice")?;
    let label = spec
        .labels
        .iter()
        .position(|l| l.name == name)
        .context("Decisions choice is not one of the prompt's labels")?;
    let probabilities = read_probabilities(answer, spec)?;
    if probabilities
        .iter()
        .any(|p| *p > probabilities[label] + 1e-8)
    {
        bail!("Decisions choice is not a most probable label");
    }
    let label_mass = answer["label_mass"]
        .as_f64()
        .filter(|p| p.is_finite() && (0.0..=1.0).contains(p));
    if answer.get("label_mass").is_some() && label_mass.is_none() {
        bail!("invalid Decisions label_mass");
    }
    Ok(Answer {
        label,
        confidence: Some(probabilities[label]),
        probabilities: Some(probabilities),
        label_mass,
        usage: openai_usage(&data["usage"]),
        model: answered_model(&data),
    })
}

fn systemone_usage(usage: &Value) -> Option<Value> {
    let input = usage["input_tokens"].as_u64()?;
    let output = usage["output_tokens"].as_u64()?;
    Some(
        json!({"prompt_tokens": input, "completion_tokens": output, "total_tokens": input.checked_add(output)?}),
    )
}

fn openai_usage(usage: &Value) -> Option<Value> {
    let input = usage["prompt_tokens"].as_u64()?;
    let output = usage["completion_tokens"].as_u64()?;
    Some(
        json!({"prompt_tokens": input, "completion_tokens": output, "total_tokens": input.checked_add(output)?}),
    )
}

/// The input tokens of a call whose model doesn't report them: about four characters a token
/// for the text, the instructions and the labels with their descriptions, which every kind sends.
/// It leaves out what a provider adds around them, so it errs low.
pub fn estimated_input_tokens(spec: &PromptSpec, text: &str) -> u64 {
    let chars = text.chars().count()
        + spec.instructions.chars().count()
        + spec
            .labels
            .iter()
            .map(|l| l.name.chars().count() + l.description.chars().count())
            .sum::<usize>();
    u64::try_from(chars.div_ceil(4)).unwrap_or(u64::MAX)
}

/// What an answered call cost at the model's pricing: the tokens it reported, or the estimate.
pub fn call_cost(m: &SystemOneModel, spec: &PromptSpec, text: &str, answer: &Answer) -> f64 {
    let reported = answer.usage.as_ref().and_then(|u| {
        Some((
            u["prompt_tokens"].as_u64()?,
            u["completion_tokens"].as_u64()?,
        ))
    });
    let (input, output) = reported.unwrap_or_else(|| (estimated_input_tokens(spec, text), 0));
    m.pricing.cost(input, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> PromptSpec {
        // 12 + 3 + 6 + 15 = 36 characters, without the text.
        serde_json::from_value(json!({"instructions": "Which topic?", "labels": ["tax",
            {"name": "travel", "description": "trips and visas"}]}))
        .unwrap()
    }

    fn model() -> SystemOneModel {
        serde_json::from_value(json!({"kind": "systemone", "base_url": "http://x"})).unwrap()
    }

    fn answer(usage: Option<Value>) -> Answer {
        Answer {
            label: 0,
            confidence: None,
            probabilities: None,
            label_mass: None,
            usage,
            model: None,
        }
    }

    #[test]
    fn a_call_costs_its_reported_tokens_or_an_estimate() {
        let text = "a".repeat(964);
        assert_eq!(estimated_input_tokens(&spec(), &text), 250); // 1,000 characters
        let m = model();
        let estimate = call_cost(&m, &spec(), &text, &answer(None));
        assert!((estimate - 250.0 * 0.042 / 1e6).abs() < 1e-15);
        // Reported tokens win; Jev's output tokens are free.
        let reported = json!({"prompt_tokens": 1000, "completion_tokens": 7, "total_tokens": 1007});
        let cost = call_cost(&m, &spec(), &text, &answer(Some(reported.clone())));
        assert!((cost - 1000.0 * 0.042 / 1e6).abs() < 1e-15);
        let priced = SystemOneModel {
            pricing: Pricing {
                usd_per_call: 0.5,
                usd_per_million_input_tokens: 0.0,
                usd_per_million_output_tokens: 1e6,
            },
            ..m
        };
        assert!((call_cost(&priced, &spec(), &text, &answer(Some(reported))) - 7.5).abs() < 1e-9);
    }

    #[test]
    fn openai_and_systemone_usage_read_the_same() {
        let openai = openai_usage(&json!({"prompt_tokens": 10, "completion_tokens": 1}));
        let systemone = systemone_usage(&json!({"input_tokens": 10, "output_tokens": 1}));
        assert_eq!(openai, systemone);
        assert_eq!(openai_usage(&json!({"total_tokens": 11})), None);
    }
}
