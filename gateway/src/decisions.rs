//! Decisions compatibility uses `SGLang`'s published contract, pending an `OpenAI` specification.

use std::collections::{HashMap, HashSet};

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::{classify_inner, err, ApiError, ClassifyReq};
use crate::prompt::{Label, PromptSpec};
use crate::Shared;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    input: Value,
    questions: Vec<Question>,
    model: Option<String>,
    #[serde(default = "one")]
    temperature: f64,
    prompt_format_version: Option<u32>,
    #[serde(default)]
    images: Vec<Value>,
    #[serde(default)]
    chat_template_kwargs: HashMap<String, Value>,
    #[serde(default)]
    return_prompt_token_ids: bool,
}

fn one() -> f64 {
    1.0
}

#[derive(Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
enum Question {
    #[serde(rename = "choice")]
    Choice {
        id: String,
        question: Value,
        options: Vec<OptionIn>,
    },
    #[serde(rename = "score")]
    Score {
        id: String,
        question: Value,
        levels: Vec<Value>,
    },
    #[serde(rename = "yes_no")]
    YesNo {
        id: String,
        question: Value,
        yes: Option<Value>,
        no: Option<Value>,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OptionIn {
    name: String,
    description: Option<Value>,
}

fn text(value: &Value) -> Result<String, String> {
    match value {
        Value::String(s) if !s.trim().is_empty() => Ok(s.clone()),
        Value::Object(_) | Value::Array(_) => Ok(value.to_string()),
        _ => Err("text fields must be non-blank strings, objects or arrays".into()),
    }
}

fn description(value: Option<&Value>) -> Result<String, String> {
    match value {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(v) => text(v),
    }
}

impl Question {
    fn id(&self) -> &str {
        match self {
            Self::Choice { id, .. } | Self::Score { id, .. } | Self::YesNo { id, .. } => id,
        }
    }

    fn prompt(&self) -> Result<PromptSpec, String> {
        let (question, labels) = match self {
            Self::Choice {
                question, options, ..
            } => {
                if !(2..=26).contains(&options.len()) {
                    return Err("choice needs 2 to 26 options".into());
                }
                let mut seen = HashSet::new();
                let labels = options.iter().map(|o| {
                    if o.name.chars().any(|c| c.is_control() || c == '\u{2028}' || c == '\u{2029}')
                        || !seen.insert(o.name.trim().to_lowercase()) {
                        return Err("option names must be unique ignoring whitespace and case, with no line breaks".into());
                    }
                    Ok(Label { name: o.name.clone(), description: description(o.description.as_ref())? })
                }).collect::<Result<Vec<_>, String>>()?;
                (question, labels)
            }
            Self::Score {
                question, levels, ..
            } => {
                if !(2..=10).contains(&levels.len()) {
                    return Err("score needs 2 to 10 levels".into());
                }
                let labels = levels
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        Ok(Label {
                            name: i.to_string(),
                            description: text(v)?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                (question, labels)
            }
            Self::YesNo {
                question, yes, no, ..
            } => (
                question,
                vec![
                    Label {
                        name: "yes".into(),
                        description: description(yes.as_ref())?,
                    },
                    Label {
                        name: "no".into(),
                        description: description(no.as_ref())?,
                    },
                ],
            ),
        };
        let instructions = match self {
            Self::YesNo { .. } => format!("Is the following true? {}", text(question)?),
            _ => text(question)?,
        };
        let prompt = PromptSpec {
            instructions,
            labels,
        };
        prompt.validate()?;
        Ok(prompt)
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "score levels are bounded to 0 through 9"
    )]
    fn answer(&self, result: &Value, prompt: &PromptSpec, temperature: f64) -> Value {
        let p: Vec<f64> = prompt
            .labels
            .iter()
            .map(|l| result["probs"][&l.name].as_f64().unwrap())
            .collect();
        // Subtract the largest log probability to keep small temperatures numerically stable.
        let max = p.iter().copied().fold(0.0, f64::max);
        let scaled: Vec<f64> = p
            .iter()
            .map(|p| {
                if *p == 0.0 {
                    0.0
                } else {
                    ((p.ln() - max.ln()) / temperature).exp()
                }
            })
            .collect();
        let total: f64 = scaled.iter().sum();
        let probabilities: HashMap<_, _> = prompt
            .labels
            .iter()
            .zip(&scaled)
            .map(|(l, p)| (l.name.clone(), p / total))
            .collect();
        let kind = match self {
            Self::Choice { .. } => "choice",
            Self::Score { .. } => "score",
            Self::YesNo { .. } => "yes_no",
        };
        let mut answer = json!({"type": kind, "probabilities": probabilities});
        match self {
            Self::Choice { .. } => answer["choice"] = result["label"].clone(),
            Self::Score { .. } => {
                answer["score"] = json!(scaled
                    .iter()
                    .enumerate()
                    .map(|(i, p)| i as f64 * p / total)
                    .sum::<f64>());
            }
            Self::YesNo { .. } => (),
        }
        if let Some(mass) = result.get("label_mass") {
            answer["label_mass"] = mass.clone();
        }
        answer
    }
}

pub(crate) async fn decisions(
    State(s): State<Shared>,
    Json(req): Json<Request>,
) -> Result<Json<Value>, ApiError> {
    let bad = |msg| err(StatusCode::BAD_REQUEST, msg);
    if req.questions.is_empty() || req.questions.len() > 32 {
        return Err(bad("send 1 to 32 questions".into()));
    }
    if !req.temperature.is_finite() || req.temperature <= 0.0 {
        return Err(bad("temperature must be finite and positive".into()));
    }
    if req.prompt_format_version.is_some_and(|v| v != 1) {
        return Err(bad("only prompt_format_version 1 is supported".into()));
    }
    let input = text(&req.input).map_err(bad)?;
    if !req.images.is_empty() || !req.chat_template_kwargs.is_empty() || req.return_prompt_token_ids
    {
        return Err(bad("images, chat_template_kwargs and prompt token replay are not supported by this gateway".into()));
    }
    // Validate the entire request before logging inputs or making upstream calls.
    let mut ids = HashSet::new();
    let prompts = req
        .questions
        .iter()
        .map(|q| {
            if q.id().trim().is_empty() || !ids.insert(q.id()) {
                return Err(bad("question ids must be non-blank and unique".into()));
            }
            q.prompt().map_err(bad)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let model = req.model.filter(|m| m != "default");
    let mut answers = HashMap::new();
    let mut usage = Some((0_u64, 0_u64));
    for (question, prompt) in req.questions.iter().zip(prompts) {
        let result = classify_inner(
            s.clone(),
            ClassifyReq {
                prompt: Some(prompt.clone()),
                prompt_id: None,
                input: input.clone(),
                model: model.clone(),
            },
            true,
        )
        .await?;
        answers.insert(
            question.id(),
            question.answer(&result.0, &prompt, req.temperature),
        );
        usage = usage.and_then(|(input, output)| {
            Some((
                input.checked_add(result.0["usage"]["prompt_tokens"].as_u64()?)?,
                output.checked_add(result.0["usage"]["completion_tokens"].as_u64()?)?,
            ))
        });
    }
    let mut response = json!({"object": "decisions", "model": model.unwrap_or_else(|| "default".into()),
        "prompt_format_version": 1, "answers": answers});
    if let Some((input, output)) = usage {
        if let Some(total) = input.checked_add(output) {
            response["usage"] =
                json!({"prompt_tokens": input, "completion_tokens": output, "total_tokens": total});
        }
    }
    Ok(Json(response))
}
