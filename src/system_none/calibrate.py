"""Temperature scaling and calibration metrics (Guo et al., ICML 2017)."""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def soft_nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    """Cross-entropy of softmax(logits / T) against soft (or one-hot) targets."""
    logp = np.log(np.clip(softmax(logits, temperature), 1e-12, 1.0))
    return float(-(targets * logp).sum(axis=-1).mean())


def fit_temperature(
    logits: np.ndarray, targets: np.ndarray, lo: float = 0.05, hi: float = 20.0, iters: int = 80
) -> float:
    """One number, fitted on held-out data: minimize NLL over log T with golden-section search."""
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = soft_nll(logits, targets, np.exp(c)), soft_nll(logits, targets, np.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = soft_nll(logits, targets, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = soft_nll(logits, targets, np.exp(d))
    return float(np.exp((a + b) / 2))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=-1)
    correct = (probs.argmax(axis=-1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict]:
    conf = probs.max(axis=-1)
    correct = (probs.argmax(axis=-1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append(
                {
                    "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                    "n": int(m.sum()),
                    "confidence": round(float(conf[m].mean()), 4),
                    "accuracy": round(float(correct[m].mean()), 4),
                }
            )
    return rows
