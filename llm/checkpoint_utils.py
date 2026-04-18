"""Helpers for loading TransformerConfig from checkpoint metadata and validating compatibility."""

from __future__ import annotations

from llm.model import TransformerConfig


def config_from_checkpoint_meta(ckpt: dict) -> TransformerConfig | None:
    meta = ckpt.get("meta", {})
    model_meta = meta.get("model_config")
    if not model_meta:
        return None
    return TransformerConfig(
        vocab_size=model_meta["vocab_size"],
        max_seq_len=model_meta["max_seq_len"],
        d_model=model_meta["d_model"],
        n_layers=model_meta["n_layers"],
        n_heads=model_meta["n_heads"],
        d_ff=model_meta["d_ff"],
        dropout=model_meta["dropout"],
        norm_type=model_meta.get("norm_type", "layernorm"),
        mlp_type=model_meta.get("mlp_type", "relu"),
        use_rope=model_meta.get("use_rope", False),
    )


def assert_checkpoint_compatible(cfg: TransformerConfig, ckpt: dict, *, hint: str | None = None) -> None:
    meta = ckpt.get("meta", {})
    model_meta = meta.get("model_config")
    if not model_meta:
        return

    fields = [
        "vocab_size",
        "max_seq_len",
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "dropout",
        "norm_type",
        "mlp_type",
        "use_rope",
    ]
    mismatches = []
    for key in fields:
        expected = model_meta.get(key)
        actual = getattr(cfg, key)
        if expected != actual:
            mismatches.append((key, expected, actual))

    if mismatches:
        msg = ["Checkpoint/config mismatch detected:"]
        for key, expected, actual in mismatches:
            msg.append(f"  - {key}: checkpoint={expected}, arg={actual}")
        msg.append(
            hint
            or "Use args that exactly match training config for this checkpoint."
        )
        raise ValueError("\n".join(msg))
