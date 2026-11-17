#!/usr/bin/env python3
"""Build a Lebai feature-configured view of official SmolVLA weights."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "pretrained"
SOURCE = ROOT / "smolvla_base"
TARGET = ROOT / "smolvla_base_lebai"


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"Refusing to overwrite {TARGET}")
    weights = SOURCE / "model.safetensors"
    if not weights.is_file() or weights.stat().st_size < 900_000_000:
        raise FileNotFoundError("Official model.safetensors download is incomplete")
    original = json.loads((SOURCE / "config.json").read_text())
    assert original["type"] == "smolvla" and original["max_state_dim"] == 32
    assert original["max_action_dim"] == 32
    original["input_features"] = {
        "observation.state": {"type": "STATE", "shape": [31]},
        "observation.images.camera1": {"type": "VISUAL", "shape": [3, 180, 320]},
        "observation.images.camera2": {"type": "VISUAL", "shape": [3, 180, 320]},
    }
    original["output_features"] = {"action": {"type": "ACTION", "shape": [8]}}
    original["push_to_hub"] = False
    original["n_action_steps"] = 10
    # Every VLM tensor is replaced by the official SmolVLA safetensors below.
    # Avoid a redundant second download of SmolVLM2 base weights at construction.
    original["load_vlm_weights"] = False
    TARGET.mkdir()
    (TARGET / "config.json").write_text(json.dumps(original, indent=2) + "\n")
    for name in (
        "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        os.link(SOURCE / name, TARGET / name)
    hasher = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    (TARGET / "provenance.json").write_text(json.dumps({
        "source_repo": "lerobot/smolvla_base",
        "source_commit": "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6",
        "weights_sha256": digest,
        "changes": "Feature config 31D state, 2 cameras, 8D action; n_action_steps=10; skip redundant VLM initialization weights; no SmolVLA weight modification",
    }, indent=2) + "\n")
    print(TARGET, "weights_sha256", digest)


if __name__ == "__main__":
    main()
