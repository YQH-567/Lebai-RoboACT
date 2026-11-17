#!/usr/bin/env python3
"""Make a 31D LeRobot v3 view of the 25Hz ACT data without re-encoding videos."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/lerobot/lebai_lm3_pick_place_25hz"
TARGET = ROOT / "datasets/lerobot/lebai_lm3_pick_place_25hz_smolvla31"
STATE = "observation.state"


def trim_table(source: Path, target: Path, episode_meta: bool = False) -> None:
    table = pq.read_table(source)
    if episode_meta:
        for name in table.column_names:
            if name.startswith("stats/observation.state/") and not name.endswith("/count"):
                column = pa.array([values[:31] for values in table[name].to_pylist()], type=table[name].type)
                table = table.set_column(table.schema.get_field_index(name), name, column)
    else:
        values = table[STATE].combine_chunks().values.to_numpy().reshape(-1, 34)[:, :31].copy()
        column = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), 31)
        table = table.set_column(table.schema.get_field_index(STATE), STATE, column)
        metadata = dict(table.schema.metadata or {})
        if b"huggingface" in metadata:
            hf = json.loads(metadata[b"huggingface"])
            hf["info"]["features"][STATE]["length"] = 31
            metadata[b"huggingface"] = json.dumps(hf).encode()
            table = table.replace_schema_metadata(metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target)


def prepare_split(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    info = json.loads((source / "meta/info.json").read_text())
    assert info["features"][STATE]["shape"] == [34]
    assert len(info["features"][STATE]["names"]) == 34
    assert info["features"]["action"]["names"] == ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]
    target.mkdir(parents=True)
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if rel == Path("meta/info.json") or rel == Path("meta/stats.json"):
            continue
        if rel.parts[0] == "data" and path.suffix == ".parquet":
            trim_table(path, dest)
        elif rel.parts[:2] == ("meta", "episodes") and path.suffix == ".parquet":
            trim_table(path, dest, episode_meta=True)
        else:
            os.link(path, dest)
    info["features"][STATE]["shape"] = [31]
    info["features"][STATE]["names"] = info["features"][STATE]["names"][:31]
    (target / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    stats = json.loads((source / "meta/stats.json").read_text())
    for key, values in stats[STATE].items():
        if key != "count":
            assert len(values) == 34, key
            stats[STATE][key] = values[:31]
    (target / "meta/stats.json").write_text(json.dumps(stats, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--target", type=Path, default=TARGET)
    a = p.parse_args()
    if a.target.exists():
        raise FileExistsError(f"Refusing to overwrite {a.target}")
    for split in ("train", "val"):
        prepare_split(a.source / split, a.target / split)
    manifest = a.source / "split_manifest.json"
    if manifest.exists():
        os.link(manifest, a.target / manifest.name)
    print(f"Prepared {a.target}: train and val, 31D state, original videos hard-linked")


if __name__ == "__main__":
    main()
