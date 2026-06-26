#!/usr/bin/env python
"""
Build a self-contained data folder for the local verification tool.

Reads a finished pipeline run (under ``work_dir``) and writes everything
``verify.html`` needs into ``work_dir/verify_data/`` — no server, no GCS:

    verify_data/
        verify.html, index.html      (copied from this folder)
        data.json                    tree + call list + modes
        calls/<call_key>.json        per-call detail (species, probs, clips, node_vals)
        audio/<example_id>.<ext>     copied iff the metadata audio_fp is a local file

Serve it with:  python -m http.server --directory work/verify_data 8000

Usage:
    python verify/build_verify_data.py --config configs/example.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from commoncalls import io
from commoncalls.config import PipelineConfig

HERE = Path(__file__).resolve().parent


def _mode_label(name: str) -> str:
    if name.startswith("continuous"):
        return name.replace("continuous_ancestor", "continuous (ancestor ") + ")"
    return name  # binary_leafX_ancestorY is already readable enough


def _resolve_audio(audio_fp: str | None, audio_root: Path | None) -> Path | None:
    if not audio_fp or "://" in str(audio_fp):
        return None  # remote audio can't be bundled locally
    p = Path(audio_fp)
    if not p.is_absolute() and audio_root is not None:
        p = audio_root / p
    return p if p.exists() else None


def build(config: PipelineConfig, out_dir: Path, audio_root: Path | None, copy_audio: bool) -> None:
    call_probs = io.load_json(config.call_probabilities_path)  # {species: {call_key: prob}}
    summary = io.load_json(config.results_dir / "summary.json")  # {call_key: [entry, ...]}
    key_to_id = io.load_json(config.work / "call_key_to_example_id.json")
    clips = io.load_json(config.matched_clips_path) if config.matched_clips_path.exists() else {}
    tree_newick = Path(config.pruned_tree_path).read_text().strip()

    metadata = io.load_metadata(config.metadata_path)
    id_col, sp_col = config.id_column, config.species_column
    id_to_species = dict(zip(metadata[id_col], metadata[sp_col].str.replace(" ", "_"), strict=False))
    id_to_audio = dict(zip(metadata[id_col], metadata.get("audio_fp", [None] * len(metadata)), strict=False))

    # Per-mode reconstructed node likelihoods.
    mode_dirs = sorted(p for p in config.results_dir.iterdir() if (p / "raw-results.json").exists())
    raw_by_mode = {p.name: io.load_json(p / "raw-results.json") for p in mode_dirs}

    out_dir.mkdir(parents=True, exist_ok=True)
    calls_dir = out_dir / "calls"
    calls_dir.mkdir(exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_seen: dict[str, str | None] = {}

    def audio_ref(example_id: str) -> str | None:
        if example_id in audio_seen:
            return audio_seen[example_id]
        ref = None
        src = _resolve_audio(id_to_audio.get(example_id), audio_root)
        if src is not None and copy_audio:
            audio_dir.mkdir(exist_ok=True)
            safe = example_id.replace("/", "_") + src.suffix
            shutil.copyfile(src, audio_dir / safe)
            ref = f"audio/{safe}"
        audio_seen[example_id] = ref
        return ref

    call_keys = sorted(key_to_id, key=lambda k: int(k.split("_")[1]))
    call_list = []
    for ck in call_keys:
        ref_id = key_to_id[ck]
        ref_species = id_to_species.get(ref_id, "?")
        ages = {e_label: None for e_label in raw_by_mode}
        for entry in summary.get(ck, []):
            label = (
                f"binary_leaf{entry['leaf_threshold']}_ancestor{entry['ancestor_threshold']}"
                if entry["mode"] == "binary"
                else f"continuous_ancestor{entry['ancestor_threshold']}"
            )
            ages[label] = entry["age"]

        # Per-species rows, sorted by call probability (descending).
        species_rows = []
        for sp, prob in sorted(
            ((sp, v.get(ck, 0.0)) for sp, v in call_probs.items()), key=lambda x: x[1], reverse=True
        ):
            row_clips = [
                {**c, "species": sp, "audio": audio_ref(c["example_id"])} for c in clips.get(ck, {}).get(sp, [])
            ]
            species_rows.append({"species": sp, "probability": prob, "clips": row_clips})

        modes = [
            {
                "label": _mode_label(name),
                "name": name,
                "age": ages.get(name),
                "node_vals": raw.get(ck, {}).get("node_vals", {}),
            }
            for name, raw in raw_by_mode.items()
        ]

        io.save_json(
            {
                "call_key": ck,
                "reference_example_id": ref_id,
                "reference_species": ref_species,
                "reference_audio": audio_ref(ref_id),
                "species": species_rows,
                "modes": modes,
            },
            calls_dir / f"{ck}.json",
        )
        call_list.append(
            {"call_key": ck, "reference_example_id": ref_id, "reference_species": ref_species, "ages": ages}
        )

    io.save_json(
        {
            "experiment": Path(config.work_dir).name,
            "tree_newick": tree_newick,
            "modes": [_mode_label(n) for n in raw_by_mode],
            "mode_names": list(raw_by_mode),
            "calls": call_list,
        },
        out_dir / "data.json",
    )

    for html in ("verify.html", "index.html"):
        src = HERE / html
        if src.exists():
            shutil.copyfile(src, out_dir / html)

    n_audio = sum(1 for v in audio_seen.values() if v)
    print(f"Wrote {out_dir} ({len(call_keys)} calls, {n_audio} audio files bundled)")
    print(f"Serve with:  python -m http.server --directory {out_dir} 8000")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local data for the verification tool.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="Output dir (default: <work_dir>/verify_data).")
    parser.add_argument("--audio-root", default=None, help="Root to resolve relative audio_fp paths against.")
    parser.add_argument("--no-audio", action="store_true", help="Do not copy audio files.")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config)
    out_dir = Path(args.out) if args.out else config.work / "verify_data"
    audio_root = Path(args.audio_root) if args.audio_root else None
    build(config, out_dir, audio_root, copy_audio=not args.no_audio)


if __name__ == "__main__":
    main()
