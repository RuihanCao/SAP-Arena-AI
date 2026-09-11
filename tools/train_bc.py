#!/usr/bin/env python3
"""Train an attention BC policy on your prepared (state, action) JSONL rows.

Required row fields: game_id (local grouping key), state (engine schema),
action (engine action schema). Supply separate --train and --val files with
disjoint game_ids. All labels receive equal weight: ordinary unmasked action
negative log likelihood, not the released model's historical weighting recipe.
Extra row fields never change training weights. No data or labels are generated.

The existing BC trainer supplies optimization and checkpoint selection. This
entry uses v4_one_turn_context (1993 inputs), slot_attn and [256,256] trunks.
--init-checkpoint optionally warm-starts a matching attention policy with strict
shape/class loading. It is never overwritten. Otherwise weights start random.
Outputs include checkpoint_best.zip, checkpoint_final.zip and training metrics.
The ZIP can be used as a BC proposer. It does NOT replace the extractor pinned
inside an existing trophy V; keep that V's original extractor checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from _training_data import digest, positive, read_train_val, require_fresh, write_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init-checkpoint", type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args(argv)
    try:
        require_fresh(args.out)
        for key in ("epochs", "batch", "lr", "torch_threads"):
            positive(getattr(args, key), key)
        train, val = read_train_val(args.train, args.val)
        from sap_ppo.schema import validate_action_schema
        from sap_ppo.train.chain_bc_dataset import map_action_index
        for row in train + val:
            if not isinstance(row.get("action"), dict):
                raise ValueError("each BC row requires an engine action")
            validate_action_schema(row["action"])
            if map_action_index(row["action"]) is None:
                raise ValueError("BC action is outside ACTION_CATALOG")
        initial_hash = digest(args.init_checkpoint) if args.init_checkpoint else None
        input_hashes = {"train": digest(args.train), "val": digest(args.val)}
        from sap_ppo.train.train_chain_bc import main as train_existing
        # The existing prepared-only loader has a margin-weight field. Giving
        # EVERY row 0.5 yields weight 1.0 exactly; its normalized loss is plain
        # mean NLL. Never forward a user's teacher/rank weighting fields.
        with tempfile.TemporaryDirectory(prefix="sap-bc-input-") as temporary:
            paths = []
            for name, rows in (("train", train), ("val", val)):
                path = Path(temporary) / f"{name}.jsonl"
                with path.open("x", encoding="utf-8") as stream:
                    for row in rows:
                        stream.write(json.dumps({"game_id": row["game_id"], "state": row["state"],
                            "action": row["action"], "source": "prepared", "teacher_margin": 0.5},
                            allow_nan=False) + "\n")
                paths.append(path)
            args.out.mkdir(parents=True, exist_ok=False)
            cli = ["--extra-dataset-only", "--extra-dataset", str(paths[0]),
                   "--extra-val-dataset", str(paths[1]), "--out", str(args.out),
                   "--features", "slot_attn", "--net-arch", "256,256", "--no-tensorboard",
                   "--device", args.device, "--epochs", str(args.epochs), "--batch", str(args.batch),
                   "--lr", str(args.lr), "--seed", str(args.seed), "--torch-threads", str(args.torch_threads)]
            if args.init_checkpoint:
                cli += ["--init-checkpoint", str(args.init_checkpoint)]
            if train_existing(cli) != 0:
                raise RuntimeError("BC trainer did not complete")
        if args.init_checkpoint and digest(args.init_checkpoint) != initial_hash:
            raise RuntimeError("input checkpoint changed")
        if input_hashes != {"train": digest(args.train), "val": digest(args.val)}:
            raise RuntimeError("input data changed during training")
        write_json(args.out / "prepared_training.json", {
            "objective": "uniform_unmasked_action_nll", "source": "user_prepared",
            "input_sha256": input_hashes, "init_checkpoint_sha256": initial_hash,
            "train_rows": len(train), "val_rows": len(val), "features": "slot_attn",
            "net_arch": [256, 256], "checkpoint": "checkpoint_best.zip",
        })
    except (OSError, ValueError, RuntimeError) as exc:
        ap.exit(2, f"training failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
