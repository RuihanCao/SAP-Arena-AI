#!/usr/bin/env python3
"""Fine-tune supplied trophy V heads using your prepared afterstate labels.

Each JSONL row has game_id, state (engine afterstate), target_trophies (finite
number in [0,10]), and race: {turn, lives, opponent_lives, wins}. Race fields
are integers: turn >= 1, lives/opponent_lives in [0,6], wins in [0,10]. Supply
separate --train and --val files with disjoint game_ids, not an evaluation pool.
Targets are expected FINAL trophies, not additional trophies or win probability.
You supply labels and are responsible for their meaning; this tool never runs
MC rollouts, generates labels, or reproduces the released model's training run.

This is supervised trophy MSE fine-tuning of the shared trunk and value head.
It keeps the supplied model's encoder weights, feature normalization, leaf map
and value scale. No prefix-ranking objective is added. Best validation MSE
selects the saved epoch. --extractor-checkpoint is the ORIGINAL BC ZIP pinned
by the V artifact, not whichever BC proposer you currently use for search.
The input weights are never overwritten; --out must be a fresh directory."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch

from _training_data import digest, positive, read_train_val, require_fresh, write_json
from sap_ppo.train.vdistill import SCORE_MIN, SCORE_SPAN
from sap_ppo.train.vgame_model import bypass_features, save_heads
from sap_ppo.tools.vgame_scorer import VGameLeafScorer


def trophy_values(logits, leaf_map):
    """The existing leafmap_mse link: fixed map of the supplied head's logit."""
    return leaf_map.torch(SCORE_MIN + torch.sigmoid(logits) * SCORE_SPAN)


def validate_labels(rows):
    for row in rows:
        target = row.get("target_trophies")
        if (isinstance(target, bool) or not isinstance(target, (int, float))
                or not math.isfinite(target) or not 0 <= target <= 10):
            raise ValueError("target_trophies must be a finite number in [0,10]")
        race = row.get("race")
        if not isinstance(race, dict):
            raise ValueError("each value row requires a race object")
        if set(race) != {"turn", "lives", "opponent_lives", "wins"}:
            raise ValueError("race requires exactly turn, lives, opponent_lives and wins")
        for key, low, high in (("turn", 1, None), ("lives", 0, 6),
                               ("opponent_lives", 0, 6), ("wins", 0, 10)):
            value = race.get(key)
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value < low or (high is not None and value > high)):
                raise ValueError(f"invalid race.{key}: {value!r}")


def encoded_data(rows, scorer, batch):
    model = scorer.models[0]
    x = np.asarray([scorer.encoder.encode(row["state"]) for row in rows], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("encoded states contain NaN or infinity")
    with torch.no_grad():
        features = torch.cat([model.extractor(torch.as_tensor(x[i:i + batch]))
                              for i in range(0, len(x), batch)])
    bypass = bypass_features(*[np.asarray([row["race"][key] for row in rows])
                               for key in ("turn", "lives", "opponent_lives", "wins")])
    targets = torch.tensor([row["target_trophies"] for row in rows], dtype=torch.float32)
    return features, bypass, targets


def predictions(heads, features, bypass, leaf_map):
    logits = heads(features, bypass if heads.bypass_dim else None)["p_win_logit"]
    return trophy_values(logits, leaf_map)


def train_heads(scorer, train, val, *, epochs, batch, lr, seed):
    """Only the heads train; no optimizer ever owns extractor or map weights."""
    heads = scorer.models[0].heads
    f_train, b_train, y_train = encoded_data(train, scorer, batch)
    f_val, b_val, y_val = encoded_data(val, scorer, batch)
    loss_fn = torch.nn.functional.mse_loss
    with torch.no_grad():
        initial_val = float(loss_fn(predictions(heads, f_val, b_val, scorer.leaf_map), y_val))
    if not math.isfinite(initial_val):
        raise ValueError("initial validation MSE is not finite")
    optimizer = torch.optim.Adam(heads.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    curve, best_state, best_epoch, best_mse = [], None, None, float("inf")
    for epoch in range(1, epochs + 1):
        heads.train()
        order = rng.permutation(len(train))
        total = 0.0
        for start in range(0, len(order), batch):
            idx = torch.as_tensor(order[start:start + batch], dtype=torch.long)
            pred = predictions(heads, f_train[idx], b_train[idx], scorer.leaf_map)
            loss = loss_fn(pred, y_train[idx])
            if not torch.isfinite(loss):
                raise ValueError("training MSE is not finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        heads.eval()
        with torch.no_grad():
            mse = float(loss_fn(predictions(heads, f_val, b_val, scorer.leaf_map), y_val))
        if not math.isfinite(mse):
            raise ValueError("validation MSE is not finite")
        curve.append({"epoch": epoch, "train_mse": total / len(train), "val_mse": mse})
        print(f"epoch {epoch}/{epochs}: train_mse={total / len(train):.6f} val_mse={mse:.6f}", flush=True)
        if mse < best_mse:
            best_state = {key: value.detach().clone() for key, value in heads.state_dict().items()}
            best_epoch, best_mse = epoch, mse
    if best_state is None:
        raise RuntimeError("no trained epoch was selected")
    heads.load_state_dict(best_state, strict=True)
    heads.eval()
    return {"initial_val_mse": initial_val, "best_val_mse": best_mse,
            "best_epoch": best_epoch, "curve": curve}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--heads", type=Path, required=True, help="supplied leaf-mapped trophy V to fine-tune")
    ap.add_argument("--extractor-checkpoint", type=Path, required=True,
                    help="original BC ZIP matching the V extractor SHA, independently of your proposer")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args(argv)
    try:
        require_fresh(args.out)
        for key in ("epochs", "batch", "lr", "torch_threads"):
            positive(getattr(args, key), key)
        train, val = read_train_val(args.train, args.val)
        validate_labels(train + val)
        input_hashes = {"train": digest(args.train), "val": digest(args.val),
                        "heads": digest(args.heads), "extractor": digest(args.extractor_checkpoint)}
        torch.set_num_threads(args.torch_threads)
        torch.manual_seed(args.seed)
        payload = torch.load(args.heads, map_location="cpu", weights_only=False)
        meta = payload.get("metadata") or {}
        if meta.get("target") != "mc8_leafmap_trophies" or not meta.get("encoder") or not meta.get("leaf_map"):
            raise ValueError("supplied V must declare mc8_leafmap_trophies with encoder and leaf_map metadata")
        if payload.get("extractor_sha256") != input_hashes["extractor"]:
            raise ValueError("extractor checkpoint SHA does not match the supplied V; a proposer is not its extractor")
        scorer = VGameLeafScorer.from_checkpoints([args.heads], args.extractor_checkpoint,
                                                 blend=0.0, pessimism=0.0)
        if scorer.value_kind != "leafmap_trophies":
            raise ValueError("supplied V is not a leaf-mapped trophy model")
        model = scorer.models[0]
        original_extractor = {key: value.detach().clone() for key, value in model.extractor.state_dict().items()}
        original_norm = {key: value.detach().clone() for key, value in model.heads.state_dict().items()
                         if key in ("feat_mu", "feat_sd")}
        args.out.mkdir(parents=True, exist_ok=False)
        report = train_heads(scorer, train, val, epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed)
        if any(not torch.equal(value, model.extractor.state_dict()[key]) for key, value in original_extractor.items()):
            raise RuntimeError("extractor unexpectedly changed")
        if any(not torch.equal(value, model.heads.state_dict()[key]) for key, value in original_norm.items()):
            raise RuntimeError("feature normalization unexpectedly changed")
        if input_hashes != {"train": digest(args.train), "val": digest(args.val),
                            "heads": digest(args.heads), "extractor": digest(args.extractor_checkpoint)}:
            raise RuntimeError("input data or weights changed during training")
        metadata = {"schema": "sap-arena-prepared-value/v1", "target": "mc8_leafmap_trophies",
                    "encoder": copy.deepcopy(meta["encoder"]), "leaf_map": copy.deepcopy(meta["leaf_map"]),
                    "objective": "supplied_trophy_mse", "extractor_frozen": True,
                    "input_sha256": input_hashes, "train_rows": len(train), "val_rows": len(val),
                    "note": "Fine-tuned on supplied final-trophy labels; no MC or ranking-label generation.",
                    "config": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seed": args.seed},
                    **report}
        path = save_heads(args.out, model.heads, metadata, extractor_sha256=input_hashes["extractor"],
                          mode=payload["mode"], extractor_state=(original_extractor if payload["mode"] == "unfrozen" else None))
        reloaded = VGameLeafScorer.from_checkpoints([path], args.extractor_checkpoint, blend=0.0, pessimism=0.0)
        row = val[0]
        before = scorer.score_boards([row["state"]], **row["race"])["score"]
        after = reloaded.score_boards([row["state"]], **row["race"])["score"]
        if before != after:
            raise RuntimeError("saved value artifact did not reproduce its validation prediction")
        write_json(args.out / "metadata.json", metadata)
        print(f"saved {path}; reload prediction identical; best val MSE={report['best_val_mse']:.6f}")
    except (OSError, ValueError, RuntimeError) as exc:
        ap.exit(2, f"training failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
