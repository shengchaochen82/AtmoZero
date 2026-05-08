"""End-to-end AtmoZero training entrypoint."""
from __future__ import annotations

import argparse
import time
from typing import Iterator, Tuple

import numpy as np
import torch

from atmozero.utils import load_config, set_seeds
from atmozero.policies import build_solver, Proposer, ProposerConfig
from atmozero.grpo import AtmoZeroGRPOTrainer, GRPOConfig
from atmozero.grpo.rollout import RewardOut
from atmozero.reward import RewardConfig, compose_reward, ShapingSchedule, shaped_reward
from atmozero.verifier import VerifierGrader
from atmozero.verifier.base import WindowContext
from atmozero.caption import parse_caption
from atmozero.caption.format import (
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    render_lexicon_block,
)


def build_reward_fn(grader, reward_cfg, schedule):
    def reward_fn(captions, x_window, regime, step):
        parsed = parse_caption(captions[0])
        ctx = WindowContext(x=x_window, metadata={}, neighbors=None, climatology=None)
        signals = grader.grade_window(parsed.as_q_list(), ctx)
        R_base = compose_reward(signals, captions[0], reward_cfg)
        R = shaped_reward(R_base, R_cycle=0.0, R_uplift=0.0, step=step, schedule=schedule)

        n_claims = len(parsed.claims)
        if n_claims == 0:
            precision = 0.0
        else:
            per_claim = grader.per_claim_grades(parsed.as_q_list(), ctx)
            supported = sum(1 for s, claim in zip(per_claim, parsed.claims)
                            if s >= grader.rules[claim.family_id].tau_plus)
            precision = supported / n_claims

        return RewardOut(reward=R, precision=precision, infeasible=False)

    return reward_fn


def _prompt_loader(window_index_path: str, tokenizer, batch_size: int) -> Iterator[Tuple]:
    """Yield (prompt_ids, x_batch, (window_idx, regime_idx)) tuples forever."""
    import pandas as pd

    index = pd.read_parquet(window_index_path)
    index = index[index["split"] == "train"].reset_index(drop=True)
    if len(index) == 0:
        raise RuntimeError(f"empty window index at {window_index_path}")

    lexicon_block = render_lexicon_block()
    rng = np.random.default_rng(0)
    while True:
        idx = rng.integers(0, len(index), size=batch_size)
        prompts = []
        x_batch = []
        for j in idx:
            row = index.iloc[int(j)]
            prompts.append(_format_prompt(row, lexicon_block))
            x_batch.append(_synthetic_channels(int(row.station_id), str(row.t_start)))
        prompt_ids = _tokenize_batch(tokenizer, prompts)
        window_idx = torch.tensor([int(i) for i in idx], dtype=torch.long)
        regime_idx = torch.zeros_like(window_idx)
        yield prompt_ids, x_batch, (window_idx, regime_idx)


def _format_prompt(row, lexicon_block: str) -> str:
    return SYSTEM_PROMPT + "\n\n" + PROMPT_TEMPLATE.format(
        T_w=int(row.T_w),
        channels="T, P, q, u, v, r",
        lat=0.0,
        lon=0.0,
        elev=0.0,
        koppen="Cfa",
        numeric_summary="see attached window",
        lexicon_block=lexicon_block,
    )


def _tokenize_batch(tokenizer, prompts):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    return enc["input_ids"]


def _synthetic_channels(station_id: int, t_start: str):
    rng = np.random.default_rng(abs(hash((station_id, t_start))) & 0xFFFFFFFF)
    T_w = 192
    t = np.arange(T_w)
    diurnal = 5 * np.sin(2 * np.pi * t / 24)
    return {
        "T": 280 + 0.05 * t + diurnal + rng.normal(0, 0.5, T_w),
        "P": 1013 + 2 * np.sin(2 * np.pi * t / 96) + rng.normal(0, 0.3, T_w),
        "q": 0.008 + 0.0003 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1e-4, T_w),
        "u": 5 + rng.normal(0, 1, T_w),
        "v": 2 + rng.normal(0, 1, T_w),
        "r": np.zeros(T_w),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--windows", default="data/processed/windows.parquet",
                    help="parquet path produced by scripts/prepare_era5.py")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seeds(args.seed)

    solver, tokenizer = build_solver()
    ref_solver, _ = build_solver()

    proposer = Proposer(ProposerConfig(
        n_windows=cfg["data"]["n_stations"],
        n_regimes=cfg["proposer"]["n_regimes"],
        entropy_bonus=cfg["proposer"]["entropy_bonus"],
        infeasibility_penalty=cfg["proposer"].get("infeasibility_penalty", 1.0),
    ))

    grader = VerifierGrader()
    reward_cfg = RewardConfig(
        w={int(k): float(v) for k, v in cfg["reward"]["weights"].items()},
        lam=cfg["reward"]["lambda"],
        mu=cfg["reward"]["mu"],
        nu=cfg["reward"]["nu"],
    )
    schedule = ShapingSchedule(
        alpha0=cfg["shaping"]["alpha0"],
        beta0=cfg["shaping"]["beta0"],
        tau_alpha=cfg["shaping"]["tau_alpha"],
        tau_beta=cfg["shaping"]["tau_beta"],
        clamp_step=cfg["shaping"]["clamp_step"],
    )

    reward_fn = build_reward_fn(grader, reward_cfg, schedule)

    grpo_cfg = GRPOConfig(
        total_steps=cfg["grpo"]["total_steps"],
        group_size=cfg["grpo"]["group_size"],
        batch_size=cfg["grpo"]["batch_size"],
        learning_rate=cfg["grpo"]["learning_rate"],
        proposer_lr=cfg["grpo"]["proposer_lr"],
        weight_decay=cfg["grpo"]["weight_decay"],
        kl_coef=cfg["grpo"]["kl_coef"],
        clip_epsilon=cfg["grpo"]["clip_epsilon"],
        cold_start_threshold=cfg["cold_start"]["threshold"],
        output_dir=cfg["experiment"]["output_dir"],
    )

    trainer = AtmoZeroGRPOTrainer(
        solver, ref_solver, tokenizer, proposer, reward_fn, grpo_cfg, schedule
    )

    loader = _prompt_loader(args.windows, tokenizer, batch_size=grpo_cfg.batch_size)
    print(f"[train_atmozero] seed={args.seed}, steps={grpo_cfg.total_steps}")
    t0 = time.time()
    trainer.fit(loader)
    print(f"[train_atmozero] training complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
