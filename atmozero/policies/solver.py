"""Solver policy pi_S, full-parameter post-trained from Qwen2.5-7B-Instruct."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = AutoTokenizer = None


SOLVER_BACKBONE = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class SolverConfig:
    backbone: str = SOLVER_BACKBONE
    bf16: bool = True
    flash_attention: bool = True
    max_new_tokens: int = 384
    temperature: float = 1.0
    top_p: float = 0.95


def build_solver(cfg: SolverConfig = SolverConfig()) -> Tuple["AutoModelForCausalLM", "AutoTokenizer"]:
    if AutoModelForCausalLM is None:
        raise RuntimeError("transformers is required to build the Solver")

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.backbone,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if cfg.flash_attention else "sdpa",
        trust_remote_code=True,
    )
    return model, tokenizer
