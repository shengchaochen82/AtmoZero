"""Generate captions for held-out windows via vLLM.

Uses the same shared prompt template as the verl GRPO loop and the same
sampling configuration, so caption distributions at evaluation time match the
distributions seen during post-training. Output is a JSONL whose schema is
the input expected by ``scripts/eval_faithfulness.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from atmozero.caption.format import (
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    render_lexicon_block,
)
from atmozero.caption.parser import parse_caption


def _format_prompt(row: pd.Series, lexicon_block: str) -> str:
    return SYSTEM_PROMPT + "\n\n" + PROMPT_TEMPLATE.format(
        T_w=int(row["T_w"]),
        channels="T, P, q, u, v, r",
        lat=float(row.get("lat", 0.0)),
        lon=float(row.get("lon", 0.0)),
        elev=float(row.get("elevation", 0.0)),
        koppen=str(row.get("koppen_zone", "Cfa")),
        numeric_summary="see attached window",
        lexicon_block=lexicon_block,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="local path or HF id of the post-trained AtmoZero captioner")
    ap.add_argument("--windows", required=True, help="parquet from scripts/prepare_era5.py")
    ap.add_argument("--stations", required=False,
                    help="optional stations parquet for lat/lon/elev/koppen lookup")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--max_rows", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True, help="captions.jsonl destination")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    index = pd.read_parquet(args.windows)
    index = index[index["split"] == args.split].reset_index(drop=True)
    if args.max_rows > 0:
        index = index.head(args.max_rows).reset_index(drop=True)

    if args.stations:
        stations = pd.read_parquet(args.stations).set_index("station_id")
        index = index.join(stations, on="station_id", how="left")
    else:
        for col, default in (("lat", 0.0), ("lon", 0.0), ("elevation", 0.0), ("koppen_zone", "Cfa")):
            if col not in index.columns:
                index[col] = default

    lexicon_block = render_lexicon_block()
    prompts: List[str] = [_format_prompt(row, lexicon_block) for _, row in index.iterrows()]

    llm = LLM(
        model=args.checkpoint,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
    )
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts, params)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for window_id, (row, out) in enumerate(zip(index.iterrows(), outputs)):
            text = out.outputs[0].text
            parsed = parse_caption(text)
            record: Dict[str, Any] = {
                "window_id": int(window_id),
                "station_id": int(row[1].get("station_id", -1)),
                "t_start": str(row[1].get("t_start", "")),
                "text": text,
                "claims": [c.as_tuple() for c in parsed.claims],
            }
            f.write(json.dumps(record) + "\n")
    print(f"[generate_captions] wrote {len(outputs)} captions to {args.out}")


if __name__ == "__main__":
    main()
