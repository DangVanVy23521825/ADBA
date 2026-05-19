"""
eval/eval_peft_runner.py
------------------------
Evaluate a HuggingFace PEFT (LoRA) adapter on data/test.jsonl using the same
metrics as eval_runner.py (Ollama baseline).

Usage:
  pip install peft transformers accelerate sentencepiece

  # Quick check on CPU (giới hạn token cho nhanh; metric không đủ ý full eval)
  python eval/eval_peft_runner.py --adapter training/checkpoint-50 --limit 5 --cap-new-tokens 256

  # Full test set
  python eval/eval_peft_runner.py --adapter training/checkpoint-50

  # Explicit Apple Metal (may OOM on 16GB; auto uses CPU instead)
  python eval/eval_peft_runner.py --adapter training/checkpoint-50 --device mps --limit 5

  # Compare with baseline
  python eval/eval_compare.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.eval_runner import (
    EVAL_MODEL,
    SKILL_MAX_TOKENS,
    SKILL_TEMPERATURES,
    TEST_FILE,
    aggregate,
    detect_skill,
    evaluate_sample,
)

DEFAULT_ADAPTER = ROOT / "training" / "checkpoint-50"
DEFAULT_BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_FILE = ROOT / "eval" / "finetuned_checkpoint50_results.json"


def _pick_device_auto() -> str:
    """Default backend when ``--device auto``."""
    if torch.cuda.is_available():
        return "cuda"
    # MPS + transformers `caching_allocator_warmup` does one giant torch.empty on the
    # accelerator — often blows up (~14 GiB) on unified 16 GiB macs. Prefer CPU fp16.
    return "cpu"


@contextlib.contextmanager
def _suppress_caching_allocator_warmup() -> Iterator[None]:
    """Skip HF pre-allocation warmup (problematic on Apple MPS for 7B fp16)."""
    try:
        import transformers.modeling_utils as modeling_utils
    except ImportError:
        yield
        return
    orig = modeling_utils.caching_allocator_warmup
    modeling_utils.caching_allocator_warmup = lambda *_a, **_k: None
    try:
        yield
    finally:
        modeling_utils.caching_allocator_warmup = orig


def load_peft_model(
    base_model: str,
    adapter_path: Path,
    *,
    device: str = "auto",
) -> tuple[Any, Any, str]:
    """Load base + LoRA.

    ``device``: ``auto`` (CUDA→4-bit GPU; else CPU fp16 — avoids buggy MPS warmup),
    ``cuda``, ``cpu``, ``mps``. CPU+MPS load weights in fp16 (~14 GiB) not fp32.
    """
    if device == "auto":
        resolved = _pick_device_auto()
    else:
        resolved = device

    # fp16 on CPU/MPS keeps weight RAM near ~14 GiB for 7B vs ~28 GiB fp32.
    dtype = torch.float16
    print(f"[PEFT] device={resolved} dtype={dtype}")
    print(f"[PEFT] base={base_model}")
    print(f"[PEFT] adapter={adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path) if (adapter_path / "tokenizer_config.json").exists() else base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {
        "dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if resolved == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["device_map"] = "auto"
            print("[PEFT] Using 4-bit quantization (CUDA)")
        except ImportError:
            load_kwargs["device_map"] = "auto"
    elif resolved != "cpu":
        load_kwargs["device_map"] = {"": resolved}

    if resolved == "cpu":
        print(
            "[PEFT] Using CPU fp16 — slower than accelerator but avoids Apple MPS "
            "caching_allocator_warmup (Invalid buffer size) on tight RAM.",
        )

    warmup_mgr = (
        _suppress_caching_allocator_warmup()
        if resolved == "mps"
        else contextlib.nullcontext()
    )
    if resolved == "mps":
        print(
            "[PEFT] MPS: skipped transformers caching_allocator_warmup for load. "
            "If inference OOMs, rerun with --device cpu.",
        )

    with warmup_mgr:
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    return model, tokenizer, resolved


def make_call_peft(model, tokenizer, device: str, *, cap_new_tokens: int = 0):
    """Return inference callable compatible with evaluate_sample injection.

    If ``cap_new_tokens`` > 0, ``max_new_tokens`` becomes ``min(skill cap, cap)`` — useful on CPU/Mac.
    """

    def call_peft(messages: list[dict], skill: str) -> tuple[str, float]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt")
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        max_new = SKILL_MAX_TOKENS.get(skill, 1024)
        if cap_new_tokens > 0:
            max_new = min(max_new, cap_new_tokens)
        temp = SKILL_TEMPERATURES.get(skill, 0.1)
        do_sample = temp > 0

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=do_sample,
                temperature=temp if do_sample else None,
                top_p=0.9 if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return generated, elapsed

    return call_peft


def run_eval(
    adapter_path: Path,
    base_model: str,
    test_file: Path,
    output_path: Path,
    limit: int = 0,
    device: str = "auto",
    cap_new_tokens: int = 0,
) -> dict:
    import psycopg2
    from eval.eval_runner import POSTGRES_URL, _status

    samples = [
        json.loads(line)
        for line in test_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit > 0:
        samples = samples[:limit]

    model, tokenizer, device = load_peft_model(base_model, adapter_path, device=device)
    call_peft = make_call_peft(model, tokenizer, device, cap_new_tokens=cap_new_tokens)

    pg_conn = None
    try:
        pg_conn = psycopg2.connect(POSTGRES_URL)
        pg_conn.autocommit = False
        print(f"[DB] Connected", flush=True)
    except Exception as e:
        print(f"[WARN] PostgreSQL unavailable ({e}). SQL exec checks disabled.")

    if device == "cpu":
        print(
            "[gen] Suy luận trên CPU (7B) rất chậm; không có output cho tới khi xong "
            "`generate` của từng mẫu — có thể vài–mười phút/mẫu tùy prompt.",
            flush=True,
        )
    if cap_new_tokens > 0:
        print(
            f"[gen] Giới hạn max_new_tokens tối đa = {cap_new_tokens} (--cap-new-tokens), "
            "kết quả eval không đầy đủ so preset skill.",
            flush=True,
        )

    results: list[dict] = []
    t0 = time.time()
    total = len(samples)

    for i, sample in enumerate(samples, start=1):
        messages = sample["messages"]
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        skill = detect_skill(system)
        input_msgs = [m for m in messages if m["role"] != "assistant"]

        max_hint = SKILL_MAX_TOKENS.get(skill, 1024)
        if cap_new_tokens > 0:
            max_hint = min(max_hint, cap_new_tokens)
        print(
            f"[gen] sample {i}/{total} skill={skill} … (max_new_tokens≤{max_hint})",
            flush=True,
        )

        try:
            generated, latency = call_peft(input_msgs, skill)
        except Exception as e:
            results.append({
                "idx": i,
                "skill": skill,
                "passed": False,
                "error": f"peft_error: {e}",
                "latency_s": 0.0,
                "generated": "",
            })
            print(_status(i, total, results[-1]))
            continue

        fake_sample = {
            "messages": input_msgs + [{"role": "assistant", "content": generated}],
        }
        r = evaluate_sample(fake_sample, pg_conn, dry_run=True, idx=i)
        r["latency_s"] = round(latency, 3)
        r["generated"] = generated[:500]
        results.append(r)
        print(_status(i, total, r))

    total_elapsed = round(time.time() - t0, 1)
    summary = aggregate(results)
    summary["total_elapsed_s"] = total_elapsed
    summary["model"] = f"{base_model} + LoRA:{adapter_path.name}"
    summary["backend"] = "peft"
    summary["adapter_path"] = str(adapter_path)
    summary["base_model"] = base_model
    summary["device"] = device
    summary["dry_run"] = False
    summary["test_file"] = str(test_file)
    summary["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    output_data = {"summary": summary, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")

    if pg_conn:
        pg_conn.close()

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ADBA PEFT adapter evaluation")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--test-file", default=str(TEST_FILE))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="auto: CUDA+4-bit if available, else CPU. Use mps to try Apple GPU (risky OOM).",
    )
    parser.add_argument(
        "--cap-new-tokens",
        type=int,
        default=0,
        metavar="N",
        help="If >0: cap max_new_tokens at N per sample (speeds smoke tests on CPU; metrics not comparable).",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not (adapter_path / "adapter_model.safetensors").exists():
        print(f"[ERROR] adapter not found: {adapter_path}", file=sys.stderr)
        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[ERROR] --device cuda but CUDA not available.", file=sys.stderr)
        sys.exit(1)
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("[ERROR] --device mps but MPS not available.", file=sys.stderr)
        sys.exit(1)

    print("ADBA PEFT Eval — LIVE")
    print(f"Baseline reference model (Ollama): {EVAL_MODEL}")
    summary = run_eval(
        adapter_path=adapter_path,
        base_model=args.base_model,
        test_file=Path(args.test_file),
        output_path=Path(args.output),
        limit=args.limit,
        device=args.device,
        cap_new_tokens=args.cap_new_tokens,
    )

    print("\n" + "=" * 60)
    print("FINETUNED (checkpoint) RESULTS")
    print("=" * 60)
    for label, key in [
        ("SQL Execution Accuracy", "sql_execution_accuracy"),
        ("Python Syntax Rate", "python_syntax_rate"),
        ("Supervisor JSON Rate", "supervisor_json_rate"),
        ("Insight JSON Rate", "insight_json_rate"),
        ("Overall JSON Valid Rate", "overall_json_valid_rate"),
    ]:
        print(f"  {label:<30} {summary.get(key, 0):>6.1f}%")
    print(f"  Avg latency: {summary.get('avg_latency_s')}s")
    print(f"  Total elapsed: {summary.get('total_elapsed_s')}s")
    print(f"[Saved] {args.output}")


if __name__ == "__main__":
    main()
