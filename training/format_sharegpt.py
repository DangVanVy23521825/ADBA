"""
ADBA — Format ShareGPT + Train/Val/Test Split
==============================================
Reads validated_dataset.jsonl and produces MLX-LM JSONL files:
  data/train.jsonl   (80%)   — one JSON object per line
  data/valid.jsonl   (10%)   — MLX-LM expects "valid", not "val"
  data/test.jsonl    (10%)

Problems addressed:
  1. Deduplication     — exact (user+assistant) duplicates removed
  2. Context trimming  — supervisor system prompt trimmed if still using full info_box
  3. Long-context filtering — records > MAX_TOKENS saved separately, not discarded
  4. Cross-domain SQL  — is_cross_domain records kept in train only, excluded from val/test
  5. Stratified split  — proportional skill_type counts in each split
  6. Output format     — JSONL (one record per line) for MLX-LM compatibility
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"

INPUT_FILE    = DATA_DIR / "validated_dataset.jsonl"
TRAIN_FILE    = DATA_DIR / "train.jsonl"
VAL_FILE      = DATA_DIR / "valid.jsonl"   # MLX-LM expects "valid", not "val"
TEST_FILE     = DATA_DIR / "test.jsonl"
LONG_CTX_FILE = DATA_DIR / "long_context_excluded.jsonl"

MAX_TOKENS_DEFAULT = 3800
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
RANDOM_SEED = 42


# =============================================================================
# SUPERVISOR CONTEXT TRIMMER
# =============================================================================

def _trim_supervisor_prompt(system: str) -> str:
    """
    Replace injected info_box JSON in supervisor prompt with compact schema summary.
    Detects the block between ## SCHEMA CONTEXT and ## OUTPUT FORMAT.
    """
    schema_marker = "## SCHEMA CONTEXT"
    output_marker = "## OUTPUT FORMAT"

    idx_s = system.find(schema_marker)
    idx_o = system.find(output_marker)
    if idx_s == -1 or idx_o == -1:
        return system

    before = system[:idx_s + len(schema_marker)]
    after  = system[idx_o:]
    blob   = system[idx_s + len(schema_marker):idx_o].strip()

    try:
        info_box = json.loads(blob)
    except json.JSONDecodeError:
        return system

    lines = ["\n(Schema summary — tables and columns only)\n"]
    for t in info_box.get("tables", []):
        tname = t.get("table_name", "?")
        cols  = ", ".join(c["name"] for c in t.get("columns", []))
        fks   = "; ".join(
            f"{f['column']}→{f['references']}" for f in t.get("foreign_keys", [])
        )
        row_count = t.get("row_count", 0)
        lines.append(f"  {tname} ({row_count:,} rows): {cols}")
        if fks:
            lines.append(f"    FK: {fks}")

    hints = info_box.get("cross_domain_hints", [])
    if hints:
        lines.append("Cross-domain links:")
        for h in hints:
            lines.append(
                f"  {h['from_table']}.{h['from_column']} → "
                f"{h['to_table']}.{h['to_column']}"
            )

    return f"{before}\n" + "\n".join(lines) + f"\n\n{after}"


# =============================================================================
# TOKEN ESTIMATION
# =============================================================================

def estimate_tokens(record: dict) -> int:
    # UTF-8 byte count / 4 is more accurate than char count / 4 for Vietnamese text.
    # Vietnamese diacritics encode as 2–3 bytes each; BPE tokenizers operate on
    # byte-level subwords, so byte count / 4 underestimates less than char count / 4.
    return sum(len(m["content"].encode("utf-8")) for m in record["messages"]) // 4


# =============================================================================
# DEDUPLICATION
# =============================================================================

def _key(record: dict) -> str:
    user = record["messages"][1]["content"]
    asst = record["messages"][2]["content"]
    return hashlib.md5(f"{user}|||{asst}".encode()).hexdigest()


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    out: list[dict] = []
    removed = 0
    for r in records:
        k = _key(r)
        if k in seen:
            removed += 1
        else:
            seen.add(k)
            out.append(r)
    return out, removed


# =============================================================================
# STRATIFIED SPLIT
# =============================================================================

def stratified_split(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    by_skill: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_skill[r["skill_type"]].append(r)

    train, val, test = [], [], []
    for skill, recs in sorted(by_skill.items()):
        rng.shuffle(recs)
        n       = len(recs)
        n_train = max(1, round(n * train_ratio))
        n_val   = max(1, round(n * val_ratio))
        train.extend(recs[:n_train])
        val.extend(recs[n_train:n_train + n_val])
        test.extend(recs[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# =============================================================================
# CONVERT TO MLX-LM FORMAT
# =============================================================================

def to_sharegpt(record: dict) -> dict:
    """Keep only messages key — MLX-LM does not use skill_type."""
    return {"messages": record["messages"]}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write one JSON object per line — required format for MLX-LM."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main(
    input_path: Path    = INPUT_FILE,
    train_path: Path    = TRAIN_FILE,
    val_path: Path      = VAL_FILE,
    test_path: Path     = TEST_FILE,
    long_ctx_path: Path = LONG_CTX_FILE,
    max_tokens: int     = MAX_TOKENS_DEFAULT,
    trim_supervisor: bool = True,
    seed: int           = RANDOM_SEED,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    raw: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except Exception as e:
                print(f"  line {i} parse error: {e}")
    print(f"Loaded:            {len(raw):>5,} records")

    # Step 1: Trim supervisor context
    n_trimmed = 0
    if trim_supervisor:
        for r in raw:
            if r.get("skill_type") == "supervisor-routing":
                orig    = r["messages"][0]["content"]
                trimmed = _trim_supervisor_prompt(orig)
                if len(trimmed) < len(orig):
                    r["messages"][0]["content"] = trimmed
                    n_trimmed += 1
        before_tok = sum(
            len(r["messages"][0]["content"]) // 4 for r in raw
            if r.get("skill_type") == "supervisor-routing"
        )
        print(f"Supervisor trim:   {n_trimmed:>5,} prompts → "
              f"avg sys tokens now ~{before_tok // max(n_trimmed, 1)}")

    # Step 2: Deduplication
    deduped, n_dupes = deduplicate(raw)
    print(f"Deduplication:     {n_dupes:>5,} exact dupes removed")
    print(f"After dedup:       {len(deduped):>5,} records")

    # Step 3: Context-length filter
    within: list[dict] = []
    over:   list[dict] = []
    for r in deduped:
        tokens = estimate_tokens(r)
        r["_tokens"] = tokens
        (within if tokens <= max_tokens else over).append(r)

    print(f"Within {max_tokens}t:     {len(within):>5,} records")
    print(f"Over   {max_tokens}t:     {len(over):>5,} records → {long_ctx_path.name}")

    _write_jsonl(long_ctx_path, [to_sharegpt(r) for r in over])

    for r in within:
        r.pop("_tokens", None)

    # Step 4: Separate cross-domain SQL — train only, never val/test
    # Records tagged is_cross_domain=True have ~5,500-token context; keeping them
    # in val/test would cause silent truncation when evaluating with Qwen-7B 4096 ctx.
    cross_domain = [r for r in within if r.get("is_cross_domain")]
    splittable   = [r for r in within if not r.get("is_cross_domain")]
    if cross_domain:
        print(f"Cross-domain SQL:  {len(cross_domain):>5,} → train only (excluded from val/test)")

    # Step 5: Distribution check (on splittable pool)
    print("\nDistribution after dedup + filter (splittable):")
    by_skill: dict[str, int] = defaultdict(int)
    for r in splittable:
        by_skill[r["skill_type"]] += 1
    for skill in sorted(by_skill):
        print(f"  {skill:30s}: {by_skill[skill]:>4,}")
    print(f"  {'TOTAL':30s}: {len(splittable):>4,}")

    # Step 6: Stratified split on splittable records, then fold cross-domain into train
    train, val, test = stratified_split(splittable, TRAIN_RATIO, VAL_RATIO, seed)
    train = cross_domain + train          # cross-domain always in train
    random.Random(seed).shuffle(train)    # re-shuffle to avoid positional bias

    # Step 7: Write JSONL output — one record per line for MLX-LM compatibility
    for path, split, name in [
        (train_path, train, "train"),
        (val_path,   val,   "valid"),
        (test_path,  test,  "test"),
    ]:
        data = [to_sharegpt(r) for r in split]
        _write_jsonl(path, data)
        print(f"\n{name}:  {len(data):>4,} records → {path.name}")
        sk: dict[str, int] = defaultdict(int)
        for r in split:
            sk[r["skill_type"]] += 1
        for skill in sorted(sk):
            print(f"  {skill:30s}: {sk[skill]:>3,}")

    # Sanity: no val/test overlap (cross-domain records are only in train — no check needed)
    def rid(r: dict) -> str:
        return hashlib.md5(
            (r["messages"][1]["content"] + r["messages"][2]["content"]).encode()
        ).hexdigest()

    # Exclude cross-domain-only train records from overlap check (they were never in val/test)
    train_splittable_ids = {rid(r) for r in train if not r.get("is_cross_domain")}
    va_ids = {rid(r) for r in val}
    te_ids = {rid(r) for r in test}
    assert not (train_splittable_ids & va_ids), "train/val overlap!"
    assert not (train_splittable_ids & te_ids), "train/test overlap!"
    assert not (va_ids & te_ids),               "val/test overlap!"

    total = len(train) + len(val) + len(test)
    print(f"\n{'='*44}")
    print(f"Total output: {total:,}  "
          f"train={len(train)}({len(train)/total*100:.0f}%)  "
          f"valid={len(val)}({len(val)/total*100:.0f}%)  "
          f"test={len(test)}({len(test)/total*100:.0f}%)")
    print(f"wc -l check: train.jsonl={len(train)}  valid.jsonl={len(val)}  test.jsonl={len(test)}")
    print("✓ No overlap between splits")
    print(f"{'='*44}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",              type=Path, default=INPUT_FILE)
    parser.add_argument("--max-tokens",         type=int,  default=MAX_TOKENS_DEFAULT)
    parser.add_argument("--no-trim-supervisor", action="store_true")
    parser.add_argument("--seed",               type=int,  default=RANDOM_SEED)
    args = parser.parse_args()

    main(
        input_path      = args.input,
        max_tokens      = args.max_tokens,
        trim_supervisor = not args.no_trim_supervisor,
        seed            = args.seed,
    )
