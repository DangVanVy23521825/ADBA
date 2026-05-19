# So sánh Baseline vs Fine-tuned (checkpoint-50)

## Baseline (đã có sẵn)

File: `eval/baseline_results.json` — model Ollama `qwen2.5-coder:7b-instruct-q5_K_M`, 98 mẫu test.

| Metric | Baseline |
|--------|----------|
| SQL execution | 84.4% |
| Python syntax | 96.7% |
| Supervisor JSON | 94.4% |
| Insight JSON | 92.9% |
| Overall JSON | 94.4% |
| Avg latency | ~75.8s |

## Fine-tuned — chạy trên máy có GPU + đủ disk (~16GB)

Mac hiện tại **không đủ dung lượng** để tải `Qwen2.5-Coder-7B-Instruct` từ Hugging Face. Chạy trên **máy Windows 5090** (nơi đã train) hoặc Colab.

```bash
cd "ADBA — Autonomous Data & Business Intelligence Agent"
pip install peft transformers accelerate bitsandbytes sentencepiece psycopg2-binary

# Copy checkpoint-50 vào project nếu chưa có
# Quick test (20 mẫu)
set PYTHONPATH=.
python eval/eval_peft_runner.py --adapter training/checkpoint-50 --limit 20

# Full test (98 mẫu, vài giờ)
python eval/eval_peft_runner.py --adapter training/checkpoint-50

# So sánh với baseline
python eval/eval_compare.py
```

Kết quả lưu tại: `eval/finetuned_checkpoint50_results.json`  
Bảng so sánh: `eval/comparison_checkpoint50.json`

## Sau khi merge → Ollama (tùy chọn)

Merge LoRA trên GPU rồi convert GGUF → `ollama create` → chạy:

```bash
set EVAL_MODEL=adba-qwen-finetuned
python eval/eval_runner.py
```

Cùng pipeline metric với baseline, latency thường thấp hơn PEFT trên Hugging Face.
