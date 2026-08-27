# Release Notes & Changelog — ADBA

Lịch sử thay đổi của dự án. Phần **Lịch sử commit** bên dưới được sinh tự động từ `git log`
mỗi khi có commit mới — đừng sửa tay, sẽ bị ghi đè.

Định dạng theo [Keep a Changelog](https://keepachangelog.com/vi/1.1.0/),
phiên bản theo [Semantic Versioning](https://semver.org/lang/vi/).

---

## Quy ước phiên bản

| Thành phần | Tăng khi |
|---|---|
| **MAJOR** (`2.0.0`) | Đổi contract công khai: `MultiAgentState`, `ExecutionPlan`, `InsightOutput`, schema DB không tương thích ngược |
| **MINOR** (`1.1.0`) | Thêm agent, thêm loại biểu đồ, thêm domain dữ liệu, thêm skill — vẫn tương thích ngược |
| **PATCH** (`1.0.1`) | Sửa lỗi, chỉnh prompt, cập nhật tài liệu, nâng phụ thuộc |

Fine-tuned weights đánh phiên bản riêng trên Hugging Face
([`dangvanvy/adba-qwen-merged`](https://huggingface.co/dangvanvy/adba-qwen-merged)) —
một release của code có thể ghim một revision cụ thể của model.

Tiền tố commit (Conventional Commits) quyết định nhóm trong changelog:

| Tiền tố | Nhóm |
|---|---|
| `feat:` | Tính năng mới |
| `fix:` | Sửa lỗi |
| `perf:` | Hiệu năng |
| `refactor:` | Tái cấu trúc |
| `docs:` | Tài liệu |
| `test:` | Kiểm thử |
| `chore:` / `build:` / `ci:` | Hạ tầng & công cụ |

---

## Các mốc đã đạt

Tóm tắt do người viết — phần chi tiết ở mục **Lịch sử commit** bên dưới.

### Chưa phát hành — hướng tới `v1.0.0`

Đang ở M3 (hardening cho production). Chưa gắn tag phiên bản nào.

**Cần xong trước khi gắn `v1.0.0`:**

- [ ] Eval end-to-end trên golden set (không chỉ eval từng skill rời)
- [ ] Role read-only + SQL guard nhiều lớp — 0 câu lệnh ghi chạm tới DB
- [ ] Ngân sách thời gian toàn cục + node `finalize` trả kết quả một phần
- [ ] Sandbox Python cô lập ở mức container
- [ ] Video demo ~5 phút

### Đã hoàn thành theo thời gian

| Mốc | Nội dung | Bằng chứng trong repo |
|---|---|---|
| **Nền tảng** | Schema 3 domain (sales/inventory/HR), seed dữ liệu tiếng Việt, Docker Compose | `data/schemas/`, `data/seed/seed_data.py` |
| **Perception layer** | Introspect PostgreSQL → `info_box` JSON nén vừa context 4096 | `perception/extract_info_box.py` |
| **Dataset** | Sinh + validate + split train/valid/test cho 5 skill | `data/*.jsonl`, `training/validate_dataset.py` |
| **Baseline eval** | 98 mẫu trên Qwen2.5-Coder-7B gốc — SQL exec 84,4%, JSON valid 94,4% | `eval/baseline_results.json` |
| **Fine-tune LoRA** | QLoRA trên checkpoint-50 — SQL exec 96,9%, latency 75,8 s → 12,4 s | `training/finetuned_checkpoint50_results.json` |
| **Multi-agent** | Supervisor + SQL + Python + Viz + Insight + Reflector trên LangGraph | `graph/` |
| **Contract Pydantic** | `ExecutionPlan` (DAG hợp lệ) và `InsightOutput` (finding/evidence/action) | `schemas/` |
| **UI** | Streamlit chat có hiển thị plan, trace, bảng, biểu đồ, insight card | `app.py` |
| **Merged weights** | Đẩy lên Hugging Face Hub | `training/kaggle_merge_lora_export.ipynb` |
| **CI/CD** | Unit test + build & push image lên GHCR | `.github/workflows/ci-cd.yml` |
| **Tài liệu** | Đặc tả VI, spec hardening, spec đa schema, bộ docs này | `docs/` |

---

## Lịch sử commit

<!-- AUTO:begin id=changelog -->

### Chưa phát hành — chưa có git tag

> Chưa có tag phiên bản nào. Sau khi gắn tag (`git tag v1.0.0`), khối này tự tách theo từng phiên bản.

**Tính năng mới**

- chế độ triển khai chặn egress, mặc định on-prem — `0e4c5bc` (2026-08-19, Đặng Văn Vỹ)

**Tài liệu**

- sửa mô tả openai ở NGUỒN sinh, không sửa trong khối AUTO — `0e1f02d` (2026-08-19, Đặng Văn Vỹ)
- bộ tài liệu dự án + đường ống tự cập nhật theo commit — `53dec53` (2026-08-16, Đặng Văn Vỹ)
- plan triển khai — đường ống schema context (pha 0+1) — `642bc51` (2026-08-15, Đặng Văn Vỹ)
- spec — tách permitted_tables khỏi retrieved_tables; ghi kết quả spike — `9bfeb37` (2026-08-15, Đặng Văn Vỹ)
- spec — agent chạy nội bộ trên schema của khách hàng — `4c90f63` (2026-08-15, Đặng Văn Vỹ)
- spec — chuẩn bị đường nối cho đa khách hàng — `20249d8` (2026-08-12, Đặng Văn Vỹ)
- spec hardening production + tách tool qua MCP — `f2c4a08` (2026-08-12, Đặng Văn Vỹ)
- README theo Best-README-Template; bỏ AGENTS.md khỏi remote — `d18f115` (2026-05-19, Đặng Văn Vỹ)

**Hạ tầng & công cụ**

- ignore .claude/worktrees/ (worktree cô lập) — `3d2b768` (2026-08-15, Đặng Văn Vỹ)

**Khác**

- Update README for clarity and language consistency — `910c784` (2026-06-15, Đặng Văn Vỹ)
- readme change — `0fded65` (2026-05-19, Đặng Văn Vỹ)
- add readme — `6415751` (2026-05-19, Đặng Văn Vỹ)
- Stop tracking training/checkpoint-50 LoRA weights (large binaries). — `8b38bfd` (2026-05-19, Đặng Văn Vỹ)
- eval finetune Qlora — `49f5159` (2026-05-19, Đặng Văn Vỹ)
- set up agents — `1ba1cce` (2026-05-14, Đặng Văn Vỹ)
- model ready (baseline finetuning unable to be ready) — `f5ca80a` (2026-03-27, DangVanVy23521825)
- data preparation for initial fine-tuning lora — `ab628bd` (2026-03-25, DangVanVy23521825)
- Stop tracking .cursor directory — `931e6db` (2026-03-21, DangVanVy23521825)
- schemas defined — `36de198` (2026-03-21, DangVanVy23521825)
- Add .gitignore, remove .venv and .env from tracking — `c3a02fb` (2026-03-20, DangVanVy23521825)
- Initial commit — `f05958a` (2026-03-20, DangVanVy23521825)

<!-- AUTO:end id=changelog -->

---

## Ghi chú nâng cấp

### Khi schema DB đổi

```bash
docker compose down -v          # xoá volume — dữ liệu cũ mất
docker compose up -d postgres
./scripts/apply_schemas_docker.sh
python data/seed/seed_data.py
python perception/extract_info_box.py   # BẮT BUỘC — info_box phải khớp schema mới
```

Bỏ bước cuối là agent sẽ viết SQL theo schema cũ và lỗi ở tầng thực thi.

### Khi `requirements.txt` đổi

```bash
pip install -r requirements.txt
docker compose build app        # image không dùng lại layer pip cũ
```

### Khi đổi model

Cập nhật `PRIMARY_MODEL` trong `.env`, `ollama pull <model>`, rồi chạy lại
`eval/eval_runner.py --limit 20` để xác nhận không hồi quy trước khi tin vào kết quả.

<!-- AUTO:begin id=stamp -->

| Trường | Giá trị |
|---|---|
| Commit nguồn gần nhất | `0e1f02d` — docs: sửa mô tả openai ở NGUỒN sinh, không sửa trong khối AUTO |
| Tác giả | Đặng Văn Vỹ |
| Ngày commit | 2026-08-19 |
| Số commit nguồn | 22 |
| Sinh bởi | `scripts/update_docs.py` (hook `post-commit`) |

<!-- AUTO:end id=stamp -->
