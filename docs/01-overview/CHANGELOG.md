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

- node finalize — thang success/partial/failed có lý do — `fc69ccb` (2026-09-01, Đặng Văn Vỹ)
- ModelClient từ chối khởi động lời gọi không kịp deadline — `3c96a2c` (2026-09-01, Đặng Văn Vỹ)
- trần cứng thay đếm retry — reflector 8→1, sql retry 3→2, trần 12 call — `2cf725e` (2026-09-01, Đặng Văn Vỹ)
- deadline_ts mang trong state, clock tiêm được — `45b7b5d` (2026-09-01, Đặng Văn Vỹ)
- role adba_readonly — lớp bảo đảm chỉ-đọc ở tầng Postgres — `0120aaa` (2026-09-01, Đặng Văn Vỹ)
- Chinook làm schema demo phát hành lại được — `ff31b35` (2026-08-23, Đặng Văn Vỹ)
- --threshold-tokens chốt chế độ schema tại build — `42ece63` (2026-08-23, Đặng Văn Vỹ)
- bridge BIRD SQLite databases into Postgres for onboarding evals — `3116622` (2026-08-22, Đặng Văn Vỹ)
- chế độ triển khai chặn egress, mặc định on-prem — `0e4c5bc` (2026-08-19, Đặng Văn Vỹ)
- đọc profile từ đĩa thay vì dựng lại mỗi câu hỏi — `b898758` (2026-08-18, Đặng Văn Vỹ)
- lệnh refresh, báo số mục người duyệt được giữ — `50e8b64` (2026-08-18, Đặng Văn Vỹ)
- lệnh verify, ngưỡng bàn giao recall 95% — `11c9593` (2026-08-18, Đặng Văn Vỹ)
- lỗi vận hành in thành câu, không thành traceback — `31d7a7d` (2026-08-18, Đặng Văn Vỹ)
- lệnh build, có cổng chặn bàn giao khi chú giải chưa duyệt — `354e219` (2026-08-18, Đặng Văn Vỹ)
- trang duyệt chú giải cho analyst nghiệp vụ — `573da16` (2026-08-18, Đặng Văn Vỹ)
- lệnh extract và annotate — `be7253b` (2026-08-18, Đặng Văn Vỹ)
- đọc/ghi thư mục profile, tách cấu trúc khỏi chú giải — `809ea5c` (2026-08-18, Đặng Văn Vỹ)
- sinh chú giải bằng model local, có cờ độ tự tin — `6b3994c` (2026-08-18, Đặng Văn Vỹ)
- kho chú giải + merge giữ nguyên mục người sửa — `f252c26` (2026-08-18, Đặng Văn Vỹ)
- Column mang mô tả, render thành comment cuối dòng — `3c837a8` (2026-08-17, Đặng Văn Vỹ)
- introspect_schema đọc Postgres thẳng ra Table — `f5b7207` (2026-08-17, Đặng Văn Vỹ)
- tách hai trục quyền; xác nhận license Spider + BIRD — `99aa7b3` (2026-08-16, Đặng Văn Vỹ)
- khoá dữ liệu benchmark + script tải có cổng license — `ddbb436` (2026-08-16, Đặng Văn Vỹ)
- thêm load_normalized + describe_dataset cho bộ dữ liệu ngoài — `b81fb16` (2026-08-16, Đặng Văn Vỹ)
- nối SchemaContext vào state, supervisor, sql_agent và app — `e638fe7` (2026-08-16, Đặng Văn Vỹ)
- execute_sql thực thi permitted_tables, bỏ whitelist 9 bảng hardcode — `3ea38e4` (2026-08-16, Đặng Văn Vỹ)
- harness recall tầng 1 + mốc lexical vs full trên golden ADBA — `b3ffc11` (2026-08-16, Đặng Văn Vỹ)
- resolve_schema_context với công tắc full/retrieval — `77f718b` (2026-08-16, Đặng Văn Vỹ)
- Retriever + FullRetriever/LexicalRetriever + expand_by_foreign_keys — `cea9a92` (2026-08-15, Đặng Văn Vỹ)
- ConnectionProfile + permitted_tables là ranh giới bảo mật — `ed142ea` (2026-08-15, Đặng Văn Vỹ)
- render_schema() DDL thay cho json.dumps info_box — `6486ddc` (2026-08-15, Đặng Văn Vỹ)
- kiểu Table/Column bất biến + adapter từ info_box — `f502f54` (2026-08-15, Đặng Văn Vỹ)
- parse ground-truth table set from gold SQL — `3b9ac1d` (2026-08-15, Đặng Văn Vỹ)

**Sửa lỗi**

- finalize_node chịu được state None/sai kiểu, không ném lỗi — `ac65f33` (2026-09-01, Đặng Văn Vỹ)
- remove unreachable fixture in test_supervisor.py (task 6 review) — `3e982da` (2026-09-01, Đặng Văn Vỹ)
- conditional edge thành hàm thuần; node giữ việc ghi state (spec 5.5) — `6abe0cb` (2026-09-01, Đặng Văn Vỹ)
- fail closed, statement_timeout 30s→10s, trần 50k dòng — `05b24f0` (2026-09-01, Đặng Văn Vỹ)
- spawn + env rỗng — ranh giới là tiến trình, không phải namespace — `55aba8e` (2026-09-01, Đặng Văn Vỹ)
- dọn ba lỗ đã park từ review Plan A — `6af3233` (2026-09-01, Đặng Văn Vỹ)
- tên cột không mang nghĩa thì luôn low, bất kể model chấm gì — `4601a03` (2026-08-27, Đặng Văn Vỹ)
- chia lô cột cho bảng rộng, annotate có ngân sách riêng — `da433e3` (2026-08-26, Đặng Văn Vỹ)
- resolve four BIRD loader bugs found against real Postgres — `749bdd7` (2026-08-23, Đặng Văn Vỹ)
- chú giải lẫn chữ Hán bị hạ xuống low, vào hàng đợi duyệt — `1df5dd6` (2026-08-23, Đặng Văn Vỹ)
- gấp định danh SQL đúng quy tắc Postgres, không hạ hết về chữ thường — `5337fd2` (2026-08-22, Đặng Văn Vỹ)
- khoá ngoại lùi về NOT VALID khi dữ liệu nguồn vi phạm — `7cd8480` (2026-08-22, Đặng Văn Vỹ)
- sáu lỗi round 3 — N1-N6 từ review lại — `2dfb752` (2026-08-19, Đặng Văn Vỹ)
- sáu lỗi Important/Minor từ review toàn nhánh — `4709459` (2026-08-19, Đặng Văn Vỹ)
- ba lỗi ở khớp nối giữa các bước onboarding — `1d22aa1` (2026-08-19, Đặng Văn Vỹ)
- verify — no bare tracebacks, grants vs annotation in the report — `3d31291` (2026-08-18, Đặng Văn Vỹ)
- reject --grant values missing '=' instead of building a phantom empty grant — `5d6ddd3` (2026-08-18, Đặng Văn Vỹ)
- lọc quyền trước khi tìm, không phải sau (spec 4.1) — `f476d90` (2026-08-18, Đặng Văn Vỹ)
- profile_store — ValueError no longer echoes the password — `45f1134` (2026-08-18, Đặng Văn Vỹ)
- profile_store — cleartext-password leak via unencoded reserved chars — `cfe8912` (2026-08-18, Đặng Văn Vỹ)
- _parse chỉ thử dấu ngoặc mở ra object thật, có trần — `c21441f` (2026-08-18, Đặng Văn Vỹ)
- thứ tự ghép chú giải không phụ thuộc PYTHONHASHSEED — `65f402d` (2026-08-18, Đặng Văn Vỹ)
- _parse quét mọi dấu { ứng viên; đếm lỗi theo văn bản rỗng — `62d303d` (2026-08-18, Đặng Văn Vỹ)
- giữ chú giải cột do người viết khi model không nhắc tới cột — `b6ad4bc` (2026-08-18, Đặng Văn Vỹ)
- fix comma placement in column descriptions and collapse whitespace — `a0a6054` (2026-08-17, Đặng Văn Vỹ)
- row_count -1 (chưa ANALYZE) chuẩn hoá về None — `1bbe725` (2026-08-17, Đặng Văn Vỹ)
- schema-qualify row_count, sql.Identifier quoting, DB-free identifier tests — `323c9e6` (2026-08-17, Đặng Văn Vỹ)
- sample_rows dùng fullmatch để chặn định danh có \n cuối — `68d338c` (2026-08-17, Đặng Văn Vỹ)
- schema_fingerprint hashes Column.is_generated — `4613001` (2026-08-16, Đặng Văn Vỹ)
- close cross-statement CTE permission bypass; structure error — `622f540` (2026-08-16, Đặng Văn Vỹ)
- wiring test actually calls run_graph; close {task}/{query} gap — `30f912f` (2026-08-16, Đặng Văn Vỹ)
- guard explain_query_plan, block writes, prove guard wiring by test — `57b9002` (2026-08-16, Đặng Văn Vỹ)
- FROM inside EXTRACT/TRIM/SUBSTRING/OVERLAY is not a table source — `19d69a1` (2026-08-16, Đặng Văn Vỹ)
- Handle Vietnamese đ/Đ (D with stroke) in tokenizer — `2f97d86` (2026-08-16, Đặng Văn Vỹ)
- Handle Vietnamese diacritics in LexicalRetriever tokenizer — `0025534` (2026-08-16, Đặng Văn Vỹ)
- deep-freeze ConnectionProfile.grants against widening — `6adfc35` (2026-08-15, Đặng Văn Vỹ)
- enforce foreign_keys immutability at construction — `b61384d` (2026-08-15, Đặng Văn Vỹ)

**Tái cấu trúc**

- mẫu số tiến độ dẫn xuất từ review_rows — `c17c93b` (2026-08-18, Đặng Văn Vỹ)
- tách kỹ năng chung khỏi đặc thù schema; schema xuống cuối prompt — `93bcf9f` (2026-08-16, Đặng Văn Vỹ)

**Tài liệu**

- cập nhật Plan B theo main hậu-merge Plan A — `8809e94` (2026-09-01, Đặng Văn Vỹ)
- sửa mô tả openai ở NGUỒN sinh, không sửa trong khối AUTO — `0e1f02d` (2026-08-19, Đặng Văn Vỹ)
- Plan B — cứng hóa ngân sách thời gian và cô lập sandbox, 12 task — `70dba92` (2026-08-17, Đặng Văn Vỹ)
- Plan A — đường onboarding, 12 task — `4fd4557` (2026-08-16, Đặng Văn Vỹ)
- lộ trình tới bản đóng gói giao khách — `f8fa0cb` (2026-08-16, Đặng Văn Vỹ)
- mục 6.2.1 — điều kiện pháp lý khi dùng ba benchmark — `49da143` (2026-08-16, Đặng Văn Vỹ)
- mục 4 nói đúng thứ code làm; dời thiết kế lại Retriever sang pha 3 — `0dfe041` (2026-08-16, Đặng Văn Vỹ)
- bộ tài liệu dự án + đường ống tự cập nhật theo commit — `53dec53` (2026-08-16, Đặng Văn Vỹ)
- correct the unmeasured 140 token/bảng assumption — `cc8f2b7` (2026-08-16, Đặng Văn Vỹ)
- disclose lexical baseline's bimodal context-size distribution — `c4a259e` (2026-08-16, Đặng Văn Vỹ)
- plan triển khai — đường ống schema context (pha 0+1) — `642bc51` (2026-08-15, Đặng Văn Vỹ)
- spec — tách permitted_tables khỏi retrieved_tables; ghi kết quả spike — `9bfeb37` (2026-08-15, Đặng Văn Vỹ)
- spec — agent chạy nội bộ trên schema của khách hàng — `4c90f63` (2026-08-15, Đặng Văn Vỹ)
- spec — chuẩn bị đường nối cho đa khách hàng — `20249d8` (2026-08-12, Đặng Văn Vỹ)
- spec hardening production + tách tool qua MCP — `f2c4a08` (2026-08-12, Đặng Văn Vỹ)
- README theo Best-README-Template; bỏ AGENTS.md khỏi remote — `d18f115` (2026-05-19, Đặng Văn Vỹ)

**Kiểm thử**

- fixture mật khẩu không còn giống bí mật thật — `03ab786` (2026-08-27, Đặng Văn Vỹ)
- cover _resolver's full/lexical paths on mini_dataset — `6a49d76` (2026-08-16, Đặng Văn Vỹ)
- assert rendered system prompt has no surviving placeholder — `bddc095` (2026-08-16, Đặng Văn Vỹ)
- make schema-identifier exclusions per file, not global — `80360bc` (2026-08-16, Đặng Văn Vỹ)
- derive schema-identifier denylist from real schema fixture — `0094c48` (2026-08-16, Đặng Văn Vỹ)
- tighten regression guards and add realistic case — `875635a` (2026-08-15, Đặng Văn Vỹ)

**Hạ tầng & công cụ**

- ignore profile-noann (mốc đo, không phải nguồn) — `5433c78` (2026-08-23, Đặng Văn Vỹ)
- ghim sha256 của BIRD dev.zip — `7cfa6ee` (2026-08-22, Đặng Văn Vỹ)
- ignore .claude/worktrees/ (worktree cô lập) — `3d2b768` (2026-08-15, Đặng Văn Vỹ)

**Khác**

- gom import HUMAN lên khối import đầu file — `2176559` (2026-08-18, Đặng Văn Vỹ)
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
| Commit nguồn gần nhất | `ac65f33` — fix(graph): finalize_node chịu được state None/sai kiểu, không ném lỗi |
| Tác giả | Đặng Văn Vỹ |
| Ngày commit | 2026-09-01 |
| Số commit nguồn | 110 |
| Sinh bởi | `scripts/update_docs.py` (hook `post-commit`) |

<!-- AUTO:end id=stamp -->
