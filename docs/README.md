# ADBA — Bộ tài liệu dự án

Tài liệu của **ADBA (Autonomous Data & Business Intelligence Agent)** — hệ thống đa-agent
trên LangGraph nhận câu hỏi tiếng Việt/Anh về dữ liệu doanh nghiệp và trả về insight có
cấu trúc (finding · evidence · action) kèm bảng và biểu đồ.

Bộ tài liệu chia theo **ba câu hỏi khác nhau, ba nhóm người đọc khác nhau**.

## 1. Tổng quan & quản lý — *"Tại sao làm, và kế hoạch ra sao?"*

| Tài liệu | Nội dung | Đọc khi |
|---|---|---|
| [PRD / Project Charter](01-overview/PRD.md) | Bài toán, mục tiêu, phạm vi / ngoài phạm vi, user persona, tiêu chí thành công | Bắt đầu tìm hiểu dự án, hoặc tranh luận xem một yêu cầu có thuộc phạm vi không |
| [Timeline & Roadmap](01-overview/ROADMAP.md) | Milestone, sprint backlog, RACI, checkpoint và phương án dự phòng | Lập kế hoạch sprint, phân công, báo cáo tiến độ |
| [Release Notes & Changelog](01-overview/CHANGELOG.md) | Lịch sử phiên bản, tính năng mới, sửa lỗi — **sinh tự động từ git** | Cần biết cái gì đã đổi giữa hai phiên bản |

## 2. Kỹ thuật & kiến trúc — *"Hệ thống được xây thế nào?"*

| Tài liệu | Nội dung | Đọc khi |
|---|---|---|
| [System Architecture](02-architecture/SYSTEM_ARCHITECTURE.md) | Sơ đồ HLA, tech stack kèm lý do chọn, kiến trúc triển khai (Docker/CI-CD), biến môi trường | Onboard dev mới, thay đổi thành phần hạ tầng |
| [Database & Data Pipeline](02-architecture/DATA_DESIGN.md) | ERD, từng bảng/cột/ràng buộc, Data Flow Diagram, đường dữ liệu huấn luyện | Viết SQL, đổi schema, dựng lại dữ liệu |
| [API & Contract](02-architecture/API.md) | Contract nội bộ (`run_graph`, `MultiAgentState`), contract JSON của LLM, HTTP endpoint và lộ trình OpenAPI | Tích hợp ADBA vào hệ thống khác |

## 3. Luồng xử lý & nghiệp vụ — *"Dữ liệu và trải nghiệm đi như thế nào?"*

| Tài liệu | Nội dung | Đọc khi |
|---|---|---|
| [Sequence Diagrams](03-flows/SEQUENCE_DIAGRAMS.md) | Sơ đồ trình tự cho truy vấn đơn giản, đa bước, và vòng self-repair | Debug một lần chạy, hình dung ai gọi ai |
| [Business Logic & Rules](03-flows/BUSINESS_RULES.md) | Quy tắc nghiệp vụ (doanh thu, tồn kho, HR), edge case, chiến lược xử lý lỗi | Sửa logic, viết test, giải thích một con số |
| [Model & Agent Architecture](03-flows/AGENT_ARCHITECTURE.md) | Prompting strategy, LangGraph state, fine-tuning LoRA, bộ metric đánh giá | Chỉnh prompt, huấn luyện lại, đọc kết quả eval |

## Tài liệu liên quan (đã có từ trước)

- [Đặc tả dự án (bản dài, VI)](ADBA_Project_Specification_vi.md) — bản đặc tả gốc 12 mục
- [Phân tích multi-agent](multiagent_analysis.md)
- [Spec: hardening production + tách tool qua MCP](superpowers/specs/2026-08-12-adba-mcp-hardening-design.md)
- [Spec: agent chạy nội bộ trên schema khách hàng](superpowers/specs/2026-08-15-adba-multi-schema-onprem-design.md)
- [Plan: đường ống schema context](superpowers/plans/2026-08-15-schema-context-pipeline.md)

---

## Tài liệu tự cập nhật theo commit

Những phần **rút ra được từ mã nguồn** không viết tay: bảng dữ liệu, biến môi trường,
tham số agent, kết quả eval, danh sách test, changelog. Chúng nằm trong các *khối AUTO*:

```markdown
<!-- AUTO:begin id=db-tables -->
...nội dung do script sinh — đừng sửa tay, sẽ bị ghi đè...
<!-- AUTO:end id=db-tables -->
```

Mỗi lần `git commit`, hook `post-commit` chạy `scripts/update_docs.py`, ghi lại các khối
này và tạo commit phụ `docs(auto): …` nếu có thay đổi.

**Cài một lần cho mỗi bản clone:**

```bash
bash scripts/install_docs_hooks.sh
```

Chi tiết cơ chế, cách thêm khối mới, cách xử lý sự cố: [DOCS_AUTOMATION.md](DOCS_AUTOMATION.md).

---

## Quy ước

- **Ngôn ngữ**: tiếng Việt cho văn xuôi, tiếng Anh cho định danh kỹ thuật (`sql_agent`, `info_box`).
- **Sơ đồ**: Mermaid nhúng trực tiếp — render được trên GitHub, diff được như text.
- **Một sự thật một chỗ**: số liệu nào lấy được từ code thì để khối AUTO sinh, không chép tay.
- **Commit message**: theo Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`) —
  changelog dựa vào tiền tố này để phân nhóm.

<!-- AUTO:begin id=stamp -->

| Trường | Giá trị |
|---|---|
| Commit nguồn gần nhất | `55aba8e` — fix(sandbox): spawn + env rỗng — ranh giới là tiến trình, không phải namespace |
| Tác giả | Đặng Văn Vỹ |
| Ngày commit | 2026-09-01 |
| Số commit nguồn | 101 |
| Sinh bởi | `scripts/update_docs.py` (hook `post-commit`) |

<!-- AUTO:end id=stamp -->
