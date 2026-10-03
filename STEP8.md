# BÁO CÁO PHÂN TÍCH HỆ THỐNG MEMORY CHO AI AGENT (DAY 17)

**Học viên:** Đào Đức Hải  
**Mã học viên:** 2A202602752  
**Khóa / Cohort:** Cohort 4 - Track 3: Memory Systems for AI Agent  

---

## 1. Kết quả Benchmark thực nghiệm

Toàn bộ số liệu dưới đây được đo đạc trực tiếp từ lệnh thực thi trên trạng thái sạch (`python src/benchmark.py` sau khi xóa thư mục `state/`):

### Bảng 1: Standard Benchmark (`data/conversations.json` - 10 hội thoại, user `dungct`)
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 1,778 | 15,124 | **0.00** | 0.59 | **0** | **0** |
| **Advanced** | 1,941 | 19,777 | **1.00** | 1.00 | **195** | **0** |

### Bảng 2: Long-Context Stress Benchmark (`data/advanced_long_context.json` - 16 lượt dài, user `dungct_stress`)
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 560 | **22,381** | **0.00** | 0.60 | **0** | **0** |
| **Advanced** | 1,246 | **14,760** | **1.00** | 1.00 | **139** | **27** |

---

## 2. Phân tích 4 câu hỏi cốt lõi (Theo Bước 8 & Rubric)

### Câu 1: Vì sao Advanced có Cross-session recall tốt hơn Baseline?
- **Số liệu chứng minh:** 
  - Ở cả hai bài benchmark, `Cross-session recall` của Advanced đạt tuyệt đối **1.00**, trong khi Baseline luôn là **0.00**.
  - `Memory growth (bytes)` của Baseline bằng **0** ở mọi bài test, trong khi Advanced ghi nhận **195 bytes** (`dungct`) và **139 bytes** (`dungct_stress`).
- **Cơ chế kỹ thuật trong code:**
  - **Baseline Agent** chỉ duy trì short-term memory thông qua dict `self.sessions[thread_id]`. Khi chuyển sang câu hỏi recall được thiết lập trên một `thread_id` mới hoàn toàn (`conv_id_recall_idx`), session của Baseline trống rỗng $\rightarrow$ Baseline phản hồi trung thực: *"Xin lỗi, tôi không có thông tin về bạn trong phiên làm việc này."* và đạt recall = 0.
  - **Advanced Agent** sở hữu persistent memory tách biệt (`UserProfileStore`). Mỗi khi người dùng chia sẻ fact ổn định, hàm `extract_profile_updates()` trích xuất và lưu ngay vào `state/profiles/<user>/User.md`. Khi nhận câu hỏi recall ở thread mới, hàm `_offline_response()` tra cứu trực tiếp từ file `User.md` độc lập với session chat, từ đó trả lời chính xác toàn bộ facts (Tên, Nơi ở hiện tại, Nghề nghiệp, Đồ uống/Món ăn yêu thích, Style trả lời).
- **Giới hạn / Lưu ý:** Nếu không tách biệt persistent memory ra file đĩa mà chỉ dựa vào bộ nhớ tóm tắt (summary) trong thread, agent sẽ có nguy cơ đánh mất fact khi thread bị đóng hoặc khi tóm tắt làm mờ chi tiết.

---

### Câu 2: Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?
- **Số liệu chứng minh:**
  - Ở Standard Benchmark (hội thoại ngắn ~10 lượt), `Prompt tokens processed` của Advanced (**19,777**) cao hơn Baseline (**15,124**), tương đương chênh lệch ~30.7%.
  - `Agent tokens only` của Advanced cũng nhỉnh hơn (1,941 vs 1,778).
- **Cơ chế kỹ thuật trong code:**
  - Ở hội thoại ngắn, tổng token của 10 lượt (~200 - 300 tokens) chưa bao giờ vượt qua ngưỡng `compact_threshold_tokens = 600`. Do đó, `Compactions = 0` ở cả hai agent (cơ chế nén chưa kích hoạt).
  - Tuy nhiên, trong mỗi lượt gọi `_estimate_prompt_context_tokens()`, Advanced Agent luôn phải nạp thêm ngữ cảnh cố định từ file `User.md` (khoảng 35 - 50 tokens) vào prompt context:
    $$\text{Prompt Tokens} = \text{tokens}(User.md) + \text{tokens}(summary) + \text{tokens}(messages)$$
  - Vì hội thoại ngắn chưa đủ dài để cơ chế nén phát huy tác dụng tiết kiệm, chi phí kéo theo của `User.md` khiến Advanced tốn nhiều prompt tokens hơn Baseline.

---

### Câu 3: Vì sao Compact Memory có lợi thế vượt trội ở hội thoại rất dài?
- **Số liệu chứng minh:**
  - Ở Stress Benchmark (16 lượt chứa nhiều văn bản tin tức dài), `Prompt tokens processed` của Baseline tăng vọt lên **22,381 tokens**, trong khi Advanced chỉ tiêu tốn **14,760 tokens** (giảm **34.05%** tải ngữ cảnh).
  - Advanced ghi nhận **27 lần compaction** (`Compactions = 27`).
- **Cơ chế kỹ thuật trong code:**
  - **Baseline Agent** giữ nguyên toàn bộ lịch sử trò chuyện. Qua mỗi lượt, lượng prompt context tích lũy theo cấp số cộng: lượt $N$ phải mang theo toàn bộ tin nhắn từ lượt $1$ đến lượt $N-1$, dẫn đến chi phí ngữ cảnh tăng phi mã.
  - **Advanced Agent** thông qua `CompactMemoryManager.append()` liên tục giám sát tổng lượng token. Khi vượt ngưỡng `threshold_tokens`, nó tự động trích xuất các tin nhắn cũ đưa vào hàm `summarize_messages()` để nén thành văn bản tóm tắt ngắn gọn, chỉ giữ lại đúng `keep_messages = 4` tin nhắn gần nhất nguyên văn.
  - **Điểm mấu chốt của Rubric:** Compact memory **tối ưu trực tiếp ở cột `Prompt tokens processed`** (lượng ngữ cảnh LLM phải đọc vào ở mỗi turn), chứ không nhằm giảm `Agent tokens only` (token do model sinh ra). Điều này giúp giảm đáng kể chi phí API và tránh hiện tượng tràn context window của mô hình.

---

### Câu 4: File memory tăng trưởng ra sao và các rủi ro đi kèm?
- **Số liệu chứng minh:**
  - File `User.md` tăng từ 0 lên 195 bytes (cho user `dungct` qua 10 phiên) và 139 bytes (cho user `dungct_stress`).
- **Phân tích rủi ro trong môi trường Production:**
  1. **Rủi ro phình to không kiểm soát (Unbounded Storage Growth):** Nếu người dùng liên tục tương tác qua hàng trăm phiên, file `User.md` có thể phình to lên hàng trăm kilobytes. Khi đó, chi phí token để nhồi `User.md` vào mỗi turn sẽ làm triệt tiêu hoàn toàn lợi ích tiết kiệm của compact memory.
  2. **Rủi ro ô nhiễm thông tin sai / nhiễu (Memory Poisoning):** Nếu bộ trích xuất quá nhạy cảm, các câu đùa (như *"product manager"*), địa điểm đi công tác tạm thời (*"Hà Nội"*), hoặc các phát ngôn ngẫu hứng sẽ bị lưu vĩnh viễn vào profile, gây hallucination hoặc trả lời sai trong tương lai.
  3. **Rủi ro xung đột dữ liệu (Fact Inconsistency):** Khi người dùng đính chính (chuyển từ Đà Nẵng sang Huế), nếu hệ thống chỉ nối chuỗi (`append`) thay vì cập nhật (`upsert`), file sẽ chứa cả hai thông tin đối lập nhau.

---

## 3. Phần mở rộng Kỹ thuật (Bonus: Mốc 90 - 100 điểm)

Để giải quyết các rủi ro nêu trên, hệ thống trong bài đã triển khai hai cơ chế mở rộng thực chiến:

### 1. Conflict Handling & Correction Resolution (Xử lý xung đột & Đính chính)
- **Vấn đề giải quyết:** Người dùng thay đổi thông tin cá nhân theo thời gian (ví dụ: chuyển nơi ở từ Đà Nẵng $\rightarrow$ Huế trong `conversations.json`, hoặc từ Huế $\rightarrow$ Đà Nẵng trong `advanced_long_context.json`; đổi nghề từ `backend engineer` $\rightarrow$ `MLOps engineer`).
- **Cách cài đặt:** `UserProfileStore` cài đặt phương thức `upsert_facts(user_id, new_facts)`. Khi có fact mới cùng khóa (ví dụ `Location` hoặc `Profession`), hệ thống tự động ghi đè giá trị cũ thay vì tạo mục mới, đảm bảo tính nhất quán độc nhất (single source of truth).
- **Cải thiện:** Giữ `Cross-session recall` đạt **1.00** tuyệt đối ngay cả ở các câu hỏi bẫy ("đâu mới là nghề nghiệp và nơi ở hiện tại?").
- **Rủi ro tạo ra cho hệ thống:** Có nguy cơ ghi đè mất lịch sử nếu người dùng chỉ đề cập đến một trạng thái tạm thời mà bộ trích xuất hiểu nhầm là đính chính chính thức.

### 2. Query & Noise Guardrails (Bộ lọc nhiễu & Phân loại Intent)
- **Vấn đề giải quyết:**
  - Ngăn chặn việc ghi nhận thông tin từ các lượt người dùng chỉ đặt câu hỏi kiểm tra (như *"Bạn có biết DũngCT không?"*, *"Mình tên gì?"*).
  - Loại bỏ các phát ngôn đùa (*"đùa với đồng nghiệp là chuyển sang product manager"*) hoặc thông tin ngắn hạn (*"bay ra Hà Nội họp 2 ngày"*).
- **Cách cài đặt:** Trong `extract_profile_updates()`:
  - Kiểm tra dấu `?` và nhận diện câu hỏi kiểm tra recall: nếu người dùng không dùng các vị từ khẳng định (`mình tên là`, `đang làm`, `nơi ở hiện tại là`...), hàm lập tức trả về `{}` để không ghi nhiễu.
  - Phân tích ngữ cảnh phủ định: nhận diện cụm *"chứ không phải nơi ở hiện tại"*, *"chỉ là câu đùa"*, *"đừng nói backend engineer nữa"* để chủ động bỏ qua thông tin sai.
- **Cải thiện:** Giữ kích thước file memory tinh gọn (dưới 200 bytes) và không làm suy giảm `Response quality`.
- **Rủi ro tạo ra cho hệ thống:** Logic lọc theo rule/heuristic có thể quá cứng nhắc, dẫn đến bỏ sót fact thật nếu người dùng diễn đạt theo cách quá khác lạ hoặc có cấu trúc câu phức tạp.

---

## 4. Quy ước Cấu hình Biến Môi trường (.env)

Hệ thống được thiết kế để chạy hoàn toàn offline (deterministic) khi không có API key. Nếu reviewer / giảng viên muốn thử nghiệm live mode với LLM thật, vui lòng cấu hình file `.env` tại thư mục gốc repo:

```ini
# Lựa chọn provider: openai | custom | gemini | anthropic | ollama | openrouter
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0

# API Keys tương ứng với provider được chọn:
OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=AIza...
# ANTHROPIC_API_KEY=sk-ant-...
# OPENROUTER_API_KEY=sk-or-...

# Cấu hình Local / Custom Endpoint nếu dùng Ollama hoặc vLLM:
# OLLAMA_BASE_URL=http://localhost:11434
# CUSTOM_BASE_URL=http://localhost:8000/v1
# CUSTOM_API_KEY=EMPTY

# Các ngưỡng điều khiển Compact Memory:
COMPACT_THRESHOLD_TOKENS=600
COMPACT_KEEP_MESSAGES=4
```

---

## 5. Bảng Tự đánh giá theo Rubric

| Mốc điểm | Tiêu chí Rubric | Trạng thái bài làm |
| :---: | :--- | :---: |
| **0 - 60** | Baseline chỉ nhớ trong thread, Advanced có `User.md`, có compact memory, cấu trúc repo chuẩn. | ✅ **ĐẠT** |
| **60 - 75** | Benchmark chạy cùng input cho cả 2 agent; 4 test case pytest pass 100%; bảng đủ 6 cột chuẩn. | ✅ **ĐẠT** |
| **75 - 90** | Có cả Standard & Stress Benchmark; stress làm lộ chi phí prompt của baseline; phân tích rõ bản chất compact tối ưu `Prompt tokens processed`. | ✅ **ĐẠT** |
| **90 - 100** | Có phần Bonus giải quyết vấn đề thực tế (Conflict handling & Noise guardrails), trả lời đủ 3 câu hỏi (vấn đề, cải thiện, rủi ro). | ✅ **ĐẠT** |
