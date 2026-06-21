# Kết quả Nghiên cứu — Ghost (Hider) 24127192

## Tóm tắt Lab 1: Hide and Seek Arena

**Mục tiêu:** Ghost phải sống sót tối đa 200 bước trước Pacman (tốc độ 2, Ghost tốc độ 1).
Ghost thắng nếu tồn tại đến hết `max_steps`, Pacman thắng nếu bắt được Ghost
(Manhattan distance < 2). Hai bên di chuyển **đồng thời**, thông tin hoàn hảo.

**Tiêu chí chấm điểm:**
- Điểm hoàn thiện giải thuật: 3đ
- Xếp hạng bài nộp ban đầu: tối đa 3đ
- Xếp hạng bài nộp tối ưu: tối đa 4đ
- **Tie-break:** Chênh lệch giữa số bước trung bình Pacman bắt được Ghost và Ghost sống sót.
  Ghost cần tối đa hóa thời gian sống, Pacman cần tối thiểu hóa thời gian bắt.

**Ràng buộc kỹ thuật:** 1 giây/bước, 128MB RAM, chỉ dùng `numpy, pandas, scipy, gurobi`.

---

## Paper 1: Carmel & Markovitch (1996) — Opponent Modeling in Multi-Agent Systems

**Ý tưởng chính:** Mô hình hóa chiến lược đối thủ như một DFA (finite automaton). Học
hành vi đối thủ từ chuỗi input/output quan sát được. Dự đoán nước đi tiếp theo và chơi
tối ưu dựa trên mô hình đó. Khi dự đoán sai, cập nhật lại mô hình.

**Ứng dụng cho Ghost:**
- Quan sát lịch sử di chuyển của Pacman để phát hiện mẫu hành vi:
  - "Đuổi thẳng A\*" → Pacman đi theo đường ngắn nhất đến Ghost
  - "Dự đoán tuyến tính" → Pacman dự đoán vị trí tiếp theo = vị trí hiện tại + vector vận tốc
  - "Interception" → Pacman nhắm đến ngã rẽ Ghost sắp đi qua thay vì vị trí hiện tại
- Khi phát hiện Pacman dùng linear prediction: Ghost đi zigzag (vuông góc) để phá vỡ dự đoán
- Duy trì confidence score cho mô hình; reset khi hành vi Pacman thay đổi

**Cách triển khai cụ thể:**
1. Lưu 5-10 vị trí gần nhất của Pacman, tính vector hướng di chuyển
2. Nếu Pacman duy trì 1 hướng ≥ 3 bước liên tiếp → dự đoán Pacman sẽ tiếp tục hướng đó
3. Ghost chọn nước đi vuông góc với hướng dự đoán để tối đa hóa khoảng cách sau 2 bước

---

## Paper 2: Teng (2026) — Reverse Engineering Pac-Man Ghost AI

**Ý tưởng chính:** Blinky (ghost đỏ) chỉ dùng greedy Manhattan-distance minimization nhưng
tạo ra hành vi có vẻ chiến lược nhờ ràng buộc của mê cung. Chuyển động tưởng "thông minh"
thực chất là topology mê cung + luật đơn giản.

**Ứng dụng cho Ghost:**
- Ghost có thể khai thác topology mê cung tốt hơn Pacman:
  - **Precompute junction graph:** Xác định tất cả ngã rẽ (ô có ≥ 3 lối ra), đo độ dài hành lang
    nối giữa các ngã rẽ
  - **Dead-end depth map:** Với mỗi ô, tính độ sâu vào ngõ cụt. Tuyệt đối tránh ngõ cụt khi
    Pacman ở gần
  - **Loop zones:** Xác định vùng có ≥ 2 đường đi giữa mọi cặp ô → Ghost có thể lòng vòng vô hạn
- **Nguyên tắc an toàn:** Không vào hành lang nếu Pacman có thể đến đầu ra trước Ghost
- Khi bị dồn, ưu tiên di chuyển vào loop zone để có nhiều lựa chọn thoát

**Cách triển khai cụ thể:**
1. Trong `__init__`, quét toàn bộ map một lần: phân loại mỗi ô là junction (≥3 exits),
   corridor (2 exits), dead-end (1 exit), hoặc pocket (0 exits)
2. Với mỗi corridor, xác định 2 đầu ra. Nếu Pacman gần 1 đầu ra, không vào corridor
3. Xây dựng đồ thị ngã rẽ (junction graph) để tra cứu O(1) khoảng cách giữa các ngã rẽ

---

## Paper 3: Yannakakis & Hallam (2004) — Evolving Opponents for Interesting Games

**Ý tưởng chính:** Đối thủ thú vị nhất KHÔNG phải là đối thủ tối ưu nhất. Công thức độ hấp dẫn:

```
I = γT + δS + εE[Hn]
```

- **T:** Chênh lệch giữa thời gian bắt trung bình và tối đa (càng lớn càng thú vị)
- **S:** Phương sai thời gian bắt (càng nhiều biến động càng thú vị)
- **Hn:** Entropy chuẩn hóa của phân phối ô đã ghé thăm (di chuyển càng đa dạng càng thú vị)

**Ứng dụng cho Ghost (nghịch đảo để sinh tồn):**
- Ghost KHÔNG nên luôn chọn nước đi "tối ưu toán học" — điều đó làm hành vi dễ đoán
- Thêm **controlled randomness** vào top-N nước đi an toàn → tăng entropy Hn, giảm khả năng
  Pacman học được mẫu hành vi
- **Phased strategy:** Chia game thành 3 giai đoạn:
  - Giai đoạn 1 (step 1-60): Bảo thủ — tối đa khoảng cách, tránh mọi rủi ro
  - Giai đoạn 2 (step 61-140): Cân bằng — kết hợp khoảng cách + entropy + kiểm soát vùng
  - Giai đoạn 3 (step 141-200): Sinh tồn — minimax chặt, ưu tiên loop zone
- **Cell visit frequency:** Theo dõi số lần ghé thăm mỗi ô, ưu tiên ô ít ghé thăm hơn trong số
  các ô an toàn tương đương → tăng entropy, giảm predictability

**Cách triển khai cụ thể:**
1. Duy trì dict `cell_visits` đếm số lần Ghost đi qua mỗi ô
2. Trong số top-3 nước đi an toàn, chọn ngẫu nhiên với trọng số:
   - 50% × điểm an toàn + 30% × (1 / (1 + cell_visits)) + 20% × random noise
3. Chuyển đổi chiến lược dựa trên `step_number` và khoảng cách đến Pacman

---

## Đề xuất Cải tiến cho GhostAgent

### 1. Mô hình hóa Đối thủ + Phá vỡ Dự đoán (từ Carmel-Markovitch)
- Lưu 5 vị trí gần nhất của Pacman, phát hiện xu hướng di chuyển
- Nếu Pacman liên tục tiến thẳng về Ghost ≥ 3 bước: đổi hướng vuông góc
- Duy trì confidence score; reset khi pattern thay đổi

### 2. Phân tích Topology Mê cung Tĩnh (từ Teng)
- Precompute trong `__init__`:
  - **Junction graph:** Đồ thị nối các ngã rẽ (≥3 exits), trọng số = độ dài hành lang
  - **Dead-end depth map:** Khoảng cách từ mỗi ô đến miệng ngõ cụt
  - **Loop zones:** Các thành phần liên thông có ≥ 2 đường đi giữa mọi cặp ô
- Sử dụng để tra cứu O(1) trong `step()` thay vì tính lại mỗi lần

### 3. Safe Flood Fill với Trọng số Tốc độ (cải tiến hiện tại)
- Hiện tại: `gd[cell] < ceil(pd[cell] / 2)` → ô an toàn nếu Ghost đến trước Pacman tốc độ 2
- Cải tiến: thêm margin an toàn 2-3 bước. Ô chỉ thực sự an toàn nếu:
  `gd[cell] + MARGIN < ceil(pd[cell] / 2)`
- Ưu tiên ô vừa an toàn vừa khiến Pacman phải đi xa nhất

### 4. Chiến lược Theo Giai đoạn (từ Yannakakis)
- **Giai đoạn 1 (step 1-60):** Mở đầu bảo thủ — trọng số cao cho khoảng cách, tránh ngõ cụt
- **Giai đoạn 2 (step 61-140):** Cân bằng — kết hợp khoảng cách + vùng an toàn + junction
- **Giai đoạn 3 (step 141-200):** Sinh tồn — minimax sâu hơn, ưu tiên loop zone, tránh rủi ro

### 5. Đa dạng hóa Di chuyển (từ Yannakakis)
- Trong top-3 nước đi an toàn, áp dụng weighted random
- Trọng số = 0.5 × distance_score + 0.3 × cell_entropy + 0.2 × random
- Theo dõi `cell_visits` toàn cục (reset mỗi game). Ưu tiên ô ít ghé thăm

### 6. Cải tiến Chống Dao động (Anti-Oscillation)
- Hiện tại: cấm cứng 6 ô gần nhất (có thể ép chọn nước đi tồi)
- Cải tiến: phạt mềm tỷ lệ với độ gần đây, đánh đổi với điểm an toàn
- Cho phép quay lại nếu lựa chọn thay thế là chắc chắn bị bắt

---

## Kết luận

Ba paper cung cấp góc nhìn bổ trợ cho Ghost:
1. **Carmel-Markovitch:** Mô hình hóa đối thủ → phá vỡ dự đoán
2. **Teng:** Khai thác topology mê cung → tránh bẫy, tận dụng loop zone
3. **Yannakakis:** Đa dạng hóa hành vi → giảm predictability, tăng entropy

Ưu tiên triển khai trước mắt: **Phân tích topology tĩnh** (từ Teng) vì chi phí
thấp (tính 1 lần) và **Chiến lược theo giai đoạn** (từ Yannakakis) vì dễ tích hợp.
Sau đó đến **Opponent modeling** (Carmel-Markovitch) và **Đa dạng hóa di chuyển**.
