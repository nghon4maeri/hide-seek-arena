# Kết quả Nghiên cứu — Pacman (Seeker) 24127561

## Tóm tắt Lab 1: Hide and Seek Arena

**Mục tiêu:** Pacman phải bắt Ghost càng nhanh càng tốt. Pacman tốc độ 2 (đi thẳng
nhiều ô), Ghost tốc độ 1. Pacman thắng khi Manhattan distance < 2. Hai bên di chuyển
**đồng thời**, thông tin hoàn hảo.

**Tiêu chí chấm điểm:**
- Điểm hoàn thiện giải thuật: 3đ
- Xếp hạng bài nộp ban đầu: tối đa 3đ
- Xếp hạng bài nộp tối ưu: tối đa 4đ
- **Tie-break:** Chênh lệch giữa số bước trung bình Pacman bắt được Ghost và Ghost sống sót.
  Pacman cần tối thiểu hóa thời gian bắt, Ghost cần tối đa hóa thời gian sống.

**Ràng buộc kỹ thuật:** 1 giây/bước, 128MB RAM, chỉ dùng `numpy, pandas, scipy, gurobi`.

---

## Paper 1: Carmel & Markovitch (1996) — Opponent Modeling in Multi-Agent Systems

**Ý tưởng chính:** Mô hình hóa chiến lược đối thủ như một DFA (finite automaton). Học
hành vi từ input/output quan sát được. Dự đoán nước đi tiếp theo và tối ưu hóa phản ứng.
Khi dự đoán sai, cập nhật mô hình.

**Ứng dụng cho Pacman:**
- Quan sát lịch sử di chuyển của Ghost để phát hiện mẫu hành vi:
  - "Chạy thẳng đến ô xa nhất" → Ghost dùng A\* đến ô xa Pacman nhất
  - "Ưu tiên ngã rẽ" → Ghost luôn chọn hướng có nhiều lối thoát
  - "Lòng vòng trong loop zone" → Ghost di chuyển tuần hoàn trong vùng an toàn
- Dự đoán Ghost sẽ chạy đến ô nào tiếp theo → Pacman chặn trước, không đuổi theo sau
- Khi Ghost thay đổi hành vi (từ chiến lược sang chiến thuật), phát hiện và thích ứng

**Cách triển khai cụ thể:**
1. Lưu 5-8 vị trí gần nhất của Ghost, tính vector escape ưu tiên
2. Nếu Ghost duy trì 1 hướng ≥ 3 bước → Pacman nhắm đến ngã rẽ phía trước Ghost
3. Nếu Ghost đảo hướng liên tục (dao động) → Pacman áp sát trực tiếp

---

## Paper 2: Teng (2026) — Reverse Engineering Pac-Man Ghost AI

**Ý tưởng chính:** Blinky (ghost đỏ) chỉ dùng greedy Manhattan-distance minimization nhưng
tạo ra hành vi có vẻ chiến lược nhờ ràng buộc của mê cung. Truy đuổi "thông minh" thực
chất là topology mê cung + luật đơn giản.

**Ứng dụng cho Pacman (nghịch đảo):**
- A\* đã tốt hơn greedy chase, nhưng insight là: **topology mê cung làm phần lớn công việc**
- **Interception > Pursuit:** Nhắm đến ngã rẽ Ghost **phải đi qua**, không phải vị trí hiện tại
- **Precompute chokepoint map:** Xác định các điểm nghẽn — hành lang hẹp, ngã rẽ quan trọng
- Khi Ghost ở gần, chuyển từ "đuổi" sang "kiểm soát vùng" — chiếm miệng hành lang

**Cách triển khai cụ thể:**
1. Trong `__init__`: xây dựng junction graph (đồ thị ngã rẽ) + chokepoint map
2. Với mỗi vị trí Ghost, tìm ngã rẽ gần nhất mà Ghost hướng đến → A\* đến ngã rẽ đó
3. Cache đường đi, chỉ tính lại khi Ghost lệch > 2 ô so với dự đoán

---

## Paper 3: Yannakakis & Hallam (2004) — Evolving Opponents for Interesting Games

**Ý tưởng chính:** Đối thủ tối ưu làm trò chơi nhàm chán. Công thức độ hấp dẫn:

```
I = γT + δS + εE[Hn]
```

- **T:** Chênh lệch giữa thời gian bắt trung bình và tối đa
- **S:** Phương sai thời gian bắt
- **Hn:** Entropy chuẩn hóa của phân phối ô đã ghé thăm

**Ứng dụng cho Pacman (nghịch đảo để bắt nhanh hơn):**
- Ghost sẽ cố gắng trở nên không thể đoán trước → **đừng đuổi theo đường thẳng**
- Ghost dùng safe-area heuristic để chọn nơi ẩn náu → **tấn công trực tiếp vào vùng an toàn đó**
- Hiểu được Ghost muốn tối đa hóa entropy → Pacman nên **thu hẹp vùng an toàn** thay vì
  đuổi theo Ghost một cách mù quáng

**Cách triển khai cụ thể:**
1. **Trap Pressure:** Với mỗi nước đi của Pacman, tính xem safe flood fill của Ghost bị thu hẹp
   bao nhiêu. Chọn nước đi tối đa hóa áp lực (thu hẹp vùng an toàn nhanh nhất)
2. Khi ở xa (>10 bước): dùng A\* đuổi theo. Khi ở gần (≤10 bước): chuyển sang zone-control
3. Kết hợp A\* + Trap Pressure: mỗi nước đi được chấm điểm bởi cả khoảng cách và áp lực bẫy

---

## Đề xuất Cải tiến cho PacmanAgent

### 1. Mô hình hóa Đối thủ (từ Carmel-Markovitch)
- Lưu 5-8 vị trí gần nhất của Ghost, phát hiện escape pattern
- Nếu Ghost dùng A\*-đến-ô-xa-nhất: dự đoán ô đó và chặn trước
- Nếu Ghost ưu tiên ngã rẽ: Pacman nhắm đến ngã rẽ tiếp theo trên đường Ghost
- Duy trì confidence score; nếu Ghost đổi pattern → reset mô hình

### 2. Lập kế hoạch Chặn đón (Interception) (từ Teng)
- Precompute junction graph của mê cung (ngã rẽ + hành lang nối)
- Thay vì A\* đến vị trí hiện tại của Ghost: A\* đến ngã rẽ Ghost đang hướng đến
- BFS distance map từ Ghost: tìm các ô Pacman có thể đến trước Ghost
- Cache A\* path; chỉ tính lại khi Ghost lệch > 2 ô

### 3. Trap Pressure — Áp lực Bẫy (từ Yannakakis — ứng dụng ngược)
- Với mỗi nước đi Pacman, tính: safe flood fill của Ghost bị thu hẹp bao nhiêu?
- Chọn nước đi tối đa hóa áp lực bẫy (giảm lựa chọn thoát của Ghost nhanh nhất)
- Kết hợp với A\*: khi ở gần, chuyển từ chase sang zone-control
- Trọng số kết hợp: 0.6 × khoảng cách + 0.4 × áp lực bẫy

### 4. Cache Đường đi
- Cache A\* path. Chỉ replan khi Ghost lệch đáng kể (> 2 ô so với dự đoán)
- Tái sử dụng các đoạn path đã cache khi có thể
- Giảm thời gian tính toán, dành ngân sách cho minimax sâu hơn

### 5. Chiến lược Khám phá (Fog of War Fallback)
- Khi không thấy Ghost: nhắm đến `last_seen_position` (đã triển khai)
- Thêm: ưu tiên ghé thăm ô có degree cao (ngã rẽ) để tối đa vùng quan sát
- Đánh dấu vùng đã khám phá, ưu tiên vùng chưa được khám phá

---

## Kết luận

Ba paper cung cấp góc nhìn bổ trợ cho Pacman:
1. **Carmel-Markovitch:** Mô hình hóa Ghost → dự đoán + chặn đón
2. **Teng:** Khai thác topology mê cung → interception, chokepoint
3. **Yannakakis:** Áp lực bẫy → thu hẹp vùng an toàn của Ghost

Ưu tiên triển khai trước mắt: **Interception Planning** (từ Teng) — thay vì A\* đến Ghost,
A\* đến ngã rẽ phía trước Ghost. Sau đó là **Trap Pressure** (từ Yannakakis) để gây
áp lực khi ở gần. Cuối cùng là **Opponent Modeling** (Carmel-Markovitch).
