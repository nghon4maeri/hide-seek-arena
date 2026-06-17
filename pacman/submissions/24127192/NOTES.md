# Ghost Agent Optimization Log

## Achieve 92% Winrate (Speed-2 Pacman Evasion)

Để đạt được tỉ lệ thắng 92% một cách ổn định, một số cải tiến kỹ thuật cốt lõi đã được áp dụng vào thuật toán Alpha-Beta search của Ghost Agent:

### 1. Tăng tốc độ tìm kiếm với Precomputed Adjacency List
Thay vì liên tục tính toán các bước đi hợp lệ (`_apply`, `_valid`, v.v.) trong hàng triệu node đệ quy của Alpha-Beta, toàn bộ cấu trúc đồ thị (bản đồ lưới) đã được tính toán sẵn một lần ở hàm `init`. 
- **Cải tiến:** Khởi tạo `self.adj` lưu trữ mọi `(Move, next_position)` hợp lệ cho từng cell.
- **Hiệu quả:** Thuật toán chạy nhanh gấp 2-3 lần, cho phép hệ thống mở rộng `max_d` (độ sâu tìm kiếm) lên từ 8 đến 12 ply (6 lượt) mà không bị vi phạm giới hạn thời gian chạy (`TIME_BUDGET = 0.75s`). Tầm nhìn xa này giúp Ghost nhận biết các bẫy chết sớm hơn.

### 2. Thuật toán phân tích Cửa tử Hành lang (Corridor Threat Heuristic)
Với tốc độ của Pacman (vận tốc 2), Ghost (vận tốc 1) cực kỳ dễ bị bắt nếu đi vào các hành lang thẳng hoặc ngõ cụt quá sâu. Ghost trước đây thường xuyên mắc kẹt ở "horizon effect" (không nhìn thấy cái chết ở ngã rẽ cách nó 5-6 turn).
- **Cải tiến:** Mở rộng logic `_de_pen`. Ghost phân tích khoảng cách từ bản thân tới ngã rẽ an toàn gần nhất (`jd_val`) và đối chiếu với khoảng cách từ Pacman (`p_val`). 
- **Toán học đằng sau:** Nếu Ghost đi vào một đoạn hành lang, và tính theo vận tốc 2 của Pacman mà Pacman đến được điểm cuối trước (hoặc cùng lúc) với Ghost (`p_val <= jd_val + 1`), thì đoạn hành lang đó lập tức bị đánh dấu là `-50000.0` (Fatal Trap).
- **Hiệu quả:** Ghost chủ động từ chối chui vào mọi hành lang nguy hiểm, tự động chọn cách ôm cua ở các ngã ba lớn (Hub) an toàn hơn.

### 3. Move Ordering Tối Ưu
- **Cải tiến:** Hàm `_order` ưu tiên sắp xếp các node dựa trên khoảng cách BFS và kết hợp chặt chẽ với án phạt từ Corridor Threat Heuristic (`de = -20000.0` với fatal trap). 
- **Hiệu quả:** Việc để trap ở vị trí xếp hạng cuối cùng bảo toàn được tỉ lệ cắt tỉa cành (pruning) ở mức tối đa cho Alpha-Beta, giúp tốc độ eval không bị sụt giảm.
