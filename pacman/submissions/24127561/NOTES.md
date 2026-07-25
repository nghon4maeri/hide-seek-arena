# Tiến độ phát triển Seek Agent

**Sinh viên:** 24127561
**Vai trò:** Seek Agent Engineer

## Tổng quan

Agent được thiết kế cho vai trò **Pacman (Seeker)** trong môi trường Hide-and-Seek Arena.

Mục tiêu là tìm và bắt Ghost trong thời gian ngắn nhất bằng cách kết hợp:

* A* Pathfinding
* Dự đoán hướng di chuyển của Ghost
* Đánh chặn tại các nút giao (Choke-point Interception)
* Khóa mục tiêu tạm thời (Target Locking)
* Multi-step Movement
* Exploration có ghi nhớ vị trí đã đi

---

## Thuật toán đã sử dụng

### 1. A* Pathfinding

Agent sử dụng thuật toán A* với heuristic Manhattan Distance:

f(n) = g(n) + h(n)

Trong đó:

* g(n): chi phí từ vị trí hiện tại đến node n
* h(n): khoảng cách Manhattan tới mục tiêu

Ưu điểm:

* Tìm đường ngắn nhất
* Nhanh hơn BFS trên bản đồ lớn
* Giảm số node phải mở rộng

Độ phức tạp:

* Thời gian: O(N log N)
* Bộ nhớ: O(N)

---

### 2. Ghost Movement Prediction

Agent lưu:

* Vị trí Ghost hiện tại
* Vị trí Ghost ở lượt trước

Từ đó suy ra hướng di chuyển:

dr = current_row - previous_row
dc = current_col - previous_col

Dự đoán vị trí tiếp theo:

predicted = current + (dr, dc)

Điều này giúp Pacman đuổi tới nơi Ghost sắp đến thay vì vị trí hiện tại của Ghost.

---

### 3. Choke-point Interception

Thay vì luôn đuổi trực tiếp theo Ghost, Agent quét phía trước đường chạy của Ghost để tìm:

* Ngã ba
* Ngã tư
* Điểm giao hành lang
* Cửa vào ngõ cụt

Nếu Pacman có thể đến đó trước Ghost, Agent sẽ:

* Khóa mục tiêu
* Di chuyển tới điểm chặn

Chiến lược này giúp:

* Cắt đầu Ghost
* Giảm thời gian truy đuổi
* Hiệu quả hơn chase trực tiếp

---

### 4. Target Locking

Khi đã chọn một điểm chặn:

* Giữ mục tiêu trong 3 lượt
* Tránh đổi hướng liên tục khi Ghost đánh lừa

Lợi ích:

* Giảm hiện tượng dao động
* Tăng độ ổn định chiến thuật

---

### 5. Multi-step Movement

Nếu hệ thống cho phép:

pacman_speed > 1

Agent sẽ:

* Phân tích đường đi A*
* Gộp nhiều bước thẳng liên tiếp

Ví dụ:

RIGHT → RIGHT

sẽ trả về:

(Move.RIGHT, 2)

Thay vì:

(Move.RIGHT, 1)

hai lần liên tiếp.

Lợi ích:

* Tận dụng tối đa tốc độ của Pacman
* Giảm số lượt cần để bắt Ghost

---

### 6. Exploration Memory

Khi không nhìn thấy Ghost:

Agent chuyển sang chế độ khám phá.

Mỗi ô đã đi qua được lưu trong:

visited

Các ô chưa từng ghé thăm được ưu tiên cao hơn.

Điều này giúp:

* Giảm đi vòng lặp
* Tăng khả năng phát hiện Ghost
* Bao phủ bản đồ hiệu quả hơn

---

## Kết quả đạt được

### Smoke Test

* Pass thành công
* Không phát sinh lỗi runtime
* Agent tương thích với framework

### Arena Test

Kết quả thử nghiệm với:

* Seek: 24127561
* Hide: example_student

Thời gian bắt Ghost thường dao động:

* 6–8 bước

Một số trận tiêu biểu:

* 6 bước
* 6 bước
* 8 bước
* 10 bước

### Win Rate

Đối với Ghost mẫu của hệ thống:

* Win rate gần như 100%
* Không ghi nhận thất bại trong các lần benchmark đã chạy

---

## Ưu điểm

* Tìm đường tối ưu bằng A*
* Có khả năng dự đoán hướng chạy của Ghost
* Hỗ trợ đánh chặn thay vì chỉ truy đuổi
* Tận dụng tốc độ Pacman > 1
* Có cơ chế khám phá khi mất dấu mục tiêu
* Hoạt động ổn định trên bản đồ mặc định

---

## Hạn chế hiện tại

* Chỉ dự đoán 1 bước tương lai
* Chưa mô hình hóa chiến thuật dài hạn của Ghost
* Chưa học từ lịch sử các trận đấu
* Chưa đánh giá xác suất các hướng chạy khác nhau

---

## Hướng phát triển tiếp theo

### Weighted A*

Sử dụng:

f(n) = g(n) + w × h(n)

với:

w > 1

để tăng tốc độ tìm kiếm.

---

### Multi-step Prediction

Dự đoán:

* 2 bước
* 3 bước
* 5 bước

thay vì chỉ 1 bước.

---

### Probabilistic Tracking

Ước lượng xác suất Ghost xuất hiện tại từng khu vực khi mất dấu.

---

### Dynamic Interception

Tự động điều chỉnh điểm đánh chặn dựa trên:

* khoảng cách
* tốc độ
* cấu trúc mê cung

---

### Endgame Optimization

Khi khoảng cách tới Ghost nhỏ:

≤ 3 ô

chuyển sang chế độ truy đuổi trực tiếp để kết thúc trận đấu nhanh hơn.

---

## Kết luận

Agent hiện tại là một phiên bản Seek Agent nâng cao sử dụng:

* A*
* Ghost Prediction
* Choke-point Interception
* Target Locking
* Multi-step Movement
* Exploration Memory

Qua các thử nghiệm hiện tại, Agent bắt Ghost nhanh, ổn định và phù hợp để tiếp tục phát triển trong các vòng đánh giá tiếp theo.
