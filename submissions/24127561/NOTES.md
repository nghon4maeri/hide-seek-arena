# Tiến độ phát triển Seek Agent

**Vai trò:** Developer Seek Agent (24127561)

## Đã triển khai

* Thuật toán tìm đường A* (A* Pathfinding)
* Ghi nhớ vị trí cuối cùng của Ghost (Last Seen Memory)
* Dự đoán vị trí tiếp theo của Ghost dựa trên hướng di chuyển gần nhất
* Hỗ trợ di chuyển nhiều ô trong một lượt (Multi-step Movement)
* Cơ chế khám phá ngẫu nhiên khi không xác định được mục tiêu (Random Exploration Fallback)

## Kết quả đạt được

* Bắt thành công Ghost của `example_student`
* Thời gian bắt trung bình khoảng **6–8 bước**
* Hỗ trợ tốc độ Pacman lớn hơn 1 (`pacman_speed > 1`)
* Hoạt động ổn định trên bản đồ mặc định của hệ thống
* Giảm số bước truy đuổi so với phiên bản BFS ban đầu

## Thuật toán sử dụng

### A* Pathfinding

Pacman sử dụng thuật toán A* với khoảng cách Manhattan làm hàm heuristic để tìm đường ngắn nhất đến mục tiêu.

### Enemy Memory

Khi Ghost biến mất khỏi tầm nhìn, Pacman tiếp tục di chuyển đến vị trí cuối cùng quan sát được thay vì đứng yên.

### Future Position Prediction

Khi Ghost đang di chuyển, Pacman dự đoán vị trí tiếp theo của Ghost bằng công thức:

predicted_position = current_enemy_position + (current_enemy_position - previous_enemy_position)

Qua đó giúp Pacman chủ động truy đuổi thay vì chỉ đi theo vị trí hiện tại.

### Multi-step Movement

Pacman tận dụng tối đa giá trị `pacman_speed` để di chuyển nhiều ô liên tiếp theo cùng một hướng trong một lượt.

## Hướng cải tiến trong tương lai

* Chiến lược đánh chặn (Interception Strategy)
* Weighted A* để tối ưu hiệu năng tìm đường
* Ghi nhớ các khu vực đã khám phá (Exploration Memory)
* Dự đoán nhiều bước di chuyển tiếp theo của Ghost
* Tối ưu hóa chiến lược truy đuổi trên bản đồ lớn
* Phân tích hành vi Ghost để lựa chọn điểm chặn tối ưu
