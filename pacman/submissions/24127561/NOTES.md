# Tiến độ phát triển Seek Agent

Vai trò: Developer Seek Agent (24127561)

## Đã triển khai

* Thuật toán tìm đường A* (A* Pathfinding)
* Ghi nhớ vị trí cuối cùng của Ghost
* Dự đoán vị trí tiếp theo của Ghost
* Hỗ trợ di chuyển nhiều ô trong một lượt (Multi-step Movement)
* Cơ chế khám phá ngẫu nhiên khi không tìm thấy mục tiêu

## Kết quả đạt được

* Bắt thành công Ghost của `example_student`
* Thời gian bắt trung bình: khoảng 6–8 bước
* Hỗ trợ tốc độ Pacman lớn hơn 1 (`pacman_speed > 1`)
* Hoạt động ổn định trên bản đồ mặc định của hệ thống

## Hướng cải tiến trong tương lai

* Chiến lược đánh chặn (Interception Strategy)
* Weighted A* để tối ưu hiệu năng tìm đường
* Ghi nhớ các khu vực đã khám phá
* Dự đoán nhiều bước di chuyển tiếp theo của Ghost
* Tối ưu hóa chiến lược truy đuổi trên bản đồ lớn