# Tiến độ phát triển Blind Seek Agent

**Vai trò:** Developer Blind Seek Agent (24127561)

## Đã triển khai (Lab 1)

* Thuật toán A* pathfinding
* Ghi nhớ vị trí cuối cùng của Ghost (Last Seen Memory)
* Dự đoán hướng di chuyển Ghost (Interception Planning)
* Multi-step Movement (pacman_speed)

# Blind Seeker Agent (Lab 2) _New update
## Thuật toán sử dụng

* **A***: Tìm đường đi ngắn nhất trên bản đồ đã khám phá.
* **Memory Map**: Lưu lại các ô đã quan sát để xây dựng bản đồ theo thời gian.
* **Frontier Exploration**: Khám phá các vùng chưa biết khi không nhìn thấy Ghost.
* **Belief State**: Ước lượng xác suất vị trí của Ghost khi Ghost khuất tầm nhìn.
* **Opponent Modeling**: Học thói quen di chuyển của Ghost từ các lần quan sát.
* **Predictive Interception**: Dự đoán vị trí Ghost trong tương lai để chặn đầu thay vì chỉ đuổi theo.

## Cải tiến so với Lab 1

* Bổ sung **Memory Map** để hỗ trợ môi trường quan sát không đầy đủ.
* Xử lý trường hợp `enemy_position = None` khi Ghost không xuất hiện trong tầm nhìn.
* Thêm **Belief State** để tiếp tục theo dõi Ghost khi mất dấu.
* Kết hợp **Frontier Exploration** để tìm kiếm hiệu quả hơn khi chưa xác định được vị trí Ghost.
* Áp dụng **Opponent Modeling** để dự đoán hướng di chuyển của Ghost.
* Sử dụng **Predictive Interception** giúp Pacman bắt Ghost nhanh hơn.
* Khắc phục lỗi cập nhật mô hình học khi Ghost biến mất nhiều lượt liên tiếp.

## Kết quả

* Hoạt động tốt trong môi trường **Partial Observability**.
* Khám phá bản đồ hiệu quả hơn.
* Dự đoán vị trí Ghost chính xác hơn sau khi mất dấu.
* Giảm các bước di chuyển dư thừa và tăng khả năng bắt Ghost.

