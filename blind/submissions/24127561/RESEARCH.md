# Kết quả Nghiên cứu — Blind Seeker 24127561

## Tóm tắt Lab 2: Blind Adversary

**Mục tiêu:** Pacman phải bắt Ghost dưới partial observability. Tầm nhìn giới hạn
cross-shaped 5 ô / 4 hướng, bị chặn bởi tường. Ghost có thể ẩn bất kỳ lúc nào.

**Khác biệt chính với Lab 1:**
- `map_state` chứa -1 cho vùng chưa biết
- `enemy_position` có thể là None
- Phải duy trì memory map tích lũy
- Tìm đường trên thông tin không đầy đủ

---

## Hướng nghiên cứu cho Blind Mode

### 1. Memory Map + Optimistic A*
- Xây dựng bản đồ tích lũy từ observation qua các step
- A* chạy trên memory map: treat -1 (unseen) là optimistic (đi được)
- Cập nhật memory map mỗi step với observation mới

### 2. Frontier Exploration
- Xác định biên giới known/unknown (frontier)
- Ưu tiên di chuyển về phía frontier gần nhất
- Kết hợp với thông tin cuối cùng về Ghost

### 3. Belief State Prediction
- Duy trì phân phối xác suất vị trí Ghost
- Cập nhật belief với observation mới (Bayesian)
- Khi Ghost mất tích: propagate belief theo possible moves

### 4. Interception under Uncertainty
- Dự đoán hướng Ghost từ last known position
- Tính interception point dựa trên topology + belief
- Kết hợp A* đến interception point với exploration
