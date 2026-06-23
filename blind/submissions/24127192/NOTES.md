# Ghost Agent Optimization Log — Blind Mode

## Adaptations for Partial Observability (Lab 2)

Để đạt hiệu quả cao trong Blind mode, các cải tiến sau cần được áp dụng:

### 1. Memory Map cho BFS/A*
- Tất cả BFS distance map và A* pathfinding chạy trên `self.memory_map`
- Memory map tích lũy observation qua các step
- Treat -1 cells là optimistic traversable trong pathfinding

### 2. Pacman Position Uncertainty
- Khi Pacman không thấy: dùng last known position
- Dự đoán possible positions của Pacman sau N steps
- Điều chỉnh flee target dựa trên worst-case Pacman position

### 3. Safe Area Estimation under Partial Info
- Flood fill chỉ tính trên known cells (0)
- Giả định pessimistic: Pacman có thể ở bất kỳ unseen cell nào
- Chọn vùng an toàn dựa trên topology đã biết

### 4. Junction Strategy
- Ưu tiên ở gần junctions (≥3 exits) đã khám phá
- Tránh dead-ends đã biết
- Khi khám phá: ưu tiên hướng có nhiều exits nhất
