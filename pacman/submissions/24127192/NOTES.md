# GhostAgent (Hider) — NOTES

---

## 1. Hướng Triển Khai

### 1.1 Tổng Quan Chiến Lược

Agent Ghost được thiết kế theo kiến trúc **đa tầng (multi-layer)**, kết hợp nhiều thuật toán tìm kiếm cổ điển để tối ưu hóa khả năng sinh tồn trên bản đồ 21×21:

1. **BFS (Breadth-First Search):** Tính toán bản đồ khoảng cách chính xác từ mọi vị trí. Kết quả được **ghi nhớ (memoised)** theo vị trí nguồn vì bản đồ là tĩnh, giúp tránh tính toán lặp lại.

2. **Flood Fill (Safe Territory):** Phân tích lãnh thổ Voronoi với **điều chỉnh tốc độ Pacman speed-2** — một ô được tính là an toàn cho Ghost chỉ khi Ghost có thể đến trước Pacman (với tốc độ hiệu dụng `ceil(pd/2)`).

3. **Minimax với Alpha-Beta Pruning:**
   - Ghost là người **tối đa hóa** (maximiser), Pacman là người **tối thiểu hóa** (minimiser)
   - Giới hạn độ sâu từ 2-6 tùy theo khoảng cách đến Pacman
   - **Iterative deepening** — tăng dần độ sâu cho đến khi hết thời gian
   - **Alpha-Beta pruning** cắt bỏ các nhánh không ảnh hưởng đến kết quả
   - Mô phỏng Pacman di chuyển **speed-2 đường thẳng** trong cây tìm kiếm

4. **Greedy Move Ordering:** Sắp xếp các nước đi theo heuristic nhanh trước khi minimax, giúp alpha-beta cắt tỉa hiệu quả hơn.

5. **Phân tích Topo Bản Đồ (tĩnh, tính 1 lần):**
   - **Dead-end:** Ô có bậc ≤ 1 (cul-de-sac)
   - **Corridor:** Ô có bậc 2, hai lối đi thẳng hàng
   - **Junction:** Ô có bậc ≥ 3 (ngã ba/ngã tư)
   - **Dead-end branch depth:** Truy vết từ dead-end đến junction gần nhất
   - **Junction distance map:** Multi-source BFS từ tất cả junction
   - **Hub score:** Đánh giá mức kết nối giữa các junction (junction gần nhiều junction khác = hub tốt)

### 1.2 Pipeline Quyết Định

```
Bước 1: Phân tích topo bản đồ (1 lần duy nhất)
Bước 2: Cập nhật vị trí Pacman
Bước 3: Tính flee target (hub an toàn nhất, xa Pacman nhất)
Bước 4: Sắp xếp nước đi theo greedy heuristic
Bước 5: Iterative deepening minimax + alpha-beta
Bước 6: Trả về nước đi tốt nhất
```

### 1.3 Hàm Đánh Giá (Evaluation Function)

Hàm đánh giá kết hợp 12 đặc trưng với trọng số thích ứng theo **pha game** (early/mid/late) và **khoảng cách** đến Pacman:

| Đặc trưng | Vai trò |
|---|---|
| BFS distance | Ưu tiên giữ khoảng cách xa Pacman |
| Safe territory | Số ô Ghost kiểm soát (Voronoi speed-2) |
| Mobility | Số ô có thể đến trong 8 bước (flood fill) |
| Branching factor | Ưu tiên ô có nhiều lối đi |
| Escape routes | Số hướng thoát an toàn |
| Junction proximity | Thưởng khi gần ngã ba/ngã tư |
| Hub connectivity | Thưởng khi gần hub kết nối tốt |
| Control ratio | Tỷ lệ kiểm soát toàn bản đồ |
| Flee target | Thưởng khi di chuyển về phía mục tiêu chiến lược |
| Dead-end penalty | Phạt nặng khi trong nhánh dead-end |
| Corridor penalty | Phạt khi trong hành lang hẹp gần Pacman |
| Oscillation penalty | Phạt khi dao động giữa 2-3 ô |

### 1.4 Trọng Số Thích Ứng

- **Early game (step < 40):** Ưu tiên khoảng cách và tránh dead-end
- **Mid game (step 40-130):** Cân bằng giữa sinh tồn và kiểm soát lãnh thổ
- **Late game (step > 130):** Bảo thủ, tránh rủi ro, ưu tiên đường thoát
- **Proximity multiplier:** Khi Pacman rất gần (BFS ≤ 3), tăng gấp 3 lần phạt dead-end

---

## 2. Kết Quả Đã Đạt Được

### 2.1 Benchmark vs Example Pacman (Greedy)

| Chỉ số | Kết quả |
|---|---|
| Tỷ lệ thắng | **70%** (14/20 games) |
| Số bước trung bình | **154.7** bước |
| Số bước tối thiểu khi thua | 32 bước |
| Khi sống qua step 64 | **100% sống sót đến 200 bước** |

### 2.2 Arena Test Chính Thức

- Ghost **thắng** trong game mặc định (deterministic start)
- Sống sót 200 bước, khoảng cách cuối cùng = 3
- Thời gian mỗi bước < 0.15s (dư sức trong giới hạn 1 giây)
- Bộ nhớ sử dụng < 10MB (dư sức trong giới hạn 128MB)

### 2.3 Đặc Điểm Nổi Bật

- **Self-contained:** Toàn bộ code nằm trong 1 file `agent.py`, không cần thư viện ngoài ngoại trừ Python standard library
- **Cache hiệu quả:** BFS distance maps được ghi nhớ, topology phân tích 1 lần
- **Robust:** Xử lý fog of war (enemy_position = None), xử lý timeout gracefully
- **Deterministic:** Không sử dụng random, kết quả có thể tái tạo

---

## 3. Cải Tiến Trong Tương Lai

### 3.1 Cải Tiến Sớm (Short-term)

1. **Chiến lược thoát đầu game:** Cải thiện 6 game thua trong 20 game (đều xảy ra trong 32-64 bước đầu). Có thể thêm logic đặc biệt cho 10-20 bước đầu tiên để Ghost di chuyển về hướng an toàn nhất trước khi minimax hoạt động hiệu quả.

2. **Adaptive depth control:** Thay vì giới hạn độ sâu cứng, sử dụng thời gian còn lại để quyết định có nên đào sâu hơn không.

3. **Move ordering tốt hơn:** Sử dụng kết quả từ depth trước (iterative deepening) để sắp xếp nước đi cho depth tiếp theo (principal variation).

### 3.2 Cải Tiến Trung Hạn (Medium-term)

4. **Transposition table:** Lưu kết quả đánh giá cho các trạng thái đã gặp trong minimax để tránh tính toán lặp lại.

5. **Quenescence search:** Khi Pacman rất gần (dist < 4), tiếp tục tìm kiếm vượt quá giới hạn depth cho đến khi trạng thái "yên tĩnh" (không có capture ngay lập tức).

6. **Chiến lược đa pha:** Thêm logic chuyển đổi giữa "chạy trốn" và "tuần tra" dựa trên vị trí tương đối với Pacman.

### 3.3 Cải Tiến Dài Hạn (Long-term)

7. **Expectiminimax:** Mô hình hóa sự không chắc chắn trong hành vi của Pacman (thay vì giả sử Pacman chơi tối ưu).

8. **Pattern recognition:** Nhận diện mẫu di chuyển của Pacman để dự đoán hành vi tương lai.

9. **Multi-map adaptation:** Tự động điều chỉnh trọng số heuristic dựa trên đặc điểm bản đồ (kích thước, mật độ dead-end, v.v.).

10. **Opponent modeling:** Xây dựng mô hình đối thủ dựa trên lịch sử di chuyển để phản ứng tốt hơn với từng loại Pacman agent.

---

## 4. Thuật Toán Sử Dụng

| Thuật toán | Mục đích | Độ phức tạp |
|---|---|---|
| BFS | Bản đồ khoảng cách, tìm đường ngắn nhất | O(V + E) |
| Flood Fill | Ước lượng di động tính, đếm ô khả dụng | O(V + E) |
| Minimax | Tìm kiếm đối kháng, mô phỏng cả hai người chơi | O(b^d) |
| Alpha-Beta | Cắt tỉa nhánh, tăng tốc minimax | O(b^(d/2)) tối ưu |
| Greedy | Sắp xếp nước đi, đánh giá nhanh | O(n log n) |
| Multi-source BFS | Khoảng cách đến junction gần nhất | O(V + E) |
| Voronoi | Phân chia lãnh thổ an toàn | O(V) |

**Ghi chú:** V ≤ 441 (21×21 grid), E ≤ 4V. Tất cả thuật toán đều chạy nhanh trên grid này.
