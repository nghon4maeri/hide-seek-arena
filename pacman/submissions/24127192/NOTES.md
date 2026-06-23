# Ghi chú Cải tiến Ghost Agent v3.5

### 1. Ý tưởng và Các Thuật toán/Chiến lược Đã Sử Dụng
Ở phiên bản v3.5, Ghost tiếp tục phát triển từ nền tảng v3 nhưng không chỉ dựa vào một mô hình A* cố định nữa. Mục tiêu chính là sống lâu nhất có thể khi Pacman có tốc độ 2, biết vị trí Ghost, và thường sử dụng A* kết hợp dự đoán vận tốc. Ghost v3.5 kết hợp nhiều lớp dự đoán, đánh giá an toàn theo thời gian, và tìm kiếm sâu có giới hạn thời gian:
- **US-L* Online Learning nâng cấp:** Thay vì chỉ nhớ một hành động cho mỗi trạng thái, Ghost lưu thống kê hành động theo trạng thái trừu tượng, tính độ tin cậy, và tạo nhiều dự đoán Pacman có trọng số.
- **A* Search với Manhattan heuristic:** Ghost dùng A* để mô phỏng đường đuổi của Pacman đến vị trí Ghost hiện tại và vị trí Ghost bị nội suy theo vận tốc.
- **Greedy Best-First Search:** Được dùng như một mô hình Pacman phụ, mô phỏng trường hợp Pacman chọn hướng làm giảm Manhattan/BFS distance nhanh nhất.
- **BFS/UCS Shortest Path Model:** Vì bản đồ có chi phí bước đi bằng nhau, BFS được dùng để tính khoảng cách thật, chaser model, safe area, và các đặc trưng topology.
- **Iterative Deepening Maximin Search:** Ghost tìm kiếm nhiều độ sâu tăng dần trong giới hạn thời gian, luôn giữ lại nước đi tốt nhất ở độ sâu đã hoàn thành.
- **Temporal Safety Analysis:** Ghost không chỉ hỏi "Pacman cách bao xa", mà còn ước lượng Pacman cần bao nhiêu lượt để bắt được một ô khi có tốc độ 2.
- **Topology Control:** Ghost ưu tiên vùng core, loop, junction nhiều lối thoát; phạt nặng dead-end branch và vùng có ít không gian thoát.
- **Anti-Velocity Prediction:** Ghost thưởng cho các nước rẽ hướng ở junction hoặc lúc nguy hiểm để phá dự đoán tuyến tính của Pacman.
- **Phase Strategy theo khoảng cách ban đầu:** Nếu hai agent xuất phát gần, Ghost dùng giai đoạn đầu để kéo giãn khoảng cách; nếu xuất phát xa, Ghost ưu tiên đứng/di chuyển quanh junction để đọc thuật toán Pacman trước khi dẫn dụ vào loop.
- **Loop Luring & Loop Commitment:** Ghost tính khoảng cách tới loop/core và chủ động kéo Pacman về vùng có vòng lặp, sau đó ưu tiên duy trì trong loop thay vì chạy vào corridor bị khóa.

### 2. Chi tiết Triển khai và Sự Kết Hợp Các Thuật Toán
- **Mô hình Pacman đa giả thuyết:** Hàm `_pacman_model_positions` tạo nhiều vị trí Pacman có thể đi tới bằng A* tốc độ 2, BFS chaser tốc độ 2, Greedy tốc độ 2, và vị trí hiện tại. Các giả thuyết này được đưa vào US-L* để xếp hạng theo độ tin cậy.
- **US-L* confidence-weighted prediction:** Lớp `USLStar` dùng state gồm hướng tương đối Ghost-Pacman, bucket khoảng cách, hướng Ghost vừa đi, cấu trúc ô Pacman đang đứng, và tường xung quanh. Khi quan sát Pacman thật sự di chuyển, Ghost cập nhật thống kê để dự đoán tốt hơn ở các lượt sau.
- **A* + tốc độ 2:** `_astar_path` tìm đường ngắn nhất bằng Manhattan heuristic, còn `_follow_path_with_speed` mô phỏng Pacman đi tiếp tối đa 2 ô nếu vẫn đi thẳng trên cùng hướng.
- **Temporal capture ETA:** `_capture_eta` ước lượng số lượt Pacman cần để vào vùng bắt Ghost với `CAPTURE_DISTANCE = 2`. Điểm an toàn bị phạt rất nặng nếu ETA nhỏ, thay vì chỉ dựa vào BFS distance.
- **Safe area / Voronoi cục bộ:** `_safe_area` flood-fill các ô Ghost có thể tới, chỉ tính điểm cao cho các ô mà Ghost tới trước hoặc còn đủ margin so với Pacman.
- **Influence Map:** `_pacman_influence` lưu lịch sử vị trí Pacman gần đây và phạt các ô nằm trong vùng Pacman vừa quét qua, tránh việc Ghost chạy ngược vào đường bị kiểm soát.
- **Greedy ordering + Iterative Deepening:** `_move_order` sắp xếp nước đi bằng heuristic trước, giúp `_search` duyệt các nhánh hứa hẹn trước. Sau đó iterative deepening từ độ sâu thấp lên cao, nếu gần timeout thì trả về nước tốt nhất đã tìm được.
- **Maximin nhiều nhánh Pacman:** Ở mỗi node, Ghost giả định Pacman có thể đi theo nhánh nguy hiểm nhất trong số các dự đoán hợp lý, nhưng vẫn trừ điểm theo trọng số confidence của US-L*.
- **Phân loại gần/xa ở lượt đầu:** Agent lưu `_initial_distance` và `_initial_far`. Nếu ban đầu gần, 20 bước đầu dùng `_survival_oracle_move` và `_opening_spread_move` để sống sót/kéo giãn; nếu ban đầu xa, `_far_reading_move` ưu tiên junction hoặc `STAY` an toàn để học hành vi Pacman.
- **Tính đường về loop:** `_compute_loop_dist` tạo bản đồ khoảng cách tới `loop_set` hoặc core. `_loop_lure_move` dùng khoảng cách này để chọn nước đi kéo Ghost dần vào vùng loop, sau đó `committed=True` để ưu tiên ở lại loop.
- **STAY có điều kiện:** `_allow_stay` chỉ cho phép đứng yên sau 20 bước, khi khoảng cách đủ xa, hoặc khi ban đầu đã xa. Điều này giúp Ghost đọc Pacman mà không đứng yên quá sớm lúc nguy hiểm.

### 3. Kết Quả Đạt Được và Khả Năng Áp Dụng
- **Ghost thông minh hơn v3 trong mô hình đối thủ:** V3.5 không còn phụ thuộc tuyệt đối vào một fallback A* duy nhất. Khi Pacman đổi cách di chuyển, US-L* có thể dần tăng trọng số cho hành vi đã quan sát.
- **Tận dụng bản đồ triệt để hơn:** Ghost biết toàn bộ topology bản đồ, nhận diện core, loop, junction, dead-end depth, safe area và các vùng Pacman vừa kiểm soát.
- **Ra quyết định có tính chiến thuật hơn:** Ghost cân bằng giữa chạy xa Pacman, giữ nhiều lối thoát, né vùng ảnh hưởng, rẽ hướng phá nội suy vận tốc, và tránh tự nhốt mình vào hành lang/ngõ cụt.
- **Tương thích với luật `capture_distance = 2`:** Trong agent đã đặt rõ `CAPTURE_DISTANCE = 2`, nên mô hình ETA và điểm nguy hiểm đang đánh giá theo luật bắt cạnh hiện tại.
- **Cải thiện hành vi ở start ngẫu nhiên:** Với một số vị trí stochastic, Ghost có thể kéo về loop và sống đủ giới hạn bước. Tuy nhiên kết quả phụ thuộc mạnh vào vị trí ban đầu do Pacman tốc độ 2 và capture distance 2.

### 4. Hạn Chế và Hướng Tối Ưu Tương Lai
- **Pacman tốc độ 2 vẫn có lợi thế rất lớn:** Với luật bắt khi Manhattan distance `< 2`, Pacman có thể khóa một số thế cờ rất sớm nếu Ghost xuất phát gần và chỉ được đi 1 ô mỗi lượt.
- **US-L* cần dữ liệu quan sát:** Ở vài bước đầu, US-L* chưa có đủ thống kê nên vẫn phải dựa nhiều vào A*, BFS và Greedy fallback.
- **Chi phí tính toán cao hơn v3:** V3.5 dùng nhiều cache cho A*, capture ETA, BFS distance và safe area. Điều này giúp chạy trong timeout trên bản đồ hiện tại, nhưng bản đồ lớn hơn có thể cần giảm độ sâu hoặc giới hạn cache.
- **Chiến lược loop phụ thuộc vị trí:** Nếu Ghost xuất phát gần Pacman ở thế bị khóa cưỡng bức, việc dẫn dụ vào loop không kịp phát huy trước khi Pacman áp sát.
- **Tối ưu trong tương lai:**
  - Thêm transposition table cho `_search` để tái sử dụng trạng thái giữa các nhánh.
  - Tối ưu mô hình Pacman tốc độ 2 theo đúng từng implementation cụ thể của đối thủ nếu được phép đọc trực tiếp agent đối phương.
  - Thêm opening book có điều kiện cho các thế xuất phát cố định, nhưng vẫn giữ fallback tổng quát cho map khác.
  - Mở rộng safe-area thành phân tích choke point / articulation point để né các vùng bị cắt đường trước khi Pacman áp sát.
  - Tách riêng policy deterministic-start và stochastic-start, vì hai chế độ có cấu trúc rủi ro khác nhau.

### 5. Cách Hoạt Động và Các Case Ra Quyết Định
Mỗi lượt, Ghost đi qua các bước sau trong hàm `step`:

- **Case 1: Khởi tạo bản đồ lần đầu.** Ghost chuyển map sang dạng tĩnh, tính danh sách ô đi được, bậc của từng ô, trung tâm bản đồ, vùng core, loop lớn nhất, junction, corridor, dead-end branch và độ sâu ngõ cụt.
- **Case 2: Không có nước đi hợp lệ.** Nếu Ghost không có ô hợp lệ để đi, trả về `Move.STAY`.
- **Case 3: Không nhìn thấy Pacman.** Nếu `enemy_position is None`, Ghost không thể dự đoán trực tiếp nên chọn nước đi tới ô có độ cơ động cao nhất, ưu tiên vùng nhiều lối thoát.
- **Case 4: Có quan sát Pacman mới.** Ghost so sánh vị trí Pacman hiện tại với vị trí Pacman lượt trước để lấy hành động thật của Pacman, cập nhật US-L*, đồng thời điều chỉnh tốc độ Pacman quan sát được nếu Pacman đi nhiều hơn dự kiến.
- **Case 5: Sinh dự đoán Pacman.** Ghost tạo nhiều giả thuyết: Pacman A* tới Ghost hiện tại, A* tới vị trí Ghost bị nội suy vận tốc, BFS chaser, Greedy best-first, và trạng thái đứng yên. US-L* xếp hạng các giả thuyết này thành danh sách `(vị trí, trọng số, nguồn dự đoán)`.
- **Case 6: Xác định chiến lược theo khoảng cách ban đầu.** Lần đầu thấy Pacman, Ghost lưu `_initial_distance`. Nếu khoảng cách ban đầu nhỏ hơn nửa chiều dài map thì xem là start gần; ngược lại xem là start xa.
- **Case 7: Start gần, 20 bước đầu kéo giãn.** Ghost dùng survival oracle và opening spread để tăng khoảng cách, tránh quay lại ô vừa đi, ưu tiên core/junction/safe area để có đủ thời gian học US-L*.
- **Case 8: Start xa, 20 bước đầu đọc Pacman.** Ghost đi tới junction/ngã rẽ hoặc có thể `STAY` nếu ETA an toàn và khoảng cách đủ xa, nhằm quan sát Pacman di chuyển để cập nhật US-L*.
- **Case 9: 10 bước tiếp theo dẫn dụ vào loop.** Sau 20 bước đầu, `_loop_lure_move` ưu tiên giảm khoảng cách tới loop/core, đồng thời vẫn giữ ETA an toàn và làm Manhattan distance biến đổi để Pacman phải replanning.
- **Case 10: Sau giai đoạn dẫn dụ, commit vào loop/counter-loop.** Nếu Ghost đã gần loop hoặc ban đầu xa, agent ưu tiên ở lại vùng loop và dùng counter-loop để kéo Pacman theo vòng thay vì chạy thẳng vào corridor.
- **Case 11: Nguy hiểm sát mặt.** Nếu một dự đoán Pacman đang cách Ghost quá gần, Ghost dùng `_panic_move`: chọn nước vừa tăng an toàn tức thì, vừa ưu tiên core/loop/junction và tránh dead-end.
- **Case 12: Trạng thái bình thường.** Ghost tạo danh sách nước đi hợp lệ, chấm điểm từng nước bằng `_cell_safety_score`, `_anti_velocity_score`, rồi dùng Greedy best-first ordering để đưa nước tốt vào search trước.
- **Case 13: Tìm kiếm sâu.** `_search` chạy iterative deepening maximin. Với mỗi nước Ghost thử đi, Pacman được mô phỏng bằng nhiều nhánh dự đoán; Ghost lấy nhánh xấu nhất hợp lý để tránh lạc quan quá mức.
- **Case 14: Gần hết thời gian.** Nếu thời gian gần chạm `TIME_BUDGET`, search dừng và trả về nước tốt nhất đã tìm được ở độ sâu hoàn chỉnh trước đó, tránh timeout.
- **Case 15: Cập nhật lịch sử.** Sau khi chọn nước, Ghost lưu vị trí hiện tại vào lịch sử để lượt sau dùng cho anti-velocity, tránh lặp lại, và cập nhật mô hình US-L*.

Tóm lại, Ghost v3.5 hoạt động theo phong cách "dự đoán nhiều kịch bản rồi chọn nước sống sót tốt nhất": trước hết học Pacman bằng US-L*, sau đó mô phỏng Pacman bằng A*/BFS/Greedy, đánh giá từng ô bằng ETA + safe area + topology + influence map, chọn phase gần/xa theo khoảng cách ban đầu, dẫn dụ về loop khi có điều kiện, cuối cùng dùng iterative deepening để chọn nước ít rủi ro nhất trong các nhánh Pacman có thể xảy ra.
