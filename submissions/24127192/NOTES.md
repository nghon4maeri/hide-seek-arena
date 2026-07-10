# Ghi chú Cải tiến Ghost Agent v3

### 1. Ý tưởng và Các Thuật toán/Chiến lược Đã Sử Dụng
Ở phiên bản v3 hiện tại, mục tiêu lớn nhất là khắc phục điểm yếu của Ghost khi đối đầu với Pacman có tốc độ 2 và sử dụng thuật toán A* kết hợp nội suy vận tốc (như bot của Tony). Thay vì sử dụng cách tiếp cận tham lam (Greedy) dựa trên khoảng cách ngắn hạn vốn dễ bị bắt bài, Ghost v3 được nâng cấp để mô phỏng chính xác đối thủ thông qua các thuật toán:
- **Predictive Deep Search (Tìm kiếm sâu kết hợp dự đoán):** Xây dựng cây trò chơi (Game Tree) sử dụng Iterative Deepening DFS để nhìn trước tương lai.
- **Exact Opponent Modeling (Mô hình hóa đối thủ chính xác):** Hardcode mô phỏng chính xác thuật toán A* và logic ngoại suy tốc độ 2 của Tony vào hàm `_predict_tony`.
- **US-L* Online DFA Learning:** Được giữ lại làm lớp bọc (fallback wrapper) nhằm bắt các hành vi khác lạ nếu đối thủ không di chuyển y hệt Tony.
- **Topology & Loop Detection (Phân tích cấu trúc bản đồ):** Chủ động tìm kiếm các ngã rẽ cụt (Dead-end branches) và các vòng lặp (Loops/2-Core) để định hướng an toàn cho cây tìm kiếm.

### 2. Chi tiết Triển khai và Sự Kết Hợp Các Thuật Toán
- **Mô phỏng đối thủ chính xác (Opponent Modeling):** Ghost sử dụng thuật toán A* y hệt Tony, tái hiện chính xác cách Tony tính toán khoảng cách và nội suy hướng di chuyển tiếp theo (`_tony_astar`). Việc này cho phép Ghost biết CHÍNH XÁC tọa độ tiếp theo mà Pacman sẽ nhảy tới ở mỗi bước.
- **Tối ưu hóa Cây Tìm Kiếm (DFS + Caching):** Để tránh việc hàm chạy quá thời gian (Timeout 1.0s), thuật toán A* của Tony được gắn `@lru_cache` để tái sử dụng lộ trình. Hàm DFS (`search_survival`) sử dụng kỹ thuật cắt tỉa cành bằng việc tính trước tọa độ dự đoán của Pacman (do hành động của Pacman ở bước t được tính độc lập với bước t của Ghost), làm giảm hệ số rẽ nhánh (branching factor).
- **Đánh giá Trạng thái (Heuristic Evaluation):** Ở các Node lá (Leaf nodes) của cây, Ghost tự đánh giá độ an toàn dựa trên khoảng cách đường đi thực tế (`true_dist`). Đặc biệt, nếu Node rơi vào nhánh ngõ cụt (`deb`) thì sẽ bị phạt nặng (`-5000`), nhưng nếu Node nằm trong Core an toàn hoặc Vòng lặp thả diều (`loop_set`) thì được cộng điểm thưởng.
- **Xử lý tình huống (Case handling):** Nếu thời gian tìm kiếm vượt ngưỡng an toàn (`0.85s`), Iterative Deepening sẽ dừng lại và trả về kết quả tốt nhất của độ sâu an toàn trước đó, đảm bảo Ghost không bao giờ bị xử thua do Time Limit Exceeded.

### 3. Kết Quả Đạt Được và Khả Năng Áp Dụng
- **Khắc chế hoàn toàn Tony (24127561):** Nếu như trước đây Ghost chỉ sống sót được khoảng 8 bước trước sức ép quá lớn từ tốc độ x2 của Pacman Tony, thì với bản cập nhật v3, Ghost **sống sót hoàn hảo suốt 200 bước (giới hạn tối đa của game)** và giành chiến thắng. Bằng việc nhìn thấu tương lai (look-ahead) lên tới 15-20 bước, Ghost luôn chủ động rẽ vào các khoảng không rộng hoặc lượn vòng (loop) để từ chối việc bị dồn vào góc.
- **Áp dụng cho các trường hợp khác:** Nhờ lớp bọc **US-L***, nếu Ghost đối đầu với một Pacman khác có cách di chuyển phi tiêu chuẩn (không theo A*), US-L* sẽ ghi nhận các "Counterexamples" thông qua Observation Table và tự động điều chỉnh dự đoán (DFA). Mặc dù hàm fallback vẫn là A* của Tony, nhưng quá trình học online sẽ dần kéo dự đoán về sát với thực tế của bất kỳ đối thủ nào.

### 4. Hạn Chế và Hướng Tối Ưu Tương Lai
- **Hạn chế về Bộ Nhớ (Memory Overhead):** Hàm `_tony_astar` hiện đang sử dụng `lru_cache` cực lớn để lưu cache tọa độ. Ở các bản đồ quá lớn, việc này có thể chiếm dụng RAM đáng kể và làm chậm tốc độ Garbage Collection.
- **Phụ thuộc quá lớn vào Fallback:** Nếu Ghost gặp một đối thủ dùng Minimax hoặc RL phức tạp, lớp US-L* có thể không học đủ nhanh trong vài bước đầu tiên, dẫn tới việc Ghost chạy nhầm vào bẫy vì quá tin tưởng vào hàm mô phỏng A*.
- **Tối ưu trong tương lai:** 
  - Triển khai thuật toán Alpha-Beta Pruning thực sự kết hợp Transposition Table (Zobrist Hashing) thay vì chỉ dùng DFS và Caching đơn thuần.
  - Tích hợp thêm Influence Map tiên tiến hơn để phát hiện trước hướng bị bao vây (Flanking) nếu Arena có tính năng 2 Pacman vây bắt 1 Ghost.
  - Chuyển logic từ DFS đơn luồng sang MCTS (Monte Carlo Tree Search) nếu muốn mở rộng sang một môi trường liên tục (continuous) hoặc có thêm yếu tố ngẫu nhiên (stochastic).
