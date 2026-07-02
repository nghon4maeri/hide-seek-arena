# Ghost Agent 24127192 - Blind Search Notes

## 1. Y tưởng thực hiện

Ghost agent được xây như một bộ não deterministic cho vai trò hider trong Blind Search. Ý tưởng chính là không chỉ chạy xa Pacman theo Manhattan distance, mà duy trì một bản đồ tinh thần, một tập giả thuyết vị trí Pacman, rồi chọn nước đi có khả năng sống sót tốt nhất trong nhiều kịch bản truy đuổi.

Các thuật toán/concept đã dùng:

- Mental Map: tích lũy `map_state` qua từng step, dùng tường toàn cục để suy luận topology.
- Pursuit-Evasion Search: mô hình Pacman là pursuer speed-2, Ghost là evader speed-1.
- Paranoid Search: minimax/alpha-beta khi Pacman visible và gần, giả định Pacman chọn nhánh bất lợi nhất cho Ghost.
- Information Set MCTS / POMCP-lite: đánh giá mỗi move của Ghost qua nhiều hypothesis Pacman và nhiều rollout scenario deterministic.
- Deterministic Monte Carlo: giữ concept rollout nhiều kịch bản nhưng bỏ sampling ngẫu nhiên, dùng scenario có thứ tự ổn định.
- USL* Start: opening book nhỏ cho deterministic start, đưa Ghost vào nhánh phải có nhiều lối thoát trước khi bàn giao cho search.
- USL* Online Learner: học xu hướng di chuyển Pacman theo abstract state khi Pacman xuất hiện.
- Topology Analysis: nhận diện junction, loop/core, corridor, dead-end để tránh bị dồn vào hành lang cụt.

## 2. Cài đặt

Đã triển khai trực tiếp trong `blind/submissions/24127192/agent.py`:

- Loại bỏ toàn bộ `random` khỏi Ghost brain để phù hợp deterministic mode.
- Thay particle filter ngẫu nhiên bằng weighted information set top-k cho vị trí Pacman.
- Thêm prior deterministic cho Pacman start mặc định `(15, 10)` khi Ghost chưa thấy Pacman ở opening.
- Khi Pacman hidden, belief được propagate bằng tập move speed-1/speed-2, xu hướng greedy chase, USL* prediction và lọc các vị trí lẽ ra đã nằm trong tầm nhìn Ghost.
- Sửa topology để phân tích toàn bộ ô không-phải-tường, vì framework luôn cho thấy tường toàn cục.
- Thêm safety score chung gồm BFS distance, Pacman speed-2 reach risk, cross-line-of-sight risk, junction/loop/core bonus, dead-end/edge-corridor penalty và anti-oscillation.
- Thêm deterministic rollout engine cho Information Set MCTS/POMCP-lite.
- Thêm sanity check giữa Paranoid Search, MC rollout và heuristic score để tránh chọn nước lookahead cục bộ nhưng dễ bị kẹp.
- Thêm USL Start opening route trong 5 bước đầu ở deterministic start.
- Giữ `PacmanAgent` placeholder hợp lệ và chỉ return action đúng format; Ghost luôn return `Move`.

## 3. Kết quả đạt được

Kiểm tra đã chạy:

- Compile-in-memory: pass.
- Trận đơn deterministic với Pacman `24127561`, obs radius 5, Pacman speed 2, capture distance 2, max 120: Ghost sống tới step 110, vượt mục tiêu sống qua 100 step đầu.

Chưa benchmark được trực tiếp với Pacman `blind/submissions/24127457/agent.py` trong sandbox hiện tại vì runtime Python không có `torch`; khi load Pacman `24127457` gặp lỗi `ModuleNotFoundError: No module named 'torch'`.

Lệnh chạy visual deterministic với Pacman `24127457` khi môi trường có torch:

```bash
cd blind/src
python arena.py --seek 24127457 --hide 24127192 \
  --pacman-obs-radius 5 --ghost-obs-radius 5 \
  --start-mode deterministic --max-steps 200 --delay 0.15
```

## 4. Những khuyết điểm và hướng cải tiến

- Chưa benchmark được với Pacman `24127457` trong runtime hiện tại do thiếu torch, nên chưa có số liệu đối đầu chính thức với model PPO/LSTM đó.
- Opening book đang tối ưu cho deterministic map mặc định; nếu map/start thay đổi, USL Start có thể kém hiệu quả hơn và cần fallback opening theo topology tự động.
- POMCP-lite hiện là deterministic scenario rollout nên ổn định, nhưng chưa phong phú bằng một IS-MCTS đầy đủ với tree reuse và backprop statistics.
- Belief update vẫn là heuristic, chưa có Bayesian observation likelihood đầy đủ cho mọi trường hợp Pacman speed-2.
- Khi Ghost bị ép vào corridor sát biên, agent vẫn có thể sống qua 100 step nhưng về late-game có nguy cơ bị Pacman quét ngang; hướng cải tiến là thêm trap detector theo biconnected components/articulation points.
- Nếu Pacman policy cố tình không greedy mà đi chặn đầu theo loop topology, Ghost cần thêm opponent model sâu hơn từ USL* và rollout nhánh intercept thay vì chỉ chase-response.
