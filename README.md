# Distinguishing Between Bots and Humans on Social Media (Reddit)

Đề tài Bài tập lớn môn **Nhập môn Học máy**: Nghiên cứu và xây dựng hệ thống phân loại tài khoản Bot và Người dùng thật (Human) trên mạng xã hội Reddit.

---

## 1. Mô tả bài toán (Problem Description)

### Bối cảnh & Tính cấp thiết
Trên các nền tảng mạng xã hội như Reddit, tài khoản tự động (Bot) thường được sử dụng để thao túng tương tác, spam nội dung hoặc phát tán thông tin sai lệch trong các cộng đồng tin tức và chính trị. Trong khi các bot truyền thống dễ lộ diện qua tần suất hoạt động bất thường, các bot thế hệ mới sử dụng mô hình ngôn ngữ lớn (Large Language Models - LLMs) có thể tạo ra văn bản tự nhiên như người thật, làm giảm hiệu quả của các phương pháp chỉ dựa trên văn bản (Text-only).

### Phát biểu bài toán (Problem Formulation)
* **Loại bài toán**: Phân loại nhị phân ở cấp độ tài khoản người dùng (User-level Binary Classification).
* **Đầu vào (Input)**: Mỗi tài khoản <i>u</i> được mô tả thông qua hai nguồn dữ liệu kết hợp:
  1. **Dữ liệu thống kê và hành vi (`Users.csv` + tổng hợp lịch sử)**: 18 đặc trưng định lượng phản ánh mật độ hoạt động, khoảng trễ thời gian giữa các bài đăng (Δt), nhịp sinh học 24 giờ (`circadian_entropy`), độ bùng phát (`burstiness`) và tương tác cộng đồng (`score`, `upvote_ratio`, `unique_subreddit_num`).
  2. **Chuỗi nội dung hoạt động theo thời gian (`user_post_comment.json`)**: Tập hợp các bài đăng (`posts`) và bình luận (`comment_1`, `comment_2`) của tài khoản <i>u</i> được sắp xếp theo thứ tự thời gian (`created_utc`).
* **Đầu ra (Output)**: Dự đoán nhãn phân loại <i>ŷ</i><sub>u</sub> ∈ {0, 1} cùng xác suất dự đoán <i>P</i><sub>final</sub>(<i>u</i>) ∈ [0, 1]:
  * `0`: **Bot** (tài khoản tự động / bot tác nhân).
  * `1`: **Human / User** (tài khoản người dùng thật).

### Hướng tiếp cận & Tiêu chí đánh giá
* **Kiến trúc mô hình**: Xây dựng hệ thống **Hybrid Multi-modal Ensemble** gồm hai nhánh chạy song song:
  * **Nhánh 1 (Tabular & Temporal)**: Mô hình **Linear SVM** kết hợp hiệu chuẩn xác suất **Platt Scaling** trên 18 đặc trưng hành vi đã xử lý Missing Values và biến đổi **Yeo-Johnson**.
  * **Nhánh 2 (Sequential Text & Behavior)**: Mô hình **RNN (Many-to-One)** tiếp nhận chuỗi biểu diễn từ **RoBERTa-base** (`Mean-Pooling`) kết hợp mã hóa loại hành động (`One-hot`), chiếu giảm chiều qua `Projection + LayerNorm + GELU + Masking`.
  * **Tầng hợp nhất**: Mô hình **5-Fold OOF Stacking (`LogisticRegression`)** tổng hợp xác suất đầu ra từ hai nhánh độc lập.
* **Tiêu chí đánh giá**: `Accuracy`, `Precision`, `Recall`, `F1-score` và `ROC-AUC`.

---

## 2. Tập dữ liệu sử dụng (Datasets)

1. **BotSim-24 (Benchmark Dataset)**: Gồm `2.907` tài khoản (`1.907` Human nhãn `1` và `1.000` Bot nhãn `0`), dùng làm tập huấn luyện và đối sánh chuẩn cho Pipeline `SVM + RNN + Stacking` (chia `70% Train / 20% Val / 10% Test`).
2. **Tập dữ liệu tự thu thập từ Arctic Shift (06/2026 – 08/2026)**: Gồm `94.272` bài đăng và `5.668.129` bình luận trong `15` subreddit tin tức và chính trị (`378` tài khoản đã gán nhãn: `142` Bot từ BotBouncer nhóm A và `236` User có verified author flair, chia `70% Train / 15% Val / 15% Test`; tách riêng tập held-out `160` BotBouncer nhóm B và `873` BotSim-24).

---

## 3. Cấu trúc thư mục hiện tại

```text
binary_classification/
├── Users.csv                           # Hồ sơ người dùng BotSim-24 (2.907 tài khoản)
├── user_post_comment.json              # Lịch sử bài đăng & bình luận chi tiết
├── preprocess_svm_features.py          # Script trích xuất & chuẩn hóa 18 đặc trưng SVM (không rò rỉ dữ liệu)
├── dataset_18_features_cleaned.xlsx    # Tập 18 đặc trưng + 11 cột cờ missing đã làm sạch
└── báo cáo bài tập lớn.docx            # Bản thảo báo cáo kỹ thuật của nhóm
```
