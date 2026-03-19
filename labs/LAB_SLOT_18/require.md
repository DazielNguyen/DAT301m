## Yêu cầu cần đạt
### Yêu cầu 1: Tiền xử lý dữ liệu tiếng Việt
- Chuẩn hóa văn bản về chữ thường.
- Giữ dấu tiếng Việt, chuẩn hóa khoảng trắng.
- Tokenize theo từ và lọc nhiễu cơ bản.

### Yêu cầu 2: Xây dựng mô hình liên kết từ
- Xây dựng Bigram Graph từ dữ liệu đã làm sạch.
- Mở rộng thêm Trigram để tăng độ mượt ngữ cảnh.

### Yêu cầu 3: Thuật toán sinh câu
- Sinh câu từ 1 từ hoặc 1 câu đầu vào.
- Có cơ chế chống lặp vô hạn (visited set).
- Có cơ chế fallback khi gặp dead-end.
- Có điều khiển coherence theo chủ đề để giảm nhảy ngữ cảnh.

### Yêu cầu 4: Đánh giá đầu ra
- Chạy tối thiểu 10 lần cho mỗi chế độ.
- Thống kê số từ và tỷ lệ đạt ngưỡng 38-42 từ.

### Yêu cầu 5: Báo cáo nộp bài
- Trình bày chức năng theo từng yêu cầu.
- Có minh họa output cho cả 2 chế độ đầu vào.
- Giải thích nguyên lý hoạt động và cơ chế tối ưu chất lượng.

