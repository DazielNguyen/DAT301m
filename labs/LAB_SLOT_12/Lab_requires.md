Sinh viên được cung cấp một tập dữ liệu nhỏ (mini-corpus). 

Yêu cầu của bài toán là viết một chương trình để mô hình hóa tập dữ liệu này, từ đó tự động sinh ra các cụm từ/câu mới dựa trên quy luật nối từ có trong dữ liệu gốc. 

Cuối cùng, chương trình phải tìm ra và xếp hạng 10 cụm từ dài nhất có thể tạo ra được. https://www.kaggle.com/datasets/allenai/aristo-mini-corpus

# Yêu cầu 1: Tiền xử lý dữ liệu (Pre-processing):

- Chuyển toàn bộ các câu trong dataset về chữ thường (lowercase).
- Tách các câu thành các từ đơn (word tokens).

# Yêu cầu 2: Xây dựng đồ thị liên kết từ (Word Graph/Bigram Model)

- Duyệt qua tập dữ liệu để tìm ra quy luật từ tiếp theo.

# Yêu cầu 3: Thuật toán sinh câu (Text Generation)

- Viết thuật toán để bắt đầu từ một từ bất kỳ và liên tục nối thêm các từ tiếp theo dựa trên đồ thị đã xây dựng ở Yêu cầu 2.

- Điều kiện bắt buộc: Phải có cơ chế chống lặp vô hạn (Infinite Loop). Chương trình không được phép đi qua một từ đã xuất hiện trong chuỗi hiện tại đang xét. Nhánh sinh câu sẽ dừng lại khi không tìm được từ tiếp theo thỏa mãn điều kiện.

# Yêu cầu 4: Xếp hạng và xuất kết quả (Ranking & Output)

- Thu thập tất cả các câu/cụm từ sinh ra được từ thuật toán.
- Loại bỏ các câu trùng lặp (nếu có).
- Đếm số lượng từ trong mỗi câu và sắp xếp giảm dần theo độ dài.
- In ra màn hình Top 10 cụm từ dài nhất kèm theo số lượng từ của mỗi cụm.

**NỘP BÀI DƯỚI DẠNG FILE DOC BÁO CÁO, CHỤP HÌNH FUNCTION CỦA CÁC YÊU CẦU VÀ OUTPUT, GHI CHÚ THÍCH, GIẢI THÍCH NGUYÊN LÝ HOẠT ĐỘNG**