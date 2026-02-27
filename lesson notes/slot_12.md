# **Slot 12 - NLP: Sequence Models và Text Generation**
**Ngày học:** 26-02-2026

**Tài liệu gốc:** 

- 3.9 Exploring different sequence model.ppt
- 3.10 Sequence models and literature for NLP.pptx 

## MỤC TIÊU BÀI HỌC:

1. Hiểu cơ chế dự đoán từ tiếp theo (Predict the next word) của máy tính.
2. Nắm vững quy trình chuẩn bị dữ liệu cho bài toán sinh văn bản.
3. Xây dựng và triển khai mạng nơ-ron (LSTM) để tự động sáng tác thơ hoặc văn bản.

### PHẦN 1: Ý TƯỞNG CỐT LÕI
- Làm sao để máy tính tự viết văn? Giải pháp là cung cấp cho máy tính một khối lượng lớn văn bản, trích xuất toàn bộ từ vựng và biến nó thành bài toán: 
- Nhìn các từ phía trước (đầu vào X) để đoán từ tiếp theo (đầu ra Y).
- Ví dụ: Cho câu "I, FPT, Love, HoaDNT".
    - Ta tạo dữ liệu huấn luyện: X = "I, FPT, Love" và Y = "HoaDNT".
    - Sau khi học, mỗi khi mạng nơ-ron thấy cụm từ "I, FPT, Love", nó sẽ tự động dự đoán từ tiếp theo là "HoaDNT".

### PHẦN 2: QUY TRÌNH CHUẨN BỊ DỮ LIỆU
- Để làm bài thực hành này, tập dữ liệu mẫu thường là các bài thơ hoặc bài hát truyền thống (ví dụ: bài hát truyền thống của Ireland).

Bước 1: Khởi tạo từ điển và Tạo chuỗi N-grams

- Tách toàn bộ bài hát thành các câu dựa trên dấu xuống dòng.
- Khởi tạo Tokenizer để lập từ điển từ vựng (tạo cặp từ và mã token).
- Biến đổi các câu thành danh sách các chuỗi số.
T- ạo N-grams: Lặp qua danh sách token để tạo ra các cụm từ tăng dần.

Bước 2: Điền khuyết (Padding)
- Tìm câu có chiều dài lớn nhất trong tập dữ liệu.
- Điền thêm các số 0 vào để tất cả các chuỗi có chung một kích thước ma trận.

Bước 3: Tách Input (X) và Label (Y)

- Từ ma trận đã điền khuyết, lấy phần tử cuối cùng của mỗi hàng làm nhãn Y, tất cả các phần tử phía trước làm Input X.
- Tạo mã hóa One-hot (One-hot encoding) cho nhãn Y.

### PHẦN 3: XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH
Kiến trúc mạng tuần tự tiêu chuẩn gồm 3 phần chính:

1. Embedding Layer: Biến đổi các giá trị số nguyên thành vector.
2. LSTM Layer: Mạng nơ-ron hồi quy giúp ghi nhớ bối cảnh từ các chuỗi từ đi trước.
3. Dense Layer: Lớp Output cuối cùng.

Mô hình sẽ được Compile và Huấn luyện (Train) dựa trên tập dữ liệu đã chuẩn bị.

### PHẦN 4: DỰ ĐOÁN VĂN BẢN

Sau khi model đã học xong, quy trình để dự đoán đoạn văn bản tiếp theo như sau:
- Nhập một cụm từ mồi (Seed Text) để gợi ý cho máy tính.
- Đưa cụm từ này vào Tokenizer và thực hiện Padding tương tự như lúc huấn luyện.
- Truyền vào mô hình để dự đoán và lấy ra từ tiếp theo.
- Nối từ mới này vào chuỗi mồi ban đầu, và lặp lại các bước trên để tiếp tục sinh ra các từ tiếp theo.
