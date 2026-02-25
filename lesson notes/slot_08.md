# **Slot 08: NLP Basics: Word Encodings & Text to Sequence**

**Ngày học:** 29-01-2026

**Tài liệu gốc:** 
* 3.1 The word based encodings in TensorFlow.pptx
* 3.2 Text to sequence in TensorFlow.pptx

**Mục tiêu bài học:**
1.  Hiểu cách máy tính tiếp cận và hiểu được văn bản.
2.  Nắm vững khái niệm Word-based encodings (Mã hóa dựa trên từ).
3.  Thực hành chuyển đổi văn bản thành các chuỗi số (Text to Sequence).

---

## I. Lý thuyết cốt lõi (Core Concepts)

### 1. Vấn đề mã hóa văn bản (Text Encoding)
* Mạng nơ-ron chỉ có thể xử lý các con số (giống như ảnh đã là các ma trận số pixel). Vì vậy, ta cần chuyển đổi chữ viết thành số.
* **Tại sao không dùng bảng mã ASCII (Mã hóa ký tự)?** * Việc gán số cho từng chữ cái (A=1, B=2...) không giúp máy tính hiểu được "nghĩa" của từ.
    * *Ví dụ:* Từ "LISTEN" và "SILENT" chứa tập hợp các chữ cái giống hệt nhau, nhưng ý nghĩa lại hoàn toàn khác nhau. Mã hóa theo ký tự sẽ làm máy tính bối rối.

### 2. Giải pháp: Word-based Encodings
* Gán cho **mỗi từ** một giá trị số nguyên duy nhất. 
* *Ví dụ:* `I = 1`, `love = 2`, `my = 3`, `dog = 4`. Câu "I love my dog" sẽ được hiểu thành chuỗi `[1, 2, 3, 4]`.

### 3. Vấn đề độ dài câu (Sentence Length)
* Trong xử lý ảnh, ta phải thay đổi kích thước (resize) ảnh về cùng một size trước khi đưa vào mô hình. 
* Tương tự với văn bản, các câu luôn có độ dài ngắn khác nhau. Ta sử dụng kỹ thuật **Text to Sequence** để biến chúng thành các tập hợp chuỗi số nguyên có thể xử lý được.


---

## II. Triển khai Code (Implementation)

Dưới đây là đoạn code cốt lõi thể hiện quy trình hoạt động của `Tokenizer` được trình bày trong các slide:

```python
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. Chuẩn bị tập dữ liệu văn bản mẫu
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!'
]

# 2. Khởi tạo Tokenizer
# num_words: Giới hạn số lượng từ phổ biến nhất muốn giữ lại
tokenizer = Tokenizer(num_words=100)

# 3. Học từ vựng (Fit on texts)
# Hàm này sẽ quét qua văn bản và tạo từ điển (Word Index)
tokenizer.fit_on_texts(sentences)

# Lấy ra từ điển vừa tạo
word_index = tokenizer.word_index
print("Word Index:", word_index)
# Kết quả mong đợi: {'love': 1, 'my': 2, 'i': 3, 'dog': 4, 'cat': 5, 'you': 6}
# Lưu ý: Ký tự đặc biệt như dấu chấm than (!) sẽ tự động bị loại bỏ, chữ hoa thành chữ thường.

# 4. Chuyển đổi câu thành chuỗi số (Text to Sequence)
# Thêm các câu mới có độ dài khác nhau để test
test_sentences = [
    'I really love my dog',
    'my dog loves my manatee'
]

# Biến văn bản thành mảng số
sequences = tokenizer.texts_to_sequences(test_sentences)
print("Sequences:", sequences)

```

---

## III. Ghi chú quan trọng (Key Takeaways) & Hiện tượng OOV

Slide 3.2 có nhắc đến một hiện tượng rất quan trọng khi gọi hàm `texts_to_sequences`: **Điều gì xảy ra với những từ mà mô hình chưa từng gặp (Out-of-Vocabulary - OOV)?**

Dựa trên code ở trên:

* Câu `'I really love my dog'` chứa từ `'really'` không có trong từ điển gốc. Keras sẽ **bỏ qua từ này**. Chuỗi kết quả sẽ là `[3, 1, 2, 4]` (tương đương "I love my dog"), từ 'really' bị mất đi.
* Câu `'my dog loves my manatee'` chứa `'loves'` và `'manatee'` là từ mới. Kết quả chỉ còn `[2, 4, 2]` (tương đương "my dog my").

*(Để khắc phục việc bị mất từ này, TensorFlow cung cấp tính năng `oov_token` mà chúng ta sẽ tìm hiểu ở các bài sau).*

