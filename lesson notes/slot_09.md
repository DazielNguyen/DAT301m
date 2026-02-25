# **Slot 09 - NLP: Advanced Tokenizer, Padding & Word Embeddings**

**Ngày học:** 02-02-2026
**Tài liệu gốc:** 
* 3.3 The Tokenizer in TensorFlow.pptx
* 3.4 The padding in TensorFlow.pptx
* 3.5 The vector in TensorFlow.pptx

**Mục tiêu bài học:**
1. Khám phá các loại Tokenizer nâng cao trong gói `tensorflow_text`.
2. Sử dụng Padding để đồng bộ chiều dài các chuỗi câu.
3. Hiểu khái niệm biểu diễn từ dưới dạng Vector (Word Embeddings) và áp dụng với bộ dữ liệu IMDB.

---

## I. Tokenizer nâng cao (Advanced Tokenizers)

Ở bài trước ta đã dùng Tokenizer cơ bản của Keras. Tuy nhiên, TensorFlow cung cấp gói `tensorflow_text` với nhiều công cụ chia tách (tokenization) phức tạp hơn cho các bài toán đặc thù:

* **WhitespaceTokenizer:** Tách chuỗi dựa trên khoảng trắng (space, tab, new line) chuẩn ICU. Tốt cho việc tạo mẫu (prototype) nhanh.
* **UnicodeScriptTokenizer:** Tách chuỗi dựa trên ranh giới mã script Unicode.
* **WordpieceTokenizer & SentencepieceTokenizer:** Các kỹ thuật tách từ phổ biến tạo ra các "sub-tokens" (mảnh từ) được điều khiển bởi dữ liệu.
* **RegexSplitter:** Tách chuỗi tại các điểm neo (breakpoints) tùy ý được định nghĩa bằng Regular Expression.
* **Detokenization:** Quá trình ngược lại, ghép các token thành chuỗi ban đầu. Lưu ý quá trình này có thể bị hao hụt (lossy), chuỗi tạo ra có thể không khớp chính xác 100% với ban đầu.

---

## II. Padding (Điền khuyết chuỗi)

Mạng nơ-ron yêu cầu dữ liệu đầu vào phải có cùng kích thước (matrix đồng nhất). Vì các câu văn có độ dài khác nhau, sau khi chuyển thành sequence (chuỗi số), ta phải dùng kỹ thuật **Padding** để làm cho chúng bằng nhau.

### Nguyên lý hoạt động
* Hàm `pad_sequences` sẽ tìm câu dài nhất trong tập dữ liệu và biến tất cả các câu khác thành độ dài đó bằng cách chèn thêm các số `0`.
* **Các tham số quan trọng:**
    * `padding='post'`: Thêm số 0 vào **cuối** câu (mặc định là 'pre' - thêm vào đầu câu).
    * `maxlen`: Ép chiều dài tối đa cho mọi câu. Những câu ngắn hơn sẽ bị thêm 0, những câu dài hơn sẽ bị cắt bớt (truncate).
    * `truncating='post'`: Nếu câu dài hơn `maxlen`, cắt bỏ phần **cuối** câu (mặc định là 'pre' - cắt ở đầu câu).

---

## III. Word Embeddings (Vector hóa từ) & IMDB Dataset

### 1. Bộ dữ liệu IMDB (IMDB Dataset)
* Là bộ dữ liệu chứa 50.000 đánh giá phim (25.000 train, 25.000 test) được gán nhãn phân cực (tích cực/tiêu cực).
* Ta có thể tải dễ dàng thông qua thư viện `tensorflow_datasets` (`tfds.load`).

### 2. Ý tưởng của Word Embeddings (Vectors)
* Nếu chỉ gán mỗi từ một số nguyên (như bài trước), máy tính không hiểu được mối quan hệ ngữ nghĩa giữa các từ.
* **Giải pháp:** Ánh xạ mỗi từ thành một **Vector** trong không gian nhiều chiều (ví dụ 16 chiều).
* **Kết quả:** Quá trình huấn luyện mạng nơ-ron sẽ tự động điều chỉnh các vector này. Những từ có cùng ngữ nghĩa hoặc xuất hiện trong cùng bối cảnh cảm xúc sẽ "cụm" (cluster) lại gần nhau. 
    * *Ví dụ:* Các từ "dull" (tẻ nhạt) và "boring" (nhàm chán) thường xuất hiện trong review tiêu cực -> Vector của chúng sẽ nằm rất gần nhau. Tương tự với "fun" và "funny".
* Kết quả của Embedding là một mảng 2D với kích thước bằng `(chiều dài câu) x (số chiều embedding)`, mảng này sau đó được đưa vào mạng Dense để phân loại.



---

## IV. Triển khai Code (Implementation)

Dưới đây là các đoạn code thực chiến được tái tạo từ logic bài giảng để kết hợp Tokenizer, Padding và chuẩn bị dữ liệu:

### 1. Code Padding Sequences
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?' # Câu này dài nhất (7 từ)
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# Padding các chuỗi
padded = pad_sequences(
    sequences, 
    padding='post',      # Thêm 0 vào cuối
    maxlen=5,            # Ép độ dài tối đa là 5
    truncating='post'    # Cắt bỏ phần dư ở cuối nếu dài hơn 5
)

print("Sequences gốc:\\n", sequences)
print("Padded Matrix:\\n", padded)

```

### 2. Code tải và tiền xử lý tập dữ liệu IMDB

```python
import tensorflow_datasets as tfds
import numpy as np

# 1. Tải dữ liệu IMDB
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

# 2. Khởi tạo list chứa câu và nhãn
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []

# 3. Lặp qua dữ liệu để tách sentences và labels
for s, l in train_data:
    training_sentences.append(s.numpy().decode('utf8'))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())

# Chuyển labels sang Numpy Array
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

