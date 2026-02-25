# **Slot 10: NLP: Sarcasm Classifier & Advanced Sequence Models (LSTM)**

**Ngày học:** 05-02-2026
**Tài liệu gốc:** * 3.6 Classifier for the sarcasm dataset.pptx
* LSTM_Tutorial.pdf

**Mục tiêu bài học:**
1. Xây dựng bộ phân loại phát hiện sự mỉa mai (Sarcasm dataset).
2. Hiểu mạng nơ-ron hồi quy (RNN) và các hạn chế của nó.
3. Nắm vững cấu trúc Long Short-Term Memory (LSTM) và cơ chế Attention.

---

## I. Thực hành: Xây dựng Sarcasm Classifier (Phân loại mỉa mai)

Bộ dữ liệu Sarcasm chứa các tiêu đề bài báo được gán nhãn là có mỉa mai (sarcastic) hoặc không.

### 1. Cài đặt Siêu tham số (Hyperparameters)
Để dễ dàng tinh chỉnh mô hình, ta nên tách các tham số cấu hình ra đầu file:

```python
vocab_size = 10000        # Số lượng từ tối đa trong từ điển
embedding_dim = 16        # Số chiều của vector Embedding
max_length = 100          # Độ dài tối đa của mỗi câu
trunc_type = 'post'       # Cắt bỏ phần thừa ở cuối câu
padding_type = 'post'     # Điền số 0 vào cuối câu
oov_tok = "<OOV>"         # Ký hiệu cho từ ngoài từ điển
training_size = 20000     # Số lượng câu dùng để train (còn lại để test)

```

### 2. Quy trình tiền xử lý (Preprocessing)

Các bước tiêu chuẩn cho bài toán NLP:

1. **Split the dataset:** Chia dữ liệu thành tập Train và Test dựa trên `training_size`.
2. **Tokenizer:** Khởi tạo `Tokenizer(num_words=vocab_size, oov_token=oov_tok)` và `fit_on_texts` trên tập Train.
3. **Text to Sequences & Padding:** Chuyển văn bản thành chuỗi số và ép về cùng `max_length`. *(Lưu ý: Phải biến đổi cả tập Train và Test, nhưng Tokenizer CHỈ được học từ tập Train).*

### 3. Xây dựng & Trực quan hóa Mô hình

```python
import tensorflow as tf

# Xây dựng mô hình
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(), # Hoặc dùng Flatten()
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện
history = model.fit(training_padded, training_labels, epochs=30, 
                    validation_data=(testing_padded, testing_labels), verbose=2)

```

* **Visualize the Results:** Sau khi train, cần vẽ biểu đồ Accuracy và Loss để xem mô hình có bị Overfitting hay không. Đối với dữ liệu văn bản nhỏ, Validation Loss thường có xu hướng tăng sau một số epoch, đây là dấu hiệu cần dừng sớm hoặc thêm Dropout.

---

## II. RNN, LSTM và Attention Mechanism

Kiến trúc mạng nơ-ron truyền thống (Feed Forward Network - FFN) không có bộ nhớ và không hiểu được thứ tự của dữ liệu tuần tự (sequential data).

### 1. Recurrent Neural Network (RNN)

* **Khái niệm:** Mạng nơ-ron hồi quy cân nhắc cả đầu vào hiện tại ($X_t$) và các đầu vào đã nhận trước đó để đưa ra quyết định.
* **Ứng dụng:** Dự đoán từ tiếp theo (ví dụ: Google Autocomplete).
* **Hạn chế:** RNN gặp vấn đề **Vanishing Gradient** (đạo hàm biến mất) hoặc **Exploding Gradient** (đạo hàm bùng nổ) khi xử lý chuỗi dài, dẫn đến việc mạng bị "quên" các thông tin ở xa trong quá khứ.

### 2. Long Short-Term Memory (LSTM)

LSTM là một dạng RNN đặc biệt, được thiết kế để giải quyết vấn đề trí nhớ ngắn hạn của RNN.

LSTM kiểm soát luồng thông tin thông qua **Trạng thái tế bào (Cell state)** và 3 cổng (Gates):

* **Cổng quên (Forget Gate):** Quyết định thông tin nào từ quá khứ nên bị vứt bỏ. Dùng hàm Sigmoid (đầu ra từ 0 đến 1, 0 là quên hoàn toàn, 1 là giữ lại).
* **Cổng đầu vào (Input Gate):** Quyết định thông tin mới nào sẽ được lưu trữ vào Cell state. Sử dụng kết hợp hàm Sigmoid và Tanh.
* **Cổng đầu ra (Output Gate):** Quyết định thông tin nào sẽ được xuất ra từ Cell state hiện tại.

### 3. Cơ chế chú ý (Attention Mechanism)

* Trong các mô hình mã hóa-giải mã (Seq2Seq) như dịch thuật, LSTM thường phải nén toàn bộ câu vào một vector duy nhất, gây mất mát thông tin với câu dài.
* **Attention Mechanism** cho phép mô hình "tập trung" (pay attention) vào các phần cụ thể của dữ liệu đầu vào khi tạo ra mỗi phần của đầu ra.
* **Deep Knowledge Tracing (DKT):** Cơ chế Attention có thể được kết hợp để tích hợp kiến thức chuyên gia (expert knowledge) vào mô hình DKT, cải thiện độ chính xác trong giáo dục.


## III. Bổ sung Thực hành: Triển khai LSTM bằng TensorFlow/Keras

Trong bài toán phân loại văn bản (như Sarcasm dataset ở trên), chúng ta đã dùng `GlobalAveragePooling1D` hoặc `Flatten` sau lớp `Embedding`. Tuy nhiên, cách này làm mất đi **thứ tự từ** trong câu. 

Để mô hình hiểu được ngữ cảnh và thứ tự từ, ta sẽ thay thế chúng bằng lớp **LSTM**.

### 1. Code Triển khai LSTM cơ bản

Dưới đây là kiến trúc mạng nơ-ron sử dụng LSTM để phân loại văn bản:

```python
import tensorflow as tf

# Kế thừa các hyperparameter từ bài Sarcasm
vocab_size = 10000
embedding_dim = 16
max_length = 100

# Xây dựng mô hình với LSTM
model_lstm = tf.keras.Sequential([
    # Lớp 1: Embedding (Chuyển số thành Vector)
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    
    # Lớp 2: LSTM (Thay thế cho Flatten/GlobalAveragePooling)
    # Tham số 64 là số lượng unit (chiều của cell state/hidden state)
    tf.keras.layers.LSTM(64),
    
    # Lớp 3: Dense layer để học các đặc trưng phức tạp
    tf.keras.layers.Dense(24, activation='relu'),
    
    # Lớp 4: Lớp Output cho Binary Classification
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Kiểm tra cấu trúc mạng
model_lstm.summary()

```

### 2. Nâng cấp: Bidirectional LSTM (LSTM 2 chiều)

Trong ngôn ngữ tự nhiên, ý nghĩa của một từ không chỉ phụ thuộc vào từ đứng trước nó, mà còn phụ thuộc vào từ đứng sau nó. (Ví dụ: "Con chuột *máy tính*" vs "Con chuột *cống*").

Để giải quyết vấn đề này, TensorFlow cung cấp hàm bọc `Bidirectional()`, cho phép LSTM đọc câu theo cả 2 chiều (từ trái sang phải, và từ phải sang trái).

```python
model_bilstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    
    # Bọc lớp LSTM bên trong Bidirectional
    # Kết quả trả về sẽ có kích thước gấp đôi (64 * 2 = 128)
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

```

### 3. Huấn luyện mô hình (Training)

Cách Compile và Fit hoàn toàn giống với mạng nơ-ron truyền thống:

```python
# Compile mô hình
model_bilstm.compile(loss='binary_crossentropy', 
                     optimizer='adam', 
                     metrics=['accuracy'])

# Huấn luyện mô hình
# (Giả sử training_padded và training_labels đã được chuẩn bị từ trước)
history_lstm = model_bilstm.fit(training_padded, training_labels, 
                                epochs=10, 
                                validation_data=(testing_padded, testing_labels), 
                                verbose=1)

```

---

## IV. Tổng kết so sánh (Takeaways)

1. **Mô hình dùng Flatten / GlobalAveragePooling1D:**
* Tốc độ train rất nhanh.
* Hiệu năng khá tốt với các câu đơn giản.
* KHÔNG hiểu ngữ pháp, chỉ nhìn vào sự xuất hiện của từng từ riêng lẻ.


2. **Mô hình dùng LSTM / Bidirectional LSTM:**
* Tốc độ train chậm hơn đáng kể (do phải tính toán tuần tự qua các cell state).
* Hiểu được ngữ cảnh, thứ tự từ và các mối quan hệ xa (Long-term dependencies) trong câu.
* Rất nhạy cảm với hiện tượng Overfitting, thường cần thêm lớp `Dropout()` nếu dữ liệu quá nhỏ.

