# **Slot 3: Techniques of CNN in Tensorflow & Lab 3**

**Ngày học:** 12-01-2026

**Môn học:** AI Development with TensorFlow (DAT301m)

**Tài liệu tham khảo:**
- 1.5 Understand Convolutional Neural Networks in TensorFlow
- 1.6 The techniques of Convolutional Neural Networks in TensorFlow
- 2.1 The large  dataset in the Neural  network

---
## I. Understand Convolutional Neural Networks in TensorFlow

### 1. Mục tiêu bài học (Objectives)

* Hiểu về Mạng nơ-ron tích chập (Convolutional Neural Networks - CNN) trong TensorFlow.
* Triển khai được mô hình CNN cho bài toán phân loại (classification).

### 2. Các khái niệm cốt lõi của CNN

**A. Tích chập (Convolutions)**

![Tích chập là gì?]()

> Minh họa về Convolutional Neural Network 

* **Định nghĩa:** Là quá trình thay đổi hình ảnh ban đầu nhằm làm nổi bật (emphasize) các đặc trưng (features) nhất định.
* **Cơ chế hoạt động:** Sử dụng các bộ lọc (filters) để quét qua ảnh:
* Có bộ lọc giúp làm nổi bật các đường nét thẳng đứng (vertical lines).
* Có bộ lọc giúp làm nổi bật các đường nét nằm ngang (horizontal lines).


**B. Gộp (Pooling)**

![Pooling là gì?]()

> Minh họa về Pooling

* **Định nghĩa:** Là kỹ thuật nén ảnh (compressing image) để giảm kích thước dữ liệu nhưng vẫn giữ lại thông tin quan trọng.

* **Max Pooling:** Kỹ thuật phổ biến được giới thiệu trong bài. Với bộ lọc 2x2 (nhóm 4 điểm ảnh), nó sẽ chỉ giữ lại điểm ảnh có giá trị lớn nhất (biggest one will survive).



### 3. Triển khai code trong TensorFlow/Keras

**A. Lớp Tích chập (Convolutional Layer)**
Cú pháp và ý nghĩa các tham số khi khai báo lớp `Conv2D`:

```python
model = tf.keras models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation=' relu',
                            input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation=' relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=' relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

```

* **Số lượng filters:** `64` (Yêu cầu Keras tạo ra 64 bộ lọc khác nhau).
* **Kích thước filter:** `(3, 3)` (Mỗi bộ lọc có kích thước 3x3 pixel).
* **Hàm kích hoạt (Activation):** `relu` (Loại bỏ các giá trị âm).
* **Input shape:** `(28, 28, 1)` (Kích thước ảnh 28x28, số 1 biểu thị cho độ sâu màu - ở đây là ảnh xám/grayscale 1 byte).


**B. Lớp Gộp (Pooling Layer)**

* Sử dụng `MaxPooling2D` với kích thước `(2, 2)` để giảm kích thước ảnh đi một nửa (về mặt diện tích là giảm 4 lần).


**C. Kiểm tra mô hình**

* Sử dụng lệnh `model.summary()` để kiểm tra các lớp mạng và theo dõi hành trình thay đổi kích thước của ảnh qua các lớp tích chập.

**Output:**

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten (Flatten)            (None, 784)               0
dense (Dense)                (None, 128)               100480
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
```

### 4. So sánh và Thảo luận (Neural Network vs. CNN)

* Slide đưa ra sự so sánh giữa mạng Nơ-ron truyền thống (Neural Network - chỉ dùng các lớp Dense) và Mạng nơ-ron tích chập (CNN - thêm các lớp Conv2D và MaxPooling2D) trong bài toán phân loại thời trang (Fashion classifier).


* **Vấn đề thảo luận:** Sự khác biệt về hiệu năng và cách xử lý dữ liệu giữa hai kiến trúc này.

**Tóm tắt ngắn gọn (Takeaway):**
Bài học này nâng cấp mô hình Deep Learning cơ bản bằng cách thêm các "đôi mắt" (Convolutions) để lọc đặc trưng và "kính lúp" (Pooling) để cô đọng thông tin trước khi đưa vào phân loại, giúp máy tính "nhìn" và hiểu hình ảnh hiệu quả hơn.

