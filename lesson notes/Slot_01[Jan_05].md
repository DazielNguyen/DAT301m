# **Slot 1: Course Intro & TensorFlow Basics**

**Môn học:** AI Development with TensorFlow (DAT301m)

**Thời gian:** Slot 11

**Tài liệu buổi học:**

- 0.0. Course Introduction.ppt
- 1.1 Working with TensorFlow.pptx
- 1.2 Machine learning and computer vision in TensorFlow.pptx

---

## I. Tổng quan môn học (Course Overview)

* **Mục tiêu:** Áp dụng Deep Learning với TensorFlow cho 3 mảng: Computer Vision (CV), NLP, và Time Series.
* **Yêu cầu:** Đã có kiến thức nền tảng về Python, Machine Learning cơ bản.
* **Đánh giá:** 4 bài Lab thực hành (Happy/Sad, Transfer Learning, Poetry Generation, Forecasting) và các bài Quiz/Test.
* **Lưu ý:** Nghiêm cấm đạo văn và sao chép code (Plagiarism).

---

## II. Giới thiệu về TensorFlow (TF)

### 1. TensorFlow là gì?
* Là nền tảng mã nguồn mở **End-to-End** cho Machine Learning, phát triển bởi Google Brain.
* **Hỗ trợ đa nền tảng:** Chạy trên Linux, Windows, macOS, Android, iOS.
* **Hỗ trợ phần cứng:** Tối ưu hóa trên CPU, GPU và đặc biệt là **TPU** (Tensor Processing Unit - chip chuyên dụng cho ML).

### 2. Các thành phần cốt lõi
* **Tensors:**
    * Là cấu trúc dữ liệu cơ bản (mảng đa chiều/multidimensional arrays).
    * Các thuộc tính quan trọng: `Tensor.shape` (kích thước), `Tensor.dtype` (kiểu dữ liệu).
* **Keras API (`tf.keras`):**
    * Là High-level API của TensorFlow.
    * Được thiết kế cho con người (human-centric), dễ sử dụng, cho phép thử nghiệm nhanh (fast experimentation).
    * **Layers:** Class `tf.keras.layers.Layer` là khối xây dựng cơ bản (chứa weights và tính toán).
    * **Models:** Class `tf.keras.Model` nhóm các layers lại thành một mô hình có thể train được.

---

## III. Tư duy Machine Learning & Computer Vision

### 1. Sự chuyển dịch tư duy (Paradigm Shift)
Khác biệt giữa Lập trình truyền thống và Machine Learning:

* **Traditional Programming:**
    * Input: Dữ liệu (Data) + Quy tắc (Rules/Code).
    * Output: Câu trả lời (Answers).
    * *Ví dụ:* Viết lệnh `if speed < 4: return WALKING`.
* **Machine Learning:**
    * Input: Dữ liệu (Data) + Câu trả lời mẫu (Answers/Labels).
    * Output: **Quy tắc (Rules)**.
    * *Ví dụ:* Đưa dữ liệu tốc độ và nhãn hành động -> Máy tự học quy tắc phân loại đi bộ/chạy/đạp xe.

### 2. Computer Vision (Thị giác máy tính)
* **Định nghĩa:** Lĩnh vực giúp máy tính "hiểu" và dán nhãn nội dung trong hình ảnh.
* **Ví dụ minh họa:** Phân loại quần áo (Fashion MNIST). Thay vì code thủ công mô tả chiếc giày trông như thế nào, ta nạp hàng nghìn ảnh giày, áo, túi xách để máy tự tìm ra đặc trưng (patterns).

### 3. Công cụ hỗ trợ CV trong TensorFlow
* **KerasCV:** Thư viện chứa các module CV xây dựng trên Keras Core (Data augmentation, object detection...).
* **`tf.image`:** Module xử lý ảnh mức thấp (Low-level), dùng để tự viết các pipeline xử lý ảnh (Flip, Grayscale, Crop...).
* **TensorFlow Datasets:** Kho dữ liệu có sẵn, hiệu năng cao (`tf.data.Datasets`).

---

## IV. Quy trình & Code mẫu (Implementation Steps)

Quy trình chuẩn khi làm việc với TensorFlow/Keras:

1.  **Load Data:** Tải dữ liệu và tiền xử lý (Chuẩn hóa/Normalize).
2.  **Build Model:** Xếp chồng các Layers (Sequential).
3.  **Compile:** Chọn Optimizer, Loss function và Metrics.
4.  **Fit:** Huấn luyện mô hình.
5.  **Evaluate/Predict:** Đánh giá và dự đoán.

### Code Demo: Phân loại ảnh (Fashion MNIST)

```python
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load Data & Pre-processing
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize dữ liệu về khoảng [0, 1] (Pixel gốc là 0-255)
training_images = training_images / 255.0
test_images = test_images / 255.0

# 2. Build Model (Sequential)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),                  # Duỗi ảnh 2D thành vector 1D
    tf.keras.layers.Dense(128, activation=tf.nn.relu), # Lớp ẩn với hàm kích hoạt ReLU
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # Lớp output (10 classes) dùng Softmax
])

# 3. Compile Model
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train Model
model.fit(training_images, training_labels, epochs=5)

# 5. Evaluate
model.evaluate(test_images, test_labels)