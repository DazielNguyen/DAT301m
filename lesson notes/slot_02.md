# **Slot 2: Google Colab & Building Computer Vision Models**

**Ngày học:** 08-01-2026

**Môn học:** AI Development with TensorFlow (DAT301m)

**Tài liệu tham khảo:**
- 1.3 Coding with TensorFlow in Google Colaboratory.pptx
- 1.4 Computer Vision Neural Network.pptx

---

## I. Môi trường thực hành: Google Colab

### 1. Google Colab là gì?
* Là môi trường Jupyter Notebook được lưu trữ trên cloud (hosted) bởi Google.
* **Ưu điểm:**
    * Không cần cài đặt (No setup required).
    * Miễn phí truy cập GPU/TPU (Rất quan trọng để train model nhanh).
    * TensorFlow đã được cài sẵn.

### 2. Các bước thiết lập cơ bản
1.  Truy cập: `colab.research.google.com`
2.  Đăng nhập Google Account -> Create New Notebook.
3.  **Bật GPU:** Vào menu *Runtime* -> *Change runtime type* -> Chọn *GPU* (hoặc TPU).

### 3. Lỗi phổ biến 
- Cách khắc phục khi chạy Tensorflow mà không dùng được GPU để chạy model. (Thường sẽ gặp trên các máy Window)

[Installing TensorFlow 2 GPU [Step-by-Step Guide]d](https://neptune.ai/blog/installing-tensorflow-2-gpu-guide)

---

## II. Xây dựng Mạng Nơ-ron cho Computer Vision

Quy trình xây dựng một mô hình Deep Learning chuẩn gồm 4 bước chính:

### Bước 1: Chuẩn bị dữ liệu (Data Loading & Preprocessing)
* **Dataset:** Fashion-MNIST (Bộ dữ liệu quần áo của Zalando).
    * Số lượng: 60,000 ảnh Train, 10,000 ảnh Test.
    * Kích thước: 28x28 pixel (Grayscale - ảnh xám).
    * Nhãn (Labels): 10 loại (0: Áo thun, 1: Quần, 9: Giày boot...).
* **Normalization (Chuẩn hóa):** Chia giá trị pixel cho 255.
    ```python
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    ```
    > **Giải thích thuật ngữ:**
    > * **Tại sao chia cho 255?**
    >     * *Minh họa:* Pixel ảnh có giá trị từ 0 (đen) đến 255 (trắng). Máy tính xử lý số nhỏ (từ 0 đến 1) nhanh và ổn định hơn số lớn. Việc này giống như việc quy đổi tiền tệ từ VNĐ sang USD để con số nhỏ gọn hơn, dễ tính toán hơn.

### Bước 2: Định nghĩa Mô hình (Model Definition)

Sử dụng `tf.keras.models.Sequential` (Mô hình tuần tự - các lớp xếp chồng lên nhau).

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

```

> **Giải thích thuật ngữ & Minh họa:**
> 1. **Flatten (Làm phẳng):**
> * *Chức năng:* Biến đổi ảnh 2D (hình vuông 28x28) thành mảng 1D (một hàng dọc 784 điểm).
> * *Minh họa:* Giống như bạn tháo một chiếc hộp giấy vuông ra và trải phẳng nó lên mặt bàn để dễ quan sát tất cả các mặt cùng lúc.
> 
> 2. **Dense (Lớp dày đặc):**
> * *Chức năng:* Lớp nơ-ron thông thường, nơi mọi nơ-ron kết nối với tất cả nơ-ron lớp trước.
> * *Minh họa:* Giống như một cuộc họp mà **tất cả** mọi người đều bắt tay nhau. 128 nơ-ron là 128 "chuyên gia" đang cố gắng tìm ra các đặc điểm của bức ảnh.
> 
> 3. **Relu (Hàm kích hoạt):**
> * *Quy tắc:* `Nếu x > 0 thì giữ nguyên, nếu x < 0 thì bằng 0`.
> * *Minh họa:* Giống như một **bộ lọc tiếng ồn**. Những tín hiệu tiêu cực (không quan trọng) bị loại bỏ, chỉ giữ lại tín hiệu tích cực (quan trọng) để truyền sang lớp sau.
>
> 4. **Softmax:**
> * *Chức năng:* Dùng ở lớp cuối cùng. Biến đổi các con số lộn xộn thành **xác suất (%)** sao cho tổng bằng 100%.
> * *Ví dụ:* Thay vì nói "Điểm số là 5, 2, 9", Softmax sẽ nói "Khả năng là Áo thun: 10%, Quần: 5%, Giày: 85%". Ta chọn cái cao nhất.

### Bước 3: Biên dịch & Huấn luyện (Compile & Train)

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

```

> **Giải thích thuật ngữ:**
> * **Loss Function (Hàm mất mát):** Thước đo xem mô hình đoán **SAI** bao nhiêu. Mục tiêu là giảm Loss càng thấp càng tốt.
> * **Optimizer (Bộ tối ưu hóa):** "Người dẫn đường" dựa trên Loss để điều chỉnh lại các tham số (weights) sao cho lần đoán sau chính xác hơn.
> * **Epochs:** Số lần mô hình được học trọn vẹn bộ dữ liệu. 5 epochs nghĩa là nó xem đi xem lại sách giáo khoa 5 lần.

---

## III. Kiểm soát quy trình Train (Callbacks)

Vấn đề: Làm sao để dừng train khi mô hình đã đủ tốt (để tiết kiệm thời gian và tránh học vẹt)? -> Sử dụng **Callbacks**.

Các loại Callbacks phổ biến:

1. **ModelCheckpoint:**
* Tự động lưu lại mô hình (save game) mỗi khi nó đạt kết quả tốt nhất. Giúp bạn không bị mất công sức nếu máy tính sập nguồn.

2. **EarlyStopping:**
* Tự động dừng train nếu thấy mô hình không còn tiến bộ nữa (học mãi không giỏi thêm thì cho nghỉ sớm).

3. **TensorBoard:**
* Công cụ vẽ biểu đồ trực quan quá trình train.

### Code mẫu sử dụng Callback (Dừng khi loss < 0.4):

```python
class MyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss') < 0.4):
      print("\nĐã đạt Loss < 0.4, dừng train!")
      self.model.stop_training = True

callbacks = MyCallback()
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

```

### Các cách sử dụng Callbacks nâng cao:

#### 1. EarlyStopping - Dừng sớm khi không cải thiện
```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',      # Theo dõi validation loss
    patience=3,              # Chờ 3 epochs không cải thiện
    restore_best_weights=True # Khôi phục weights tốt nhất
)
```

#### 2. ModelCheckpoint - Lưu model tốt nhất
```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
```

#### 3. ReduceLROnPlateau - Giảm learning rate tự động
```python
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,              # Giảm LR xuống 50%
    patience=2,
    min_lr=1e-7,
    verbose=1
)
```

#### 4. TensorBoard - Trực quan hóa quá trình train
```python
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True
)
```

#### 5. Kết hợp nhiều Callbacks
```python
model.fit(
    training_images, training_labels,
    validation_data=(test_images, test_labels),
    epochs=50,
    callbacks=[early_stopping, checkpoint, reduce_lr, tensorboard]
)
```

---

## III.A. Cách tính Parameters và So sánh với GPU

### 1. Công thức tính Parameters (Tham số)

Mỗi lớp Dense có số parameters được tính như sau:
$$\text{Parameters} = (\text{Input} \times \text{Output}) + \text{Bias}$$

**Ví dụ với model Fashion-MNIST:**
```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),  # 0 params
  tf.keras.layers.Dense(128, activation='relu'),  # ?
  tf.keras.layers.Dense(10, activation='softmax') # ?
])
```

**Tính toán chi tiết:**
1. **Flatten:** 0 parameters (chỉ reshape, không học gì)
2. **Dense(128):** 
   - Input: 28×28 = 784 neurons
   - Output: 128 neurons
   - Parameters = (784 × 128) + 128 = **100,480** parameters
3. **Dense(10):**
   - Input: 128 neurons
   - Output: 10 neurons  
   - Parameters = (128 × 10) + 10 = **1,290** parameters

**Tổng cộng: 101,770 parameters**

### 2. Kiểm tra Parameters trong code

```python
model.summary()
```

Output:
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

### 3. So sánh với GPU - Có đủ sức chạy không?

#### Bộ nhớ cần thiết cho training:
$$\text{Memory} = \text{Parameters} \times \text{Bytes per param} \times \text{Overhead factor}$$

**Ví dụ tính toán thực tế:**
- Model có 101,770 params
- Mỗi param dùng float32 = 4 bytes
- Overhead (gradients, optimizer states) ≈ **4x** (Adam optimizer cần lưu momentum)

```
Memory needed = 101,770 × 4 bytes × 4 
              ≈ 1.6 MB (chỉ cho model + gradients)
```

**Với batch size = 32:**
```
Batch memory = 32 × 784 × 4 bytes ≈ 100 KB
Total ≈ 1.7 MB
```

#### Bảng so sánh GPU phổ biến:

| GPU Model | VRAM | Có thể train model gì? |
|-----------|------|------------------------|
| **GTX 1650** | 4GB | Models < 50M params (ResNet-18, MobileNet) |
| **RTX 3060** | 12GB | Models < 200M params (ResNet-50, EfficientNet) |
| **RTX 4090** | 24GB | Models < 500M params (BERT-large, GPT-2) |
| **A100 (Colab Pro)** | 40GB | Models < 1B params (GPT-3 small, CLIP) |

**Quy tắc ngón tay cái:**
- Model nhỏ (< 10M params): Chạy được trên CPU/GPU cơ bản
- Model vừa (10M-100M params): Cần GPU có 6GB+ VRAM
- Model lớn (100M-1B params): Cần GPU chuyên nghiệp (16GB+)
- Model siêu lớn (> 1B params): Cần nhiều GPU hoặc TPU

**Công thức ước lượng nhanh:**
```python
def estimate_memory_gb(num_params, batch_size=32, bytes_per_param=4, overhead=4):
    """
    Ước lượng VRAM cần thiết
    overhead: 4 cho Adam, 2 cho SGD
    """
    model_memory = num_params * bytes_per_param * overhead
    return model_memory / (1024**3)  # Convert to GB

# Ví dụ: ResNet-50 có 25M params
print(f"ResNet-50 cần: {estimate_memory_gb(25_000_000):.2f} GB")
# Output: ResNet-50 cần: 0.37 GB (thực tế cần 2-3GB do activation maps)
```

---

## III.B. TensorBoard - Trực quan hóa quá trình Training

### 1. TensorBoard là gì?
TensorBoard là công cụ trực quan hóa của TensorFlow, giúp bạn:
- Theo dõi loss và accuracy theo thời gian
- Xem kiến trúc mô hình (graph)
- Phân tích phân phối weights
- So sánh nhiều lần chạy

### 2. Setup TensorBoard từng bước

#### Bước 1: Import và tạo log directory
```python
import tensorflow as tf
import datetime
import os

# Tạo thư mục logs với timestamp
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,      # Ghi histogram mỗi epoch
    write_graph=True,      # Lưu computational graph
    write_images=True,     # Lưu ảnh (nếu có)
    update_freq='epoch',   # Cập nhật mỗi epoch
    profile_batch='500,520' # Profile batch 500-520
)
```

#### Bước 2: Train model với TensorBoard callback
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train với TensorBoard callback
history = model.fit(
    training_images, training_labels,
    validation_data=(test_images, test_labels),
    epochs=10,
    callbacks=[tensorboard_callback]
)
```

#### Bước 3: Khởi chạy TensorBoard

**Trong Jupyter Notebook/Colab:**
```python
# Load extension
%load_ext tensorboard

# Khởi chạy TensorBoard
%tensorboard --logdir logs/fit
```

**Trong Terminal:**
```bash
tensorboard --logdir logs/fit --port 6006
# Mở browser: http://localhost:6006
```

### 3. Các Tab quan trọng trong TensorBoard

#### Tab SCALARS - Theo dõi metrics
```python
# Log custom metrics
file_writer = tf.summary.create_file_writer(log_dir)

with file_writer.as_default():
    for epoch in range(10):
        # Giả lập metrics
        loss = 1.0 / (epoch + 1)
        accuracy = epoch / 10.0
        
        tf.summary.scalar('custom_loss', loss, step=epoch)
        tf.summary.scalar('custom_accuracy', accuracy, step=epoch)
```

#### Tab GRAPHS - Xem kiến trúc model
```python
# TensorBoard tự động tạo graph khi write_graph=True
# Bạn sẽ thấy visual representation của model architecture
```

#### Tab DISTRIBUTIONS - Phân phối weights
```python
# Xem phân phối weights/biases của từng layer qua các epoch
# Giúp phát hiện:
# - Vanishing gradients (weights gần 0)
# - Exploding gradients (weights quá lớn)
```

#### Tab HISTOGRAMS - Chi tiết hơn distributions
```python
# Hiển thị histogram 3D của weights theo thời gian
```

### 4. So sánh nhiều lần chạy

```python
# Run 1: Learning rate = 0.001
log_dir_1 = "logs/fit/run1_lr0.001"
tensorboard_callback_1 = tf.keras.callbacks.TensorBoard(log_dir=log_dir_1)

# Run 2: Learning rate = 0.01
log_dir_2 = "logs/fit/run2_lr0.01"
tensorboard_callback_2 = tf.keras.callbacks.TensorBoard(log_dir=log_dir_2)

# Trong TensorBoard, cả 2 runs sẽ hiển thị trên cùng 1 đồ thị
%tensorboard --logdir logs/fit
```

### 5. Tips sử dụng TensorBoard hiệu quả

```python
# Tip 1: Sử dụng name prefix rõ ràng
log_dir = f"logs/{model_name}/lr_{learning_rate}/batch_{batch_size}/{timestamp}"

# Tip 2: Log learning rate
lr_callback = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-3 * 0.9 ** epoch
)

# Tip 3: Custom callback để log thêm metrics
class CustomTensorBoard(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        with self.file_writer.as_default():
            # Log custom metrics
            tf.summary.scalar('learning_rate', 
                            self.model.optimizer.lr.numpy(), 
                            step=epoch)
```

---

## III.C. TensorFlow Graph (tf.Graph)

### 1. tf.Graph là gì?

**Định nghĩa:** 
tf.Graph là một cấu trúc dữ liệu biểu diễn các phép tính (operations) dưới dạng **đồ thị có hướng** (directed graph).

**Minh họa:**
```
Input (28x28) 
    ↓
[Flatten: 784 neurons]
    ↓
[Dense: 128 neurons] ← Weights (784×128)
    ↓
[ReLU Activation]
    ↓
[Dense: 10 neurons] ← Weights (128×10)
    ↓
[Softmax]
    ↓
Output (10 classes)
```

### 2. Tại sao cần tf.Graph?

**Lợi ích:**
1. **Tối ưu hóa:** TensorFlow có thể tối ưu toán đồ thị trước khi chạy (fusion, pruning)
2. **Parallel execution:** Các node độc lập chạy song song
3. **Deployment:** Export graph để deploy lên mobile/server
4. **Performance:** Eager execution (TF 2.x) vs Graph mode (TF 1.x)

### 3. Eager Execution vs Graph Mode

#### Eager Execution (TF 2.x - Default)
```python
# Code chạy ngay lập tức, giống Python thông thường
import tensorflow as tf

x = tf.constant([[1.0, 2.0]])
y = tf.constant([[3.0], [4.0]])
result = tf.matmul(x, y)
print(result.numpy())  # [[11.]]
```

#### Graph Mode (Dùng @tf.function)
```python
@tf.function  # Decorator này compile thành graph
def my_function(x, y):
    return tf.matmul(x, y)

x = tf.constant([[1.0, 2.0]])
y = tf.constant([[3.0], [4.0]])
result = my_function(x, y)
print(result.numpy())  # [[11.]] - Nhanh hơn eager mode
```

### 4. Visualize Graph với TensorBoard

```python
# Tạo một model đơn giản
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Tạo dummy data
import numpy as np
dummy_x = np.random.random((1, 784))
dummy_y = np.array([5])

# Trace graph bằng tf.function
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    return loss

# Log graph vào TensorBoard
log_dir = "logs/graph"
writer = tf.summary.create_file_writer(log_dir)

# Trace và log
tf.summary.trace_on(graph=True, profiler=True)
train_step(dummy_x, dummy_y)
with writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0)
```

### 5. Khi nào dùng @tf.function?

**Nên dùng khi:**
- Training loop lặp đi lặp lại nhiều lần
- Deploy model lên production (tăng tốc 10-50x)
- Model phức tạp, cần tối ưu performance

**Không nên dùng khi:**
- Debugging (khó debug hơn eager mode)
- Prototype nhanh
- Logic phức tạp với Python conditionals

---

## III.D. Max Pooling - Giảm kích thước dữ liệu

### 1. Max Pooling là gì?

**Định nghĩa:** 
Max Pooling là kỹ thuật **giảm kích thước** (downsampling) của feature map bằng cách chọn **giá trị lớn nhất** trong một vùng nhỏ.

**Minh họa trực quan:**

```
Input (4×4):                  MaxPool 2×2:
┌─────────────┐              ┌─────────┐
│ 1  3  2  4 │              │ 3  4   │
│ 5  6  7  8 │    ──→       │ 9  11  │
│ 3  2  1  0 │              └─────────┘
│ 9  7  5  11│
└─────────────┘

Cách hoạt động:
┌───┬───┐───┬───┐
│ 1 │ 3 │ 2 │ 4 │  → Max(1,3,5,6) = 6  → 3
├───┼───┼───┼───┤     Max(2,4,7,8) = 8  → 4
│ 5 │ 6 │ 7 │ 8 │
├───┼───┼───┼───┤
│ 3 │ 2 │ 1 │ 0 │  → Max(3,2,9,7) = 9  → 9
├───┼───┼───┼───┤     Max(1,0,5,11) = 11 → 11
│ 9 │ 7 │ 5 │11│
└───┴───┴───┴───┘
```

### 2. Tại sao cần Max Pooling?

**Lợi ích:**
1. **Giảm số lượng parameters:** Ít neurons hơn → Model nhẹ hơn
2. **Giảm overfitting:** Bỏ đi thông tin chi tiết không quan trọng
3. **Translation invariance:** Nhận diện vật thể dù nó dịch chuyển vị trí
4. **Tăng receptive field:** Mỗi neuron "nhìn" vùng lớn hơn của ảnh gốc

**Ví dụ thực tế:**
```
Ảnh 28×28 → Conv(32 filters) → 28×28×32 (còn lớn)
          ↓
MaxPool 2×2 → 14×14×32 (giảm 75% kích thước)
          ↓
Conv(64 filters) → 14×14×64
          ↓
MaxPool 2×2 → 7×7×64 (giảm tiếp 75%)
          ↓
Flatten → 3,136 neurons (thay vì 25,088)
```

### 3. Code sử dụng Max Pooling

```python
model = tf.keras.models.Sequential([
    # Conv Layer 1
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', 
                          input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),  # 28×28 → 14×14
    
    # Conv Layer 2
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),  # 14×14 → 7×7
    
    # Fully Connected
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
```

Output:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
max_pooling2d_1 (MaxPooling2D)(None, 5, 5, 64)         0         
flatten (Flatten)            (None, 1600)              0         
dense (Dense)                (None, 128)               204928    
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 225,034
```

### 4. So sánh các loại Pooling

| Loại | Công thức | Khi nào dùng |
|------|-----------|--------------|
| **Max Pooling** | $\max(x_{ij})$ | Nhận diện features nổi bật (edges, textures) |
| **Average Pooling** | $\frac{1}{n}\sum x_{ij}$ | Làm mượt, giảm noise |
| **Global Average Pooling** | Average toàn bộ feature map | Thay thế Flatten, giảm overfitting |

```python
# Max Pooling
tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

# Average Pooling
tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)

# Global Average Pooling (giảm 7×7×64 → 1×1×64)
tf.keras.layers.GlobalAveragePooling2D()
```

### 5. Visualize hiệu ứng Max Pooling

```python
import matplotlib.pyplot as plt
import numpy as np

# Tạo ảnh test
img = training_images[0].reshape(28, 28)

# Tạo model chỉ có Conv + MaxPool
test_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', 
                          input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2)
])

# Xem kết quả
result = test_model.predict(img.reshape(1, 28, 28, 1))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original (28×28)')

axes[1].imshow(result[0, :, :, 0], cmap='gray')
axes[1].set_title('After Conv (26×26)')

axes[2].imshow(result[0, :, :, 0], cmap='gray')
axes[2].set_title('After MaxPool (13×13)')
plt.show()
```

---

## III.E. Black Box trong Deep Learning

### 1. Black Box là gì?

**Định nghĩa:**
"Black Box" là hiện tượng mà ta **biết input và output**, nhưng **không hiểu cách mô hình đưa ra quyết định bên trong**.

**Minh họa:**
```
┌──────────────────────────────────┐
│        🎁 BLACK BOX              │
│                                  │
│  Input: Ảnh chó                 │
│     ↓                           │
│  [784 neurons]                  │
│     ↓                           │
│  [128 neurons] ← ??? weights    │
│     ↓                           │
│  [64 neurons]  ← ??? logic      │
│     ↓                           │
│  [10 neurons]                   │
│     ↓                           │
│  Output: 95% chắc là chó        │
│                                  │
│  ❓ Tại sao lại 95%?            │
│  ❓ Nó nhìn vào đâu?            │
│  ❓ Nếu sai, sai ở đâu?         │
└──────────────────────────────────┘
```

### 2. Tại sao Deep Learning là Black Box?

**Nguyên nhân:**
1. **Quá nhiều parameters:** Model có hàng triệu weights, con người không thể xem hết
2. **Non-linear transformations:** Nhiều lớp ReLU, Sigmoid làm relationship phức tạp
3. **High-dimensional space:** Dữ liệu được transform qua không gian 128D, 256D...

**Ví dụ:**
```python
# Model Fashion-MNIST: 101,770 parameters
# Làm sao kiểm tra 101,770 con số này để hiểu logic?
model.get_weights()[0].shape  # (784, 128) = 100,352 weights
```

### 3. Tại sao Black Box lại là vấn đề?

**Trong các lĩnh vực quan trọng:**
- **Y tế:** AI chẩn đoán ung thư → Bác sĩ cần biết "Tại sao AI nghĩ là ung thư?"
- **Pháp lý:** AI từ chối cho vay → Người dùng có quyền biết lý do
- **Tự lái xe:** AI quyết định phanh/rẽ → Cần giải thích cho tai nạn

**Ví dụ thực tế:** 
Model phân loại ảnh động vật đạt 98% accuracy, nhưng khi test:
- Model dự đoán "CHÓ" khi thấy ảnh có... **cỏ xanh** (vì ảnh chó trong dataset đều có cỏ)
- Model không học đặc điểm CHÓ, mà học **background pattern**!

### 4. Các kỹ thuật "mở" Black Box (Explainable AI)

#### A. Visualize Intermediate Layers
```python
# Xem layer 1 học được gì
layer_outputs = [layer.output for layer in model.layers[:3]]
activation_model = tf.keras.models.Model(
    inputs=model.input, 
    outputs=layer_outputs
)

# Predict và xem activation
activations = activation_model.predict(test_images[0].reshape(1, 28, 28, 1))

# Plot
import matplotlib.pyplot as plt
plt.imshow(activations[0][0, :, :, 0], cmap='viridis')
plt.title('Layer 1 - Feature Map 0')
plt.show()
```

#### B. Grad-CAM (Gradient-weighted Class Activation Mapping)
```python
# Hiển thị vùng nào model "nhìn" để đưa ra quyết định
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Tạo model output cả predictions và activations
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradient của class wrt activations
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight activation maps by gradients
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
```

#### C. LIME (Local Interpretable Model-agnostic Explanations)
```python
# Install: pip install lime
from lime import lime_image

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    test_images[0], 
    model.predict, 
    top_labels=3, 
    hide_color=0, 
    num_samples=1000
)

# Hiển thị vùng quan trọng nhất
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], 
    positive_only=True, 
    num_features=5, 
    hide_rest=False
)
plt.imshow(temp)
```

#### D. Feature Importance với Permutation
```python
# Đo xem feature nào quan trọng bằng cách shuffle nó
import numpy as np

def feature_importance(model, X_test, y_test):
    baseline_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    importance = []
    
    for i in range(X_test.shape[1]):
        X_permuted = X_test.copy()
        np.random.shuffle(X_permuted[:, i])  # Shuffle feature i
        permuted_acc = model.evaluate(X_permuted, y_test, verbose=0)[1]
        importance.append(baseline_acc - permuted_acc)
    
    return importance
```

### 5. Trade-off: Accuracy vs Interpretability

```
High Interpretability
    ↑
    │  Linear Regression
    │  Decision Tree (shallow)
    │  
    │  Random Forest
    │  
    │  Neural Network (small)
    │  
    │  Deep Neural Network
    │  ResNet, Transformers
    ↓
Low Interpretability (Black Box)
    
    Low Accuracy → High Accuracy →
```

**Nguyên tắc:**
- **High-stakes decisions:** Ưu tiên interpretability (y tế, pháp lý)
- **Low-stakes, high-volume:** Ưu tiên accuracy (gợi ý phim, quảng cáo)

---

## III.F. Accuracy vs Validation Accuracy - Hiểu đúng để tránh Overfitting

### 1. Định nghĩa và sự khác biệt

| Metric | Định nghĩa | Ý nghĩa |
|--------|-----------|---------|
| **Accuracy** | Độ chính xác trên **training set** | Model học tốt dữ liệu huấn luyện như thế nào |
| **Val_accuracy** | Độ chính xác trên **validation set** | Model tổng quát hóa tốt với dữ liệu chưa thấy như thế nào |

**Minh họa:**
```
Dataset (70,000 ảnh)
    ↓
├─ Training Set (60,000) → Dùng để học
│                        → Tính ACCURACY
│
└─ Validation Set (10,000) → Dùng để kiểm tra
                           → Tính VAL_ACCURACY
```

### 2. Các trường hợp phân tích

#### Case 1: Healthy Model (Mô hình tốt)
```
Epoch 1: loss=0.5, acc=0.85  | val_loss=0.52, val_acc=0.83
Epoch 2: loss=0.4, acc=0.88  | val_loss=0.43, val_acc=0.86
Epoch 3: loss=0.3, acc=0.91  | val_loss=0.35, val_acc=0.89
Epoch 4: loss=0.25, acc=0.93 | val_loss=0.30, val_acc=0.91
```
**Đánh giá:** ✅ Cả 2 đều tăng đều → Model học tốt và tổng quát hóa tốt

#### Case 2: Overfitting (Học vẹt)
```
Epoch 1: loss=0.5, acc=0.85  | val_loss=0.52, val_acc=0.83
Epoch 2: loss=0.3, acc=0.92  | val_loss=0.45, val_acc=0.87
Epoch 3: loss=0.15, acc=0.96 | val_loss=0.55, val_acc=0.85 ⚠️
Epoch 4: loss=0.08, acc=0.98 | val_loss=0.70, val_acc=0.82 ⚠️
```
**Đánh giá:** ❌ Accuracy tăng, Val_accuracy giảm → Model học thuộc lòng training data

**Minh họa trực quan:**
```
Accuracy vs Val_Accuracy

Acc ┐
    │     ╱───────────────── Training (tiếp tục tăng)
100%│   ╱
    │  ╱
 90%│ ╱    ╱───╲  Validation (tăng rồi giảm)
    │╱   ╱      ╲___
 80%├───────────────────────► Epochs
    0   5   10  15  20
        ↑
    Sweet spot (dừng ở đây!)
```

#### Case 3: Underfitting (Chưa học đủ)
```
Epoch 1: loss=0.8, acc=0.60  | val_loss=0.82, val_acc=0.58
Epoch 2: loss=0.75, acc=0.62 | val_loss=0.78, val_acc=0.60
Epoch 3: loss=0.72, acc=0.64 | val_loss=0.75, val_acc=0.62
```
**Đánh giá:** ⚠️ Cả 2 đều thấp và tăng chậm → Model quá đơn giản

### 3. Code để monitor và visualize

```python
import matplotlib.pyplot as plt

# Train model và lưu history
history = model.fit(
    training_images, training_labels,
    validation_data=(test_images, test_labels),
    epochs=20,
    verbose=1
)

# Plot Accuracy vs Val_Accuracy
plt.figure(figsize=(12, 4))

# Plot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.grid(True)

# Plot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.grid(True)

plt.tight_layout()
plt.show()

# Tìm epoch tốt nhất
best_epoch = np.argmax(history.history['val_accuracy'])
print(f"Best epoch: {best_epoch + 1}")
print(f"Val_accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
```

### 4. Giải pháp khi có Overfitting (Accuracy >> Val_Accuracy)

#### Solution 1: Early Stopping
```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,                   # Dừng nếu 5 epochs không cải thiện
    restore_best_weights=True,
    mode='max'
)
```

#### Solution 2: Dropout (Bỏ ngẫu nhiên neurons)
```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Bỏ 30% neurons mỗi iteration
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Solution 3: Regularization (L1/L2 - sẽ giải thích chi tiết sau)
```python
from tensorflow.keras import regularizers

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

#### Solution 4: Data Augmentation (Tăng cường dữ liệu)
```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
```

### 5. Rule of thumb - Quy tắc ngón tay cái

| Gap (Accuracy - Val_Accuracy) | Tình trạng | Hành động |
|-------------------------------|-----------|-----------|
| **< 5%** | ✅ Healthy | Tiếp tục train hoặc deploy |
| **5-10%** | ⚠️ Slight overfitting | Thêm Dropout, giảm complexity |
| **> 10%** | ❌ Severe overfitting | Cần Regularization, Early stopping, hoặc thêm data |

**Ví dụ:**
```python
# Gap = 2% → OK
Accuracy: 0.92, Val_accuracy: 0.90  ✅

# Gap = 15% → Overfitting!
Accuracy: 0.95, Val_accuracy: 0.80  ❌
```

---

## III.G. Edge Detection - Nền tảng của Computer Vision

### 1. Edge Detection là gì?

**Định nghĩa:** 
Edge (cạnh) là vùng có **sự thay đổi đột ngột về cường độ sáng** (brightness). Edge detection giúp tìm ra contour (đường viền) của vật thể.

**Tại sao quan trọng?**
- CNN học được edges ở layer đầu tiên
- Edges chứa thông tin hình dạng quan trọng
- Nền tảng cho nhận diện vật thể

**Minh họa:**
```
Ảnh gốc:        Edges detected:
┌──────┐        ┌──────┐
│  🏠  │   →    │ ┌──┐ │
│      │        │ │  │ │
└──────┘        └─┴──┴─┘
```

### 2. Các thuật toán Edge Detection

#### A. Sobel Operator (Phát hiện edges theo hướng)

**Nguyên lý:** Sử dụng 2 kernels để tính gradient theo chiều ngang (Gx) và dọc (Gy)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load ảnh
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel X (vertical edges)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

# Sobel Y (horizontal edges)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Combine both
sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original')

axes[0, 1].imshow(sobel_x, cmap='gray')
axes[0, 1].set_title('Sobel X (Vertical edges)')

axes[1, 0].imshow(sobel_y, cmap='gray')
axes[1, 0].set_title('Sobel Y (Horizontal edges)')

axes[1, 1].imshow(sobel_combined, cmap='gray')
axes[1, 1].set_title('Sobel Combined')
plt.show()
```

**Sobel Kernels (Ma trận):**
```
Gx (Vertical edges):     Gy (Horizontal edges):
┌─────────┐             ┌─────────┐
│ -1  0  1│             │  1  2  1│
│ -2  0  2│             │  0  0  0│
│ -1  0  1│             │ -1 -2 -1│
└─────────┘             └─────────┘
```

#### B. Canny Edge Detector (Tốt nhất, phổ biến nhất)

**Ưu điểm:**
- Phát hiện edges mỏng, rõ nét
- Giảm noise tốt
- Kết nối edges thành contours hoàn chỉnh

**5 bước của Canny:**
1. **Gaussian Blur:** Làm mịn ảnh, giảm noise
2. **Gradient Calculation:** Tính intensity gradient (Sobel)
3. **Non-maximum Suppression:** Làm mỏng edges
4. **Double Threshold:** Phân loại strong/weak edges
5. **Edge Tracking:** Nối weak edges với strong edges

```python
# Canny Edge Detection
edges_canny = cv2.Canny(image, 
                        threshold1=50,   # Lower threshold
                        threshold2=150)  # Upper threshold

# Visualize
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(edges_canny, cmap='gray')
plt.title('Canny Edges')
plt.show()
```

**Điều chỉnh thresholds:**
```python
# Low threshold → nhiều edges (có noise)
edges_low = cv2.Canny(image, 30, 100)

# High threshold → ít edges (chỉ edges mạnh)
edges_high = cv2.Canny(image, 100, 200)

# Medium threshold → balanced (khuyến nghị)
edges_medium = cv2.Canny(image, 50, 150)
```

#### C. Contour Detection (Tìm đường viền)

**Contour** là đường cong nối các điểm liên tục có cùng màu/intensity.

```python
# Tìm contours từ edges
contours, hierarchy = cv2.findContours(
    edges_canny, 
    cv2.RETR_EXTERNAL,     # Chỉ lấy contours ngoài
    cv2.CHAIN_APPROX_SIMPLE # Nén contours
)

# Vẽ contours lên ảnh gốc
image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

# Lọc contours theo diện tích
large_contours = [c for c in contours if cv2.contourArea(c) > 500]

# Phân tích contours
for i, contour in enumerate(large_contours):
    # Tính diện tích
    area = cv2.contourArea(contour)
    
    # Tính chu vi
    perimeter = cv2.arcLength(contour, True)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    print(f"Contour {i}: Area={area}, Perimeter={perimeter}")
    cv2.rectangle(image_with_contours, (x, y), (x+w, y+h), (255, 0, 0), 2)

plt.imshow(image_with_contours)
plt.title(f'Found {len(large_contours)} objects')
plt.show()
```

### 3. So sánh các phương pháp

| Phương pháp | Ưu điểm | Nhược điểm | Khi nào dùng |
|-------------|---------|------------|--------------|
| **Sobel** | Nhanh, đơn giản | Nhiều noise, edges dày | Prototype nhanh, detect direction |
| **Canny** | Edges mỏng, chính xác | Chậm hơn, cần tune params | Production, high quality needed |
| **Contours** | Tìm hình dạng hoàn chỉnh | Cần preprocessing tốt | Object detection, counting |

### 4. CNN tự học Edge Detection

**Thú vị:** CNN học edges tự động ở layer đầu!

```python
# Visualize filters của Conv layer đầu tiên
import matplotlib.pyplot as plt

# Lấy weights của layer đầu
first_layer_weights = model.layers[0].get_weights()[0]

# Plot 32 filters (nếu có 32 filters)
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    if i < first_layer_weights.shape[3]:
        # Lấy filter thứ i
        filter_img = first_layer_weights[:, :, 0, i]
        ax.imshow(filter_img, cmap='viridis')
        ax.set_title(f'Filter {i}')
        ax.axis('off')
plt.tight_layout()
plt.show()
```

**Kết quả:** Bạn sẽ thấy các filters giống như Sobel, Canny kernels!

```
Filter 0: Vertical edges
Filter 1: Horizontal edges
Filter 2: Diagonal edges
Filter 3: Corners
...
```

---

## III.H. Zero-Centering và Data Normalization

### 1. Zero-Centering là gì?

**Định nghĩa:** 
Zero-centering (mean subtraction) là kỹ thuật **dịch chuyển dữ liệu** sao cho **trung bình = 0**.

**Công thức:**
$$X_{centered} = X - \text{mean}(X)$$

**Minh họa bằng White Balance trên máy ảnh:**

```
Chưa white balance:         Sau white balance:
Ảnh bị vàng (bias)         Ảnh cân bằng màu

┌────────────┐             ┌────────────┐
│ 🌅 (vàng)  │   →        │ 🌄 (trung tính) │
│ R: 200     │             │ R: 0           │
│ G: 180     │             │ G: -20         │
│ B: 100     │             │ B: -100        │
└────────────┘             └────────────────┘
     ↓                          ↓
Mean = 160             Mean = 0 (centered!)
```

**Giống white balance:**
- **White balance:** Loại bỏ color cast (màu lệch) bằng cách điều chỉnh về điểm trắng chuẩn
- **Zero-centering:** Loại bỏ bias trong data bằng cách dịch về mean = 0

### 2. Tại sao cần Zero-Centering?

#### Lợi ích 1: Tăng tốc Gradient Descent
```
Không centered:           Có centered:
(Weights zigzag)         (Weights đi thẳng)

  w₂                       w₂
   ↑                        ↑
   │   ╱╲╱╲╱╲              │    ╲
   │  ╱      ╲             │     ╲
   │ ╱        ╲            │      ╲
   │╱__________╲→ w₁       │_______╲→ w₁
   
Chậm, không ổn định       Nhanh, ổn định
```

#### Lợi ích 2: Tránh Exploding/Vanishing Gradients
```python
# Ví dụ với dữ liệu không centered
X = [100, 200, 300, 400]  # Mean = 250
# Sau nhiều layer, activations sẽ quá lớn → Exploding!

# Sau zero-centering
X_centered = [-150, -50, 50, 150]  # Mean = 0
# Activations cân bằng hơn
```

### 3. Các kỹ thuật Normalization phổ biến

#### A. Min-Max Scaling (Chuẩn hóa về [0, 1])
```python
# Công thức
X_normalized = (X - X.min()) / (X.max() - X.min())

# Code
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Ví dụ
X = [10, 20, 30, 40]
# → [0, 0.33, 0.67, 1.0]
```
**Khi nào dùng:** Pixel values (ảnh), Neural Networks

#### B. Standardization (Z-score normalization)
```python
# Công thức
X_standardized = (X - mean(X)) / std(X)

# Code
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ví dụ
X = [10, 20, 30, 40]
Mean = 25, Std = 11.18
# → [-1.34, -0.45, 0.45, 1.34]
# Mean = 0, Std = 1
```
**Khi nào dùng:** SVM, Linear Regression, PCA

#### C. Mean Subtraction (Zero-centering thuần túy)
```python
# Công thức
X_centered = X - mean(X)

# Code
X_centered = X - np.mean(X, axis=0)

# Ví dụ
X = [10, 20, 30, 40]
Mean = 25
# → [-15, -5, 5, 15]
# Mean = 0, nhưng Std giữ nguyên
```
**Khi nào dùng:** CNN (thường kết hợp với chia cho 255)

### 4. Best Practice trong Deep Learning

```python
# Cách 1: Min-Max (phổ biến cho ảnh)
training_images = training_images / 255.0
test_images = test_images / 255.0
# Kết quả: [0, 1]

# Cách 2: Standardization (ImageNet preprocessing)
mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
std = np.array([0.229, 0.224, 0.225])   # ImageNet std
training_images = (training_images - mean) / std
# Kết quả: Mean ≈ 0, Std ≈ 1

# Cách 3: Zero-center + Scale
mean = np.mean(training_images, axis=0)
std = np.std(training_images, axis=0)
training_images = (training_images - mean) / std
test_images = (test_images - mean) / std  # Dùng mean/std từ training!
```

**⚠️ LƯU Ý QUAN TRỌNG:**
```python
# ❌ SAI: Normalize train và test riêng
train_normalized = (train - train.mean()) / train.std()
test_normalized = (test - test.mean()) / test.std()

# ✅ ĐÚNG: Dùng mean/std từ training cho test
train_mean = train.mean()
train_std = train.std()
train_normalized = (train - train_mean) / train_std
test_normalized = (test - train_mean) / train_std  # Dùng train_mean/std!
```

### 5. Visualize hiệu quả Zero-Centering

```python
import numpy as np
import matplotlib.pyplot as plt

# Data không centered
X_original = np.array([100, 150, 200, 250, 300])

# Zero-centered
X_centered = X_original - np.mean(X_original)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(X_original, bins=10)
axes[0].axvline(np.mean(X_original), color='r', linestyle='--', 
                label=f'Mean = {np.mean(X_original):.1f}')
axes[0].set_title('Original (Mean ≠ 0)')
axes[0].legend()

axes[1].hist(X_centered, bins=10)
axes[1].axvline(np.mean(X_centered), color='r', linestyle='--', 
                label=f'Mean = {np.mean(X_centered):.1f}')
axes[1].set_title('Zero-Centered (Mean = 0)')
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## III.I. Activation Functions Chi Tiết

### 1. Activation Function là gì?

**Định nghĩa:** 
Hàm kích hoạt (activation function) là hàm **phi tuyến** (non-linear) được áp dụng sau mỗi layer để model có thể học các patterns phức tạp.

**Tại sao cần Activation?**
```python
# Không có activation (chỉ linear)
output = W3 * (W2 * (W1 * X))
       = (W3 * W2 * W1) * X
       = W_combined * X
# → Chỉ là Linear Regression, dù có 100 layers!

# Có activation (non-linear)
output = relu(W3 * relu(W2 * relu(W1 * X)))
# → Có thể học patterns phức tạp!
```

### 2. Các loại Activation Functions

#### A. ReLU (Rectified Linear Unit) - ⭐ Phổ biến nhất

**Công thức:**
$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Code:**
```python
def relu(x):
    return np.maximum(0, x)

# Hoặc trong Keras
tf.keras.layers.Dense(128, activation='relu')
```

**Đồ thị:**
```
f(x)
  │     ╱
  │    ╱
  │   ╱
  │  ╱
──┼─────── x
  │ (x<0 → 0)
```

**Ưu điểm:**
- ✅ Đơn giản, tính toán nhanh
- ✅ Giải quyết vanishing gradient problem
- ✅ Sparse activation (nhiều neurons = 0)

**Nhược điểm:**
- ❌ Dying ReLU: Nếu x < 0, gradient = 0 → neuron "chết"
- ❌ Không centered (output luôn >= 0)

**Khi nào dùng:** Hidden layers của CNN, MLP (DEFAULT CHOICE)

#### B. Leaky ReLU - Khắc phục Dying ReLU

**Công thức:**
$$\text{Leaky ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

Với $\alpha = 0.01$ (thường dùng)

**Code:**
```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Keras
tf.keras.layers.LeakyReLU(alpha=0.01)
# Hoặc
tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
```

**Đồ thị:**
```
f(x)
  │     ╱
  │    ╱
  │   ╱
  │  ╱
──┼─────── x
  │╱ (x<0 → 0.01x)
```

**Khi nào dùng:** Khi gặp dying ReLU problem

#### C. ELU (Exponential Linear Unit)

**Công thức:**
$$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

**Code:**
```python
tf.keras.layers.Dense(128, activation='elu')
```

**Ưu điểm:**
- Mean activation gần 0 (zero-centered)
- Smooth gradient

**Nhược điểm:**
- Tính $e^x$ chậm hơn ReLU

#### D. Sigmoid - Cho output layer (binary classification)

**Công thức:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Đồ thị:**
```
f(x)
1 ┤────────
  │      ╱
0.5│    ╱
  │  ╱
0 ┤────────── x
 -∞    0    +∞
```

**Code:**
```python
tf.keras.layers.Dense(1, activation='sigmoid')
```

**Ưu điểm:**
- Output trong khoảng (0, 1) → xác suất
- Smooth, differentiable

**Nhược điểm:**
- ❌ Vanishing gradient (gradient gần 0 ở 2 đầu)
- ❌ Not zero-centered
- ❌ Chậm

**Khi nào dùng:** **Output layer** của binary classification (0 hoặc 1)

#### E. Tanh (Hyperbolic Tangent)

**Công thức:**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**Đồ thị:**
```
f(x)
1 ┤────────
  │      ╱
0 ┤────╱────
  │  ╱
-1┤────────── x
```

**Code:**
```python
tf.keras.layers.Dense(128, activation='tanh')
```

**Ưu điểm:**
- Zero-centered (output trong [-1, 1])
- Tốt hơn Sigmoid cho hidden layers

**Nhược điểm:**
- Vẫn có vanishing gradient

**Khi nào dùng:** RNN, LSTM (ít dùng cho CNN)

#### F. Softmax - Cho multi-class classification

**Công thức:**
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

**Giải thích chi tiết:**
Biến đổi vector số thành **phân phối xác suất** (tổng = 1)

**Ví dụ cụ thể:**
```python
# Input: Logits từ layer cuối
logits = [2.0, 1.0, 0.1]

# Tính Softmax thủ công
import numpy as np
exp_logits = np.exp(logits)  # [7.39, 2.72, 1.11]
softmax_output = exp_logits / np.sum(exp_logits)
print(softmax_output)
# [0.659, 0.242, 0.099]
# → 65.9% class 0, 24.2% class 1, 9.9% class 2
```

**Tính chất:**
1. Output trong (0, 1)
2. Tổng tất cả outputs = 1
3. Class có logit cao nhất → xác suất cao nhất

**Code:**
```python
tf.keras.layers.Dense(10, activation='softmax')
```

**Khi nào dùng:** **Output layer** của multi-class classification

**Softmax vs Sigmoid:**
```python
# Binary classification (2 classes)
# Cách 1: Sigmoid (1 output neuron)
model.add(Dense(1, activation='sigmoid'))
# Output: [0.8] → 80% class 1, 20% class 0

# Cách 2: Softmax (2 output neurons)
model.add(Dense(2, activation='softmax'))
# Output: [0.2, 0.8] → 20% class 0, 80% class 1

# Multi-class (>2 classes) → PHẢI dùng Softmax
model.add(Dense(10, activation='softmax'))
```

### 3. Bảng tổng hợp Activation Functions

| Activation | Range | Khi nào dùng | Ưu điểm | Nhược điểm |
|-----------|-------|--------------|---------|------------|
| **ReLU** | [0, ∞) | Hidden layers (DEFAULT) | Nhanh, đơn giản | Dying ReLU |
| **Leaky ReLU** | (-∞, ∞) | Hidden layers (nếu ReLU chết) | Fix dying ReLU | Thêm hyperparameter α |
| **ELU** | (-α, ∞) | Hidden layers (cần performance tốt) | Zero-centered | Chậm hơn ReLU |
| **Sigmoid** | (0, 1) | Binary classification output | Xác suất | Vanishing gradient |
| **Tanh** | (-1, 1) | RNN, LSTM | Zero-centered | Vanishing gradient |
| **Softmax** | (0, 1), sum=1 | Multi-class output | Xác suất chuẩn | Chỉ dùng output layer |

### 4. Code so sánh Activations

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# Các activation functions
relu = np.maximum(0, x)
leaky_relu = np.where(x > 0, x, 0.01 * x)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
elu = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))

# Plot
plt.figure(figsize=(15, 4))

plt.subplot(1, 5, 1)
plt.plot(x, relu)
plt.title('ReLU')
plt.grid(True)

plt.subplot(1, 5, 2)
plt.plot(x, leaky_relu)
plt.title('Leaky ReLU')
plt.grid(True)

plt.subplot(1, 5, 3)
plt.plot(x, sigmoid)
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(1, 5, 4)
plt.plot(x, tanh)
plt.title('Tanh')
plt.grid(True)

plt.subplot(1, 5, 5)
plt.plot(x, elu)
plt.title('ELU')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 5. Rule of thumb - Chọn Activation nào?

```
Hidden Layers:
    Start với ReLU → Nếu gặp dying ReLU → Thử Leaky ReLU/ELU
    
Output Layer:
    Binary classification → Sigmoid
    Multi-class classification → Softmax
    Regression (dự đoán số) → Linear (không activation)
```

---

## III.J. Regularization - Chống Overfitting

### 1. Overfitting là gì? (Ôn lại)

**Vấn đề:** Model học thuộc lòng training data thay vì học patterns chung

```
Training accuracy: 98% 📈
Validation accuracy: 75% 📉
→ Overfitting! Model quá phức tạp
```

### 2. L1 Regularization (Lasso)

**Công thức:**
$$\text{Loss} = \text{Loss}_{\text{original}} + \lambda \sum_{i} |w_i|$$

**Cơ chế:**
- Thêm penalty dựa trên **giá trị tuyệt đối** của weights
- Đẩy nhiều weights về **chính xác bằng 0**
- Tạo **sparse model** (nhiều weights = 0)

**Code:**
```python
from tensorflow.keras import regularizers

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l1(0.001)),  # λ = 0.001
    tf.keras.layers.Dense(10, activation='softmax')
])
```

**Khi nào dùng:**
- Muốn **feature selection** (loại bỏ features không quan trọng)
- Model có quá nhiều features
- Cần model nhẹ để deploy

**Ví dụ:**
```python
# Trước L1: weights = [0.5, 0.3, 0.2, 0.1, 0.05]
# Sau L1:  weights = [0.5, 0.3, 0.0, 0.0, 0.0]  ← 3 weights bị "kill"
```

### 3. L2 Regularization (Ridge) - ⭐ Phổ biến nhất

**Công thức:**
$$\text{Loss} = \text{Loss}_{\text{original}} + \lambda \sum_{i} w_i^2$$

**Cơ chế:**
- Thêm penalty dựa trên **bình phương** của weights
- Đẩy weights về **gần 0** (nhưng không bằng 0)
- **Weight decay** - Giảm magnitude của weights

**Code:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01)),  # λ = 0.01
    tf.keras.layers.Dense(10, activation='softmax')
])
```

**Khi nào dùng:**
- **DEFAULT CHOICE** cho regularization
- Model overfitting nhẹ đến trung bình
- Muốn giữ tất cả features nhưng giảm influence

**Ví dụ:**
```python
# Trước L2: weights = [0.5, 0.3, 0.2, 0.1, 0.05]
# Sau L2:  weights = [0.3, 0.2, 0.1, 0.05, 0.02]  ← Tất cả giảm, không ai = 0
```

### 4. Elastic Net (L1 + L2)

**Công thức:**
$$\text{Loss} = \text{Loss}_{\text{original}} + \lambda_1 \sum_{i} |w_i| + \lambda_2 \sum_{i} w_i^2$$

**Code:**
```python
# Keras không có built-in, phải custom
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01)),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

**Khi nào dùng:**
- Muốn **cả feature selection và weight decay**
- Model rất phức tạp

### 5. Dropout - ⭐ Mạnh nhất cho Neural Networks

**Cơ chế:**
Trong mỗi training step, **ngẫu nhiên tắt** (drop) một số neurons

**Minh họa:**
```
Training iteration 1:       Training iteration 2:
┌─────────────┐            ┌─────────────┐
│ ● ● × ● ×  │            │ × ● ● × ●  │
│  ╲│╲ │╱    │            │  ╲│╲ │╱    │
│   ● × ●    │            │   × ● ●    │
└─────────────┘            └─────────────┘
(× = dropped)              (Khác nhau mỗi iteration)
```

**Code:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Bỏ 50% neurons
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Bỏ 30% neurons
    tf.keras.layers.Dense(10, activation='softmax')
])
```

**Tại sao hiệu quả?**
- Model không thể rely on bất kỳ neuron cụ thể nào
- Học features robust hơn
- Giống như **ensemble** nhiều sub-networks

**Khi nào dùng:**
- Overfitting nghiêm trọng
- Fully connected layers (Dense)
- **KHÔNG dùng** cho Conv layers (dùng Batch Normalization thay thế)

**Dropout rate nên chọn bao nhiêu?**
- Small model: 0.2 - 0.3
- Medium model: 0.4 - 0.5
- Large model: 0.5 - 0.7

### 6. So sánh Regularization Techniques

| Kỹ thuật | Cơ chế | Khi nào dùng | Strength |
|----------|--------|--------------|----------|
| **L1 (Lasso)** | Weights → 0 | Feature selection | ⭐⭐ |
| **L2 (Ridge)** | Weights → small | General purpose (DEFAULT) | ⭐⭐⭐ |
| **Elastic Net** | L1 + L2 | Nhiều features tương quan | ⭐⭐ |
| **Dropout** | Randomly drop neurons | Deep networks | ⭐⭐⭐⭐ |

### 7. Code ví dụ với Regularization

```python
# Model không có regularization (Baseline)
model_baseline = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Model với L2 + Dropout
model_regularized = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Train cả 2 models
history_baseline = model_baseline.fit(
    training_images, training_labels,
    validation_data=(test_images, test_labels),
    epochs=20
)

history_regularized = model_regularized.fit(
    training_images, training_labels,
    validation_data=(test_images, test_labels),
    epochs=20
)

# So sánh
print("Baseline - Train acc:", max(history_baseline.history['accuracy']))
print("Baseline - Val acc:", max(history_baseline.history['val_accuracy']))
print("Regularized - Train acc:", max(history_regularized.history['accuracy']))
print("Regularized - Val acc:", max(history_regularized.history['val_accuracy']))

# Kết quả mong đợi:
# Baseline: Train 98%, Val 85% (Overfitting!)
# Regularized: Train 93%, Val 91% (Better generalization!)
```

### 8. Chiến lược chống Overfitting

**Step-by-step approach:**
```python
1. Phát hiện overfitting:
   if (train_acc - val_acc) > 0.1:
       print("Overfitting detected!")

2. Thử giải pháp theo thứ tự:
   a. More data (tốt nhất nhưng tốn kém)
   b. Data augmentation (xoay, flip, zoom ảnh)
   c. Dropout (0.3 - 0.5)
   d. L2 regularization (λ = 0.001 - 0.01)
   e. Early stopping
   f. Giảm model complexity (ít layers/neurons hơn)

3. Monitor val_accuracy:
   - Nếu val_acc tăng → Tiếp tục
   - Nếu val_acc không tăng sau 5 epochs → Dừng
```

---

## III.K. Learning Rate và Khi nào Điều chỉnh

### 1. Learning Rate là gì?

**Định nghĩa:**
Learning rate (LR) là **bước nhảy** khi optimizer cập nhật weights.

**Công thức cập nhật weights:**
$$w_{\text{new}} = w_{\text{old}} - \text{LR} \times \frac{\partial \text{Loss}}{\partial w}$$

**Minh họa:**
```
Loss landscape:

        Loss
         ↑
         │     ╱╲      ← Global minimum
         │    ╱  ╲
         │   ╱    ╲___╱ ← Local minimum
         │  ╱
         │_╱________________→ Weights

LR lớn:   ├─────────────┤ (Nhảy xa, có thể miss minimum)
LR vừa:   ├──────┤       (Cân bằng)
LR nhỏ:   ├─┤            (Nhảy chậm, chính xác nhưng lâu)
```

### 2. Tác động của Learning Rate

#### LR quá lớn (e.g., LR = 0.1)
```
Epoch 1: loss = 2.5
Epoch 2: loss = 3.0  ⚠️ (tăng!)
Epoch 3: loss = 2.8
Epoch 4: loss = 4.5  ⚠️
→ Model không hội tụ, loss nhảy lung tung
```

#### LR quá nhỏ (e.g., LR = 0.00001)
```
Epoch 1: loss = 2.5
Epoch 2: loss = 2.498
Epoch 3: loss = 2.496
Epoch 4: loss = 2.494
...
Epoch 100: loss = 2.3
→ Hội tụ quá chậm, tốn thời gian
```

#### LR vừa phải (e.g., LR = 0.001)
```
Epoch 1: loss = 2.5
Epoch 2: loss = 1.8
Epoch 3: loss = 1.2
Epoch 4: loss = 0.8
→ Hội tụ nhanh và ổn định ✅
```

### 3. Khi nào cần điều chỉnh Learning Rate?

#### Tín hiệu 1: Loss không giảm
```python
# Nếu thấy:
Epoch 5: loss = 1.5
Epoch 10: loss = 1.48
Epoch 15: loss = 1.47
→ LR quá nhỏ! Tăng lên 10x
```

#### Tín hiệu 2: Loss tăng hoặc NaN
```python
# Nếu thấy:
Epoch 1: loss = 2.5
Epoch 2: loss = nan
→ LR quá lớn! Giảm xuống 10x
```

#### Tín hiệu 3: Loss giảm rồi dừng (plateau)
```python
# Nếu thấy:
Epoch 10: val_loss = 0.5
Epoch 15: val_loss = 0.48
Epoch 20: val_loss = 0.48
→ Giảm LR để fine-tune
```

### 4. Kỹ thuật điều chỉnh Learning Rate

#### A. Learning Rate Decay (Giảm dần theo epoch)

```python
# Exponential Decay
initial_lr = 0.001
decay_rate = 0.96
decay_steps = 1000

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

**Minh họa:**
```
LR
  │
0.001├─────╲___
  │          ╲___
0.0005│              ╲___
  │                    ╲___
0.00025│                    ╲___
  │_________________________→ Epochs
  0    10    20    30    40
```

#### B. ReduceLROnPlateau - Callback tự động giảm LR

```python
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',      # Theo dõi val_loss
    factor=0.5,              # Giảm LR xuống 50% (LR_new = LR_old * 0.5)
    patience=5,              # Chờ 5 epochs không cải thiện
    min_lr=1e-7,             # LR thấp nhất
    verbose=1
)

model.fit(
    training_images, training_labels,
    validation_data=(test_images, test_labels),
    epochs=50,
    callbacks=[reduce_lr]
)

# Output:
# Epoch 15: val_loss did not improve, reducing LR to 0.0005
# Epoch 25: val_loss did not improve, reducing LR to 0.00025
```

#### C. Cyclic Learning Rate (CLR)

**Ý tưởng:** LR tăng giảm theo chu kỳ, giúp thoát local minima

```python
# Triangular CLR
def triangular_lr(epoch, base_lr=0.001, max_lr=0.01, step_size=10):
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, (1 - x))

lr_callback = tf.keras.callbacks.LearningRateScheduler(triangular_lr)
```

**Minh họa:**
```
LR
  │   ╱╲      ╱╲      ╱╲
0.01├  ╱  ╲    ╱  ╲    ╱  ╲
  │ ╱    ╲  ╱    ╲  ╱    ╲
0.001├──────╲╱──────╲╱──────╲→ Epochs
  0   5   10  15  20  25  30
```

#### D. Learning Rate Finder (Tìm LR tốt nhất)

```python
# Kỹ thuật từ fastai
import numpy as np
import matplotlib.pyplot as plt

def find_lr(model, X_train, y_train, start_lr=1e-7, end_lr=1, epochs=5):
    num_batches = len(X_train) // 32
    lr_mult = (end_lr / start_lr) ** (1 / num_batches)
    
    lrs = []
    losses = []
    lr = start_lr
    
    for epoch in range(epochs):
        for batch in range(num_batches):
            # Train 1 batch
            tf.keras.backend.set_value(model.optimizer.lr, lr)
            # ... train code ...
            
            lrs.append(lr)
            losses.append(loss)
            lr *= lr_mult
            
            if loss > 4 * min(losses):
                break
    
    # Plot
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()
    
    # Chọn LR tại điểm loss giảm nhanh nhất
    # Thường là 1/10 của LR tại loss thấp nhất

# Sử dụng
find_lr(model, training_images, training_labels)
```

### 5. Best Practices cho Learning Rate

| Optimizer | Default LR | Recommended Range | Notes |
|-----------|-----------|-------------------|-------|
| **SGD** | 0.01 | 0.001 - 0.1 | Cần tune nhiều |
| **Adam** | 0.001 | 0.0001 - 0.01 | Thường không cần tune |
| **RMSprop** | 0.001 | 0.0001 - 0.01 | Tốt cho RNN |
| **Adagrad** | 0.01 | 0.001 - 0.1 | LR tự giảm theo thời gian |

**Quy trình chọn Learning Rate:**
```python
1. Start với default:
   optimizer = tf.keras.optimizers.Adam(lr=0.001)

2. Nếu loss không giảm sau 5 epochs:
   LR *= 10  # Tăng lên 0.01

3. Nếu loss tăng hoặc NaN:
   LR /= 100  # Giảm xuống 0.00001

4. Nếu train OK nhưng muốn tốt hơn:
   Use Learning Rate Scheduler (ReduceLROnPlateau)

5. Fine-tuning cuối:
   LR = 0.0001 (rất nhỏ để tinh chỉnh)
```

### 6. Code ví dụ hoàn chỉnh

```python
# Setup model với LR scheduling
initial_lr = 0.001

# Callback 1: ReduceLROnPlateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Callback 2: Custom LR logger
class LRLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        if hasattr(lr, 'numpy'):
            lr = lr.numpy()
        print(f"\nEpoch {epoch+1}: Learning Rate = {lr:.6f}")

lr_logger = LRLogger()

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with LR scheduling
history = model.fit(
    training_images, training_labels,
    validation_data=(test_images, test_labels),
    epochs=50,
    callbacks=[reduce_lr, lr_logger]
)

# Plot LR changes
lrs = [history.history.get('lr', [initial_lr] * 50)]
plt.plot(lrs)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.yscale('log')
plt.show()
```


