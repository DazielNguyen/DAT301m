# **Slot 1: Course Intro & TensorFlow Basics**

**Môn học:** AI Development with TensorFlow (DAT301m)

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

```

## V. Phân tích kiến thức của bài LAB 01: NEURAL STYLE TRANSFER

---

### 1. Tổng quan về Neural Style Transfer (NST)

**Neural Style Transfer** là kỹ thuật Deep Learning cho phép tạo ra ảnh mới bằng cách kết hợp **nội dung (content)** của một bức ảnh với **phong cách nghệ thuật (style)** của bức ảnh khác.

**Công thức tổng quát:**
```
Generated Image = Content của ảnh A + Style của ảnh B
```

**Ứng dụng thực tế:**
- Chuyển đổi ảnh thành tác phẩm nghệ thuật (Van Gogh, Picasso style)
- Nghệ thuật số (Digital Art)
- Filter cho ứng dụng photo editing
- Video style transfer cho film production

---

### 2. Kiến trúc VGG19 và Transfer Learning

#### 2.1. Tại sao chọn VGG19?

**VGG19** là mạng CNN 19 lớp được huấn luyện trên ImageNet (14 triệu ảnh). Trong NST, VGG19 đóng vai trò là **Feature Extractor** (Bộ trích xuất đặc trưng).

**Ưu điểm:**
- Các lớp sâu capture được đặc trưng semantic cao cấp (hình dạng, cấu trúc vật thể)
- Các lớp nông capture được texture, màu sắc, nét vẽ cơ bản
- Pretrained weights đã học được cách "nhìn" ảnh hiệu quả

**Code implementation:**

```python
from tensorflow.keras.applications import vgg19

# Load VGG19 pretrained trên ImageNet
vgg = vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False  # Freeze weights - không train lại

# Lựa chọn các layers quan trọng
content_layer = 'block5_conv2'  # Đặc trưng semantic cao cấp
style_layers = [
    'block1_conv1',  # Texture cơ bản (edges, colors)
    'block2_conv1',  # Pattern nhỏ
    'block3_conv1',  # Texture phức tạp hơn
    'block4_conv1',  # Cấu trúc tổng thể
    'block5_conv1'   # Trừu tượng cao nhất
]
```

**Minh họa phân cấp đặc trưng:**
```
Block 1: ───────────> Edges, Colors (Low-level)
Block 2: ───────────> Small Patterns
Block 3: ───────────> Complex Textures
Block 4: ───────────> Structural Elements
Block 5: ───────────> High-level Semantics (Content)
```

---

### 3. Image Preprocessing - Tiền xử lý ảnh

#### 3.1. Zero-Centering với ImageNet Mean

**VGG19 được train trên ImageNet với một chuẩn màu cụ thể**. Phải preprocessing theo đúng chuẩn đó:

```python
def preprocess_image(image_path):
    # 1. Load & Resize
    img = load_img(image_path, target_size=(img_height, img_width))
    
    # 2. Convert to array
    img = img_to_array(img)  # Shape: (255, 255, 3)
    
    # 3. Add batch dimension
    img = np.expand_dims(img, axis=0)  # Shape: (1, 255, 255, 3)
    
    # 4. VGG19 preprocessing (Critical!)
    img = vgg19.preprocess_input(img)
    # - Chuyển RGB → BGR
    # - Trừ mean ImageNet: [103.939, 116.779, 123.68]
    
    return tf.convert_to_tensor(img)
```

**Tại sao phải deprocess:**

```python
def deprocess_image(img):
    img = img.reshape((img_height, img_width, 3))
    
    # Đảo ngược zero-centering
    img[:, :, 0] += 103.939  # Red channel
    img[:, :, 1] += 116.779  # Green channel
    img[:, :, 2] += 123.68   # Blue channel
    
    # Đảo ngược BGR → RGB
    img = img[:, :, ::-1]
    
    # Clip về range [0, 255]
    img = np.clip(img, 0, 255).astype('uint8')
    return img
```

---

### 4. Loss Functions - Trái tim của NST

NST tối ưu hóa **3 loss functions** đồng thời:

#### 4.1. Content Loss - Bảo toàn nội dung

**Mục đích:** Đảm bảo ảnh output giữ nguyên cấu trúc vật thể của ảnh gốc.

**Công thức:**
$$L_{content} = \frac{1}{2} \sum_{i,j} (F_{ij}^{content} - F_{ij}^{generated})^2$$

Trong đó:
- $F_{ij}^{content}$: Feature map của ảnh content tại lớp `block5_conv2`
- $F_{ij}^{generated}$: Feature map của ảnh đang tạo

**Code:**

```python
def compute_content_loss(content_features, generated_features):
    # Mean Squared Error giữa content và generated
    return tf.reduce_mean(tf.square(content_features - generated_features))
```

**Ý nghĩa:** Nếu loss cao → Ảnh generated đang "lạc" khỏi nội dung gốc.

---

#### 4.2. Style Loss - Ma trận Gram (Gram Matrix)

**Đây là kỹ thuật QUAN TRỌNG NHẤT của NST!**

**Gram Matrix là gì?**
- Đo lường **correlation** (tương quan) giữa các feature maps
- Capture được "texture" và "pattern" mà không quan tâm vị trí không gian

**Công thức Gram Matrix:**
$$G_{ij} = \sum_{k} F_{ik} \cdot F_{jk}$$

**Code implementation:**

```python
def gram_matrix(input_tensor):
    # input_tensor shape: (batch, height, width, channels)
    
    # Sử dụng Einstein Summation (hiệu quả hơn)
    # 'bijc,bijd->bcd': Nhân từng cặp channels và sum theo spatial dimensions
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    
    # Normalize bởi số lượng pixels
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    
    return result / num_locations
```

**Minh họa Gram Matrix:**
```
Feature Maps (3 channels):        Gram Matrix (3x3):
┌─────┬─────┬─────┐              ┌─────────────┐
│ F1  │ F2  │ F3  │              │ F1·F1 F1·F2 F1·F3 │
│ ... │ ... │ ... │    ──────>   │ F2·F1 F2·F2 F2·F3 │
│(H×W)│(H×W)│(H×W)│              │ F3·F1 F3·F2 F3·F3 │
└─────┴─────┴─────┘              └─────────────┘
                                  (Correlation Matrix)
```

**Style Loss từ nhiều layers:**

```python
def compute_style_loss(style_features, generated_features):
    loss = 0
    for target_style, comb_style in zip(style_features, generated_features):
        # So sánh Gram matrix của style vs generated
        loss += tf.reduce_mean(tf.square(target_style - comb_style))
    
    # Trung bình trên tất cả style layers
    return loss / num_style_layers
```

**Tại sao dùng nhiều layers?**
- `block1_conv1`: Màu sắc, edges cơ bản
- `block2_conv1`: Patterns nhỏ
- `block3_conv1`: Textures phức tạp
- `block4_conv1`: Cấu trúc tổng thể
- `block5_conv1`: Bố cục trừu tượng

---

#### 4.3. Total Variation Loss - Làm mịn ảnh

**Mục đích:** Giảm nhiễu, tạo ảnh mượt mà tự nhiên.

**Công thức:**
$$L_{TV} = \sum_{i,j} \sqrt{(x_{i,j+1} - x_{i,j})^2 + (x_{i+1,j} - x_{i,j})^2}$$

**Code:**

```python
def compute_total_variation_loss(image):
    # Tính gradient theo chiều ngang
    x_deltas = image[:, :, 1:, :] - image[:, :, :-1, :]
    
    # Tính gradient theo chiều dọc
    y_deltas = image[:, 1:, :, :] - image[:, :-1, :, :]
    
    # Tổng absolute values
    return tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))
```

**Minh họa:**
```
Without TV Loss:          With TV Loss:
┌──────────────┐         ┌──────────────┐
│ ▓▒▓▒░▒▓░▒▓▒ │         │ ▓▓▒▒░░▓▓▒▒▓ │
│ ░▓▒░▓▒▓░▓▒░ │   ──>   │ ░░▓▓▒▒▓▓░░▓ │
│ ▓░▒▓░▒▓▒░▓▒ │         │ ▓▓▒▒░░▒▒▓▓▒ │
└──────────────┘         └──────────────┘
   (Noisy)               (Smooth)
```

---

#### 4.4. Tổng hợp Total Loss

**Weighted Sum của 3 losses:**

```python
def compute_total_loss(model, loss_weights, init_image, gram_style_features, content_features):
    content_weight, style_weight, tv_weight = loss_weights
    
    # Forward pass qua VGG19
    model_outputs = model(init_image)
    generated_style_features = model_outputs[:-1]  # 5 style layers
    generated_content_features = model_outputs[-1]  # 1 content layer
    
    # Tính Gram matrix cho style
    generated_gram_features = [gram_matrix(feature) for feature in generated_style_features]
    
    # Tính từng loss
    content_loss = compute_content_loss(content_features, generated_content_features)
    style_loss = compute_style_loss(gram_style_features, generated_gram_features)
    tv_loss = compute_total_variation_loss(init_image)
    
    # Tổng có trọng số
    total_loss = (content_weight * content_loss + 
                  style_weight * style_loss + 
                  tv_weight * tv_loss)
    
    return total_loss, content_loss, style_loss, tv_loss
```

**Điều chỉnh hyperparameters:**

| Parameter | Giá trị thấp | Giá trị cao | Effect |
|-----------|--------------|-------------|--------|
| `content_weight` | Mất nội dung gốc | Giữ nguyên cấu trúc | Balance content preservation |
| `style_weight` | Style yếu | Style đậm nét | Artistic intensity |
| `tv_weight` | Nhiễu, hạt | Quá mịn, mờ | Smoothness control |

---

### 5. Gradient-Based Optimization với tf.GradientTape

**Điểm đặc biệt của NST:** Không train model weights, mà train **chính bức ảnh output**!

#### 5.1. Cơ chế GradientTape

```python
@tf.function  # Compile thành graph để tăng tốc
def compute_grads(model, loss_weights, init_image, gram_style_features, content_features):
    with tf.GradientTape() as tape:
        # GradientTape theo dõi tất cả operations
        all_losses = compute_total_loss(
            model, loss_weights, init_image, 
            gram_style_features, content_features
        )
        total_loss = all_losses[0]
    
    # Tính gradient của loss theo pixels của ảnh
    grads = tape.gradient(total_loss, init_image)
    
    return grads, all_losses
```

**Flow diagram:**
```
┌────────────────┐
│  init_image    │ (Variable - có thể optimize)
│  [255×255×3]   │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│    VGG19       │ (Frozen weights)
│  Feature       │
│  Extraction    │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Loss          │
│  Calculation   │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Gradients     │ ∂L/∂pixel
│  ────────────> │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Update Pixels │ pixel -= learning_rate × gradient
└────────────────┘
```

---

#### 5.2. Training Loop với TensorBoard

**Vòng lặp tối ưu hóa:**

```python
def run_style_transfer(content_path, style_path, num_iterations=1000):
    # Setup TensorBoard logging
    log_dir = "logs/style_transfer/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # Khởi tạo ảnh từ content image
    init_image = preprocess_image(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    
    # Adam optimizer
    opt = tf.optimizers.Adam(learning_rate=5.0)
    
    for i in range(1, num_iterations + 1):
        # 1. Tính gradients
        grads, losses = compute_grads(model, loss_weights, init_image, 
                                       style_targets, content_targets)
        
        # 2. Update pixels
        opt.apply_gradients([(grads, init_image)])
        
        # 3. Clip values (VGG19 expects range)
        init_image.assign(tf.clip_by_value(init_image, -150, 150))
        
        # 4. Log to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('total_loss', losses[0], step=i)
            tf.summary.scalar('content_loss', losses[1], step=i)
            tf.summary.scalar('style_loss', losses[2], step=i)
            tf.summary.scalar('tv_loss', losses[3], step=i)
            
            if i % 100 == 0:
                display_img = deprocess_image(init_image.numpy().copy())
                tf.summary.image('generated_image', np.expand_dims(display_img, 0), step=i)
    
    summary_writer.close()
    return init_image.numpy()
```

---

### 6. Kỹ thuật nâng cao được áp dụng

#### 6.1. Exponential Decay Learning Rate

**Tại sao cần giảm learning rate?**
- Đầu: Learning rate cao → Di chuyển nhanh về vùng tối ưu
- Cuối: Learning rate thấp → Fine-tune chi tiết

```python
# Alternative: Exponential decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=10.0,
    decay_steps=100,
    decay_rate=0.96
)
opt = tf.optimizers.Adam(learning_rate=lr_schedule)
```

---

#### 6.2. Milestone Checkpointing

**Lưu ảnh tại các mốc quan trọng để quan sát tiến trình:**

```python
milestones = [100, 200, 300, 500, 1000, 1500, 2000]
milestone_images = {}

for i in range(1, num_iterations + 1):
    # ... training code ...
    
    if i in milestones:
        milestone_images[i] = deprocess_image(init_image.numpy().copy())
```

**Visualization:**
```python
# So sánh evolution của ảnh
plt.figure(figsize=(18, 6))
for idx, iteration in enumerate([100, 500, 1000, 2000]):
    plt.subplot(1, 4, idx+1)
    plt.imshow(milestone_images[iteration])
    plt.title(f"Iteration {iteration}")
    plt.axis('off')
plt.show()
```

---

#### 6.3. Hyperparameter Experimentation

**Test nhiều style_weight để tìm sweet spot:**

```python
test_style_weights = [1e-15, 1e-10, 1e-5]
test_results = {}

for sw in test_style_weights:
    # Quick 500 iterations test
    loss_weights = (content_weight, sw, total_variation_weight)
    result_image = run_style_transfer(..., iterations=500)
    test_results[sw] = result_image

# Compare side by side
for sw in test_style_weights:
    plt.imshow(test_results[sw])
    plt.title(f"style_weight = {sw}")
```

**Kết quả quan sát:**
- `1e-15`: Hầu như không có style, giữ nguyên content
- `1e-10`: Balance tốt
- `1e-5`: Style quá mạnh, mất content

---

### 7. TensorBoard Integration

**Monitor training real-time:**

```python
# Launch TensorBoard
%load_ext tensorboard
%tensorboard --logdir logs/style_transfer/20260109-105119
```

**Metrics được track:**
1. **Scalar Plots:**
   - Total Loss curve (hội tụ?)
   - Content Loss (content preservation)
   - Style Loss (style strength)
   - TV Loss (smoothness)

2. **Image Timeline:**
   - Ảnh generated mỗi 100 iterations
   - Xem quá trình style transfer diễn ra

**Điều chỉnh dựa trên TensorBoard:**
- Loss không giảm → Tăng learning rate
- Content loss cao → Tăng content_weight
- Ảnh nhiễu → Tăng tv_weight

---

### 8. Thuật ngữ quan trọng (Glossary)

| Thuật ngữ | Giải thích | Ví dụ trong code |
|-----------|-----------|-----------------|
| **Feature Map** | Output của một convolutional layer | `vgg.get_layer('block5_conv2').output` |
| **Gram Matrix** | Ma trận correlation giữa feature channels | `gram_matrix(input_tensor)` |
| **Zero-Centering** | Trừ mean để data center quanh 0 | `img -= [103.939, 116.779, 123.68]` |
| **Semantic Features** | Đặc trưng cao cấp (hình dạng, vật thể) | `block5_conv2` |
| **Texture Features** | Đặc trưng thấp (màu, nét vẽ) | `block1_conv1` |
| **Total Variation** | Độ biến thiên giữa pixels liền kề | `x_deltas + y_deltas` |
| **Gradient Descent** | Thuật toán tối ưu theo hướng giảm loss | `opt.apply_gradients()` |
| **Transfer Learning** | Sử dụng pretrained model | `VGG19(weights='imagenet')` |
| **Content Reconstruction** | Tái tạo cấu trúc content | `compute_content_loss()` |
| **Style Reconstruction** | Tái tạo phong cách style | `compute_style_loss()` |

---

### 9. Tổng kết và Best Practices

#### Quy trình chuẩn NST:
1. Load pretrained VGG19
2. Chọn content layer (block5_conv2) và style layers (block1-5_conv1)
3. Preprocess ảnh theo chuẩn VGG19
4. Tính Gram matrix cho style features
5. Khởi tạo generated image từ content image
6. Optimize pixels bằng gradient descent
7. Monitor với TensorBoard
8. Deprocess về ảnh RGB bình thường

#### Tips để kết quả tốt hơn:
- **Hyperparameters:** Start với `content_weight=1e-8`, `style_weight=1e-6`
- **Iterations:** Minimum 500, tốt nhất 1500-2000
- **Learning rate:** 5.0 với Adam optimizer
- **Image size:** 255×255 là balance giữa quality và speed
- **TV weight:** 1e-6 đến 1e-7 cho ảnh mịn tự nhiên

#### Common Issues:
| Problem | Cause | Solution |
|---------|-------|----------|
| Ảnh nhiễu, hạt | TV weight quá thấp | Tăng `total_variation_weight` |
| Mất content | Style weight quá cao | Giảm `style_weight` hoặc tăng `content_weight` |
| Style yếu | Style weight quá thấp | Tăng `style_weight` |
| Hội tụ chậm | Learning rate thấp | Tăng lên 5.0-10.0 |
| Loss oscillate | Learning rate cao | Giảm hoặc dùng decay |

---

### 10. Tài liệu tham khảo

**Papers:**
- Gatys et al. (2015): "A Neural Algorithm of Artistic Style"
- Simonyan & Zisserman (2014): "Very Deep Convolutional Networks (VGG)"

**TensorFlow Resources:**
- [TensorFlow NST Tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer)
- [Repository Git Hub](https://github.com/nazianafis/Neural-Style-Transfer/blob/main/README.md)
- [VGG19 Documentation](https://keras.io/api/applications/vgg/)

**Visualization Tools:**
- TensorBoard: Real-time monitoring
- Matplotlib: Static visualization

