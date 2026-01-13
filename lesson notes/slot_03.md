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

![Tích chập là gì?](https://github.com/DazielNguyen/DAT301m/blob/main/images/slot_03/01-CNN.png)

> Minh họa về Convolutional Neural Network 

* **Định nghĩa:** Là quá trình thay đổi hình ảnh ban đầu nhằm làm nổi bật (emphasize) các đặc trưng (features) nhất định.
* **Cơ chế hoạt động:** Sử dụng các bộ lọc (filters) để quét qua ảnh:
* Có bộ lọc giúp làm nổi bật các đường nét thẳng đứng (vertical lines).
* Có bộ lọc giúp làm nổi bật các đường nét nằm ngang (horizontal lines).


**B. Gộp (Pooling)**

![Pooling là gì?](https://github.com/DazielNguyen/DAT301m/blob/main/images/slot_03/02-max-pool.png)

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

### 4. Fashion classifier with convolutions

```python
import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0
# Define the model
model = tf.keras.models.Sequential([
    
    # Add convolutions and max pooling
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Add the same layers as before
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
# Print the model summary
model.summary()

# Use same settings
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print(f'\nMODEL TRAINING:')
model.fit(training_images, training_labels, epochs=5, verbose = 1)

# Evaluate on the test set
print(f'\nMODEL EVALUATION:')
test_loss = model.evaluate(test_images, test_labels)
model.evaluate(test_images, test_labels)
```

**Ý nghĩa của các layer:**


**Xử lý dữ liệu**

* `/ 255.0`: Chuẩn hóa giá trị pixel từ [0, 255] về khoảng [0, 1] để tính toán nhanh hơn.

**`Conv2D` (Lớp tích chập)**

* `32`: Số lượng bộ lọc (filters) dùng để dò tìm đặc điểm ảnh.
* `(3,3)`: Kích thước của mỗi bộ lọc (kernel size).
* `activation='relu'`: Hàm loại bỏ giá trị âm (biến thành 0) để giữ lại đặc điểm quan trọng.
* `input_shape=(28, 28, 1)`: Định dạng đầu vào (Cao 28, Rộng 28, 1 kênh màu xám).

**`MaxPooling2D` (Lớp gộp)**

* `(2, 2)`: Kích thước vùng lấy mẫu, giúp giảm kích thước ảnh đi một nửa (downsampling).

**`Flatten` (Lớp duỗi)**

* *(Không tham số)*: Ép ma trận ảnh 2 chiều thành vector 1 chiều để đưa vào mạng nơ-ron phẳng.

**`Dense` (Lớp ẩn - Hidden Layer)**

* `128`: Số lượng nơ-ron để học các tổ hợp đặc điểm phức tạp.
* `activation='relu'`: Tương tự trên, giúp model học phi tuyến tính.

**`Dense` (Lớp đầu ra - Output Layer)**

* `10`: Số nơ-ron tương ứng với 10 nhãn (class) cần phân loại.
* `activation='softmax'`: Chuyển đổi kết quả thành xác suất % (tổng các nơ-ron cộng lại bằng 1).

**`model.compile` (Thiết lập huấn luyện)**

* `optimizer='adam'`: Thuật toán tối ưu hóa tự động điều chỉnh tốc độ học (learning rate).
* `loss='sparse_categorical_crossentropy'`: Hàm tính lỗi dành cho nhãn dạng số nguyên (integer).
* `metrics=['accuracy']`: Hiển thị độ chính xác trong quá trình học.

**`model.fit` (Huấn luyện)**

* `epochs=5`: Số lần model học lặp lại toàn bộ tập dữ liệu.

> Output parameter: 
![Output-01](https://github.com/DazielNguyen/DAT301m/blob/main/images/slot_03/03-output1.png)

> Output đánh giá mô hình: 
![Output-02](https://github.com/DazielNguyen/DAT301m/blob/main/images/slot_03/04-output2.png)

> Neural Network
![05-Output-NN](https://github.com/DazielNguyen/DAT301m/blob/main/images/slot_03/05-Output-NN.png)

> Convolutional Neural Network

![06-Output-CNN](https://github.com/DazielNguyen/DAT301m/blob/main/images/slot_03/06-Output-CNN.png)

### 5. So sánh và Thảo luận (Neural Network vs. CNN)

* Slide đưa ra sự so sánh giữa mạng Nơ-ron truyền thống (Neural Network - chỉ dùng các lớp Dense) và Mạng nơ-ron tích chập (CNN - thêm các lớp Conv2D và MaxPooling2D) trong bài toán phân loại thời trang (Fashion classifier).


* **Vấn đề thảo luận:** Sự khác biệt về hiệu năng và cách xử lý dữ liệu giữa hai kiến trúc này.

**Tóm tắt ngắn gọn (Takeaway):**
Bài học này nâng cấp mô hình Deep Learning cơ bản bằng cách thêm các "đôi mắt" (Convolutions) để lọc đặc trưng và "kính lúp" (Pooling) để cô đọng thông tin trước khi đưa vào phân loại, giúp máy tính "nhìn" và hiểu hình ảnh hiệu quả hơn.

---
## II. The techniques of Convolutional Neural Networks in TensorFlow.

### 1. Vấn đề với dữ liệu thực tế
Khác với bộ dữ liệu chuẩn (như MNIST hay Fashion MNIST), ảnh thực tế thường:
* Có kích thước khác nhau.
* Tỷ lệ khung hình (aspect ratio) khác nhau.
* Chủ thể (Subject) có thể nằm ở bất kỳ đâu trong ảnh, hoặc có nhiều chủ thể.

**Giải pháp:** Sử dụng `ImageDataGenerator` để tiền xử lý và chuẩn hóa dữ liệu tự động.

### 2. ImageDataGenerator
* **Chức năng:** Là một class trong `keras.preprocessing.image` giúp:
    * Tự động load ảnh từ ổ cứng.
    * Gán nhãn (label) tự động dựa trên tên thư mục.
    * Rescale (chuẩn hóa) giá trị pixel (ví dụ: chia cho 255).
    * Augmentation (tăng cường dữ liệu - xoay, lật, zoom...) *[sẽ học sâu hơn ở bài sau]*.

### 3. Cấu trúc thư mục (Directory Structure)
Để `ImageDataGenerator` hoạt động, bạn cần tổ chức thư mục như sau:
```text
Training/
  ├── Horses/  (Chứa ảnh ngựa -> Label tự động là Horses)
  └── Humans/  (Chứa ảnh người -> Label tự động là Humans)
Validation/
  ├── Horses/
  └── Humans/

```

## **Triển khai Code (Implementation)**

### 1. Khởi tạo ImageDataGenerator & Load dữ liệu

Sử dụng `flow_from_directory` để tạo luồng dữ liệu trực tiếp từ thư mục.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Khởi tạo Generator và chuẩn hóa ảnh (Rescale 1./255)
train_datagen = ImageDataGenerator(rescale=1./255)

# 2. Point vào thư mục chứa ảnh train
train_generator = train_datagen.flow_from_directory(
    '/tmp/horse-or-human/',    # Đường dẫn thư mục gốc
    target_size=(300, 300),    # Resize toàn bộ ảnh về kích thước 300x300
    batch_size=128,            # Xử lý theo lô 128 ảnh mỗi lần
    class_mode='binary'        # Chế độ nhị phân (2 class: Horse/Human)
)

# Lưu ý: Nếu có nhiều hơn 2 lớp (VD: Rock/Paper/Scissors), dùng class_mode='categorical'

```

### 2. Xây dựng mô hình CNN (Build Model)

Mô hình Convolutional Neural Network sâu hơn để xử lý ảnh màu phức tạp (300x300x3).

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    # Conv Layer 1: 16 filters, 3x3, Input shape lớn (300,300,3)
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Conv Layer 2: Tăng số filters lên 32
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Conv Layer 3: Tăng lên 64
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Có thể thêm nhiều lớp Conv nữa tùy độ phức tạp...

    # Flatten & Dense Layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    
    # Output Layer: 1 neuron cho bài toán Binary (Sigmoid)
    # Nếu là bài toán nhiều lớp, dùng Softmax
    tf.keras.layers.Dense(1, activation='sigmoid')
])

```

### 3. Compile & Train Model

Sử dụng `binary_crossentropy` cho bài toán 2 lớp và `RMSprop` (thường hiệu quả hơn Adam cho bài toán này).

```python
from tensorflow.keras.optimizers import RMSprop

# 1. Compile
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# 2. Training với Generator
history = model.fit(
    train_generator,      # Dữ liệu từ generator
    steps_per_epoch=8,    # Tổng ảnh / batch_size (VD: 1024 / 128 = 8)
    epochs=15,
    verbose=1
)

```

### 4. Dự đoán (Model Prediction)

Code để upload file ảnh từ máy tính và dự đoán (thường chạy trên Google Colab).

```python
import numpy as np
from google.colab import files
from tensorflow.keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
  # Load ảnh và resize về đúng kích thước input của model (300x300)
  path = '/content/' + fn
  img = image.load_img(path, target_size=(300, 300))
  
  # Chuyển ảnh thành mảng 2D
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0) # Thêm chiều batch (1, 300, 300, 3)

  # Dự đoán
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  
  print(classes[0])
  if classes[0] > 0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")

```

## Ghi chú quan trọng (Key Takeaways)

1. **Automation:** `ImageDataGenerator` là công cụ cực mạnh để giải phóng bạn khỏi việc viết code đọc file thủ công.
2. **Binary vs Categorical:**
* 2 class: Dùng `class_mode='binary'`, Output Dense(1, 'sigmoid'), Loss `binary_crossentropy`.
* 2 class: Dùng `class_mode='categorical'`, Output Dense(n, 'softmax'), Loss `categorical_crossentropy`.

3. **Input Size:** Khi dùng `target_size=(300,300)` trong generator, nó sẽ tự động nén/kéo ảnh về kích thước này. Kích thước càng nhỏ train càng nhanh nhưng có thể mất thông tin chi tiết.


---
## III. Handling Large Datasets & Data Augmentation

Bài này tập trung vào việc xử lý **tập dữ liệu lớn (Large Dataset)** cụ thể là Cats vs Dogs, và đi sâu hơn vào kỹ thuật **Data Augmentation** (Tăng cường dữ liệu) để giải quyết vấn đề Overfitting.


## **Lý thuyết cốt lõi (Core Concepts)**

### 1. Tập dữ liệu Cats vs Dogs
* **Nguồn:** Kaggle Competition. [https://www.kaggle.com/c/dogs-vs-cats/data]
* **Quy mô:** 25.000 hình ảnh chó và mèo.
* **Thách thức:** Với dữ liệu lớn, không thể load toàn bộ vào RAM cùng lúc -> Cần dùng Generator để load từng batch "on-the-fly" (vừa chạy vừa load).

### 2. Data Augmentation (Tăng cường dữ liệu)
Để tránh **Overfitting** (khi mô hình học vẹt trên tập train nhưng dự đoán kém trên tập validation), ta sử dụng kỹ thuật biến đổi ảnh gốc thành nhiều phiên bản khác nhau ngay trong quá trình train.

Các kỹ thuật biến đổi phổ biến trong `ImageDataGenerator`:
* **Rotation:** Xoay ảnh một góc ngẫu nhiên.
* **Width/Height Shift:** Dịch chuyển ảnh theo chiều ngang/dọc.
* **Shear:** Kéo nghiêng ảnh (biến dạng hình học).
* **Zoom:** Phóng to/thu nhỏ ngẫu nhiên.
* **Horizontal Flip:** Lật ảnh theo chiều ngang (như soi gương).



## Triển khai Code (Implementation)

Dưới đây là các đoạn code được trích xuất và tái tạo từ logic trong slide, tập trung vào cấu hình Augmentation và trực quan hóa.

### 1. Cấu hình ImageDataGenerator với Augmentation
Khác với bài trước chỉ rescale, bài này thêm các tham số làm méo ảnh.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Cấu hình Train Generator với Data Augmentation
train_datagen = ImageDataGenerator(
      rescale=1./255,             # Chuẩn hóa pixel
      rotation_range=40,          # Xoay ngẫu nhiên tới 40 độ
      width_shift_range=0.2,      # Dịch ngang 20%
      height_shift_range=0.2,     # Dịch dọc 20%
      shear_range=0.2,            # Kéo nghiêng 20%
      zoom_range=0.2,             # Zoom ngẫu nhiên 20%
      horizontal_flip=True,       # Lật ngang ngẫu nhiên
      fill_mode='nearest'         # Điền khuyết các vùng bị trống sau khi biến đổi
)

# 2. Validation Generator (KHÔNG Augment, chỉ Rescale)
# Lưu ý: Ta không bao giờ làm méo dữ liệu kiểm thử (Validation/Test)
validation_datagen = ImageDataGenerator(rescale=1./255)

# 3. Tạo dòng dữ liệu từ thư mục
train_generator = train_datagen.flow_from_directory(
        'cats_and_dogs_filtered/train',
        target_size=(150, 150),   # Resize về 150x150 (Lớn hơn bài trước) 11]
        batch_size=20,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        'cats_and_dogs_filtered/validation',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

```

### 2. Xây dựng Mô hình (ConvNet)

Mô hình xử lý ảnh đầu vào kích thước 150x150.

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    # Conv 1: 32 filters, 3x3. Input (150, 150, 3) -> Output (148, 148, 32)
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Conv 2: 64 filters
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Conv 3: 128 filters
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Conv 4: 128 filters
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Binary classification
])

# Compile
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
              metrics=['accuracy'])

```

3. Vẽ biểu đồ đánh giá (Visualization) 

Code này rất quan trọng để phát hiện Overfitting (khi đường training đi lên nhưng validation đi ngang hoặc đi xuống).

```python
import matplotlib.pyplot as plt

# Lấy lịch sử huấn luyện
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# Vẽ biểu đồ độ chính xác (Accuracy)
plt.plot(epochs, acc, 'bo', label='Training accuracy') # 'bo' = blue dot
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

# Vẽ biểu đồ mất mát (Loss)
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

```


## Ghi chú quan trọng (Key Takeaways)

1. **Chiến thuật Augmentation:** Là kỹ thuật quan trọng nhất khi làm việc với Computer Vision mà dữ liệu ít hoặc dễ bị Overfit. Nó giúp mô hình "nhìn thấy" nhiều biến thể của ảnh hơn thực tế.
2. **Quy tắc vàng của Validation:** KHÔNG BAO GIỜ áp dụng augmentation lên tập Validation/Test. Chỉ áp dụng `rescale` để đảm bảo đánh giá công bằng.
3. **Kích thước ảnh:** Bài này dùng `(150, 150)` lớn hơn bài MNIST `(28, 28)`. Ảnh càng lớn, mô hình càng cần nhiều lớp Conv và Pooling để nén thông tin (ở đây dùng tới 4 lớp Conv).
4. **Dấu hiệu Overfitting:** Nhìn vào biểu đồ, nếu `Training Accuracy` tăng tiệm cận 1.0 (100%) mà `Validation Accuracy` dậm chân tại chỗ (ví dụ 0.7) thì mô hình đang bị học vẹt -> Cần tăng cường Augmentation hoặc dùng Dropout.

