# **Slot 04: Advanced CNN: Overfitting, Augmentation & Transfer Learning**

**Ngày học: 15/01/2026**

**Môn học:** AI Development with TensorFlow

**Tài liệu gốc:**
* 2.2 Image Augmentation.pptx
* 2.3 Overfitting and solutions.pptx
* 2.4 The concept of transfer learning.pptx

**Mục tiêu bài học:**
1.  Hiểu sâu về hiện tượng Overfitting: Nguyên nhân và giải pháp.
2.  Thực hành kỹ thuật Image Augmentation để tăng độ đa dạng dữ liệu.
3.  Áp dụng Transfer Learning (Học chuyển giao) để tận dụng các mô hình đã huấn luyện sẵn.

---

## I. Image Augmentation (Tăng cường dữ liệu ảnh)

> Đây là giải pháp kỹ thuật cụ thể để chống Overfitting trong xử lý ảnh bằng cách tạo ra các biến thể mới từ ảnh gốc.

- **Tăng cường dữ liệu (Augmentation)** là một kỹ thuật được sử dụng trong học máy, đặc biệt là trong bối cảnh huấn luyện mạng nơ-ron sâu, để tránh hiện tượng quá khớp (overfitting).

- **Overfiting xảy ra** khi một mô hình học cách hoạt động tốt trên dữ liệu huấn luyện nhưng không thể khái quát hóa sang dữ liệu mới, chưa từng thấy.

- **Tăng cường dữ liệu** đưa các biến thể vào dữ liệu huấn luyện bằng cách áp dụng các **phép biến đổi ngẫu nhiên, chẳng hạn như xoay, thu phóng, lật hoặc cắt xén**, để tạo ra các ví dụ huấn luyện mới.

- **Ý tưởng chính** đằng sau việc tăng cường dữ liệu là **tăng tính đa dạng của tập dữ liệu huấn luyện** mà không cần thu thập thêm các ví dụ đã được gán nhãn.
- Ví dụ:
    + Trong các bài toán phân loại hình ảnh, việc tăng cường dữ liệu có thể bao gồm việc xoay, thu phóng hoặc lật ngẫu nhiên hình ảnh trong quá trình huấn luyện.
    + Trong các bài toán xử lý ngôn ngữ tự nhiên, các kỹ thuật tăng cường văn bản có thể bao gồm thay thế từ đồng nghĩa, xáo trộn từ hoặc đưa ra những thay đổi nhỏ cho văn bản đầu vào.

### 1. Các kỹ thuật biến đổi
`ImageDataGenerator` cung cấp các tham số để biến đổi ảnh "on-the-fly":
* **rotation_range:** là giá trị tính bằng độ (0–180) để xoay ảnh ngẫu nhiên.
* **width_shift_range / height_shift_range:** là các phạm vi (dưới dạng phân số của tổng chiều rộng hoặc chiều cao) để dịch chuyển ảnh ngẫu nhiên theo chiều dọc hoặc chiều ngang.
* **shear_range:** dùng để áp dụng các phép biến đổi cắt xén ngẫu nhiên và làm méo ảnh (biến dạng hình học).
* **zoom_range:** dùng để phóng to ngẫu nhiên bên trong ảnh.
* **horizontal_flip:** dùng để lật ngẫu nhiên một nửa ảnh theo chiều ngang.
* **fill_mode:** là chiến lược được sử dụng để tô màu các pixel mới được tạo ra, có thể xuất hiện sau khi xoay hoặc dịch chuyển chiều rộng/chiều cao. Cách điền vào các điểm ảnh bị trống sau khi biến đổi (VD: 'nearest').

![ImageDataGenerator](/images/slot_04/01-Different-image-augmentation-techniques-demonstrated-using-keras-ImageDataGenerator.jpg)
> Hình ảnh thông qua các kĩ thuật

### 2. Code triển khai (Keras)

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Cấu hình Augmentation cho tập Train
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Lưu ý: Tập Validation chỉ rescale, KHÔNG Augment
val_datagen = ImageDataGenerator(rescale=1./255)

```



## II. Overfitting & Giải pháp (Overfitting and Solutions)

### 1. Định nghĩa & Nguyên nhân
* **Overfitting là gì?** 
    + Là hiện tượng mô hình dự đoán rất chính xác trên tập dữ liệu huấn luyện (training data) nhưng lại dự đoán kém trên dữ liệu mới (new/unseen data).
    + Một mô hình quá khớp có thể đưa ra dự đoán không chính xác và không thể hoạt động tốt với tất cả các loại dữ liệu mới.

![Overfitting](/images/slot_04/02-overfitting.jpg)

> Mô hình bị overfitting


* **Nguyên nhân:**
    * Dữ liệu huấn luyện quá ít hoặc không đại diện đủ các trường hợp.
    * Dữ liệu chứa nhiều nhiễu (noisy data).
    * Huấn luyện quá lâu (train too long) trên một tập mẫu.
    * Mô hình quá phức tạp (high complexity) nên học thuộc lòng cả nhiễu.

### 2. Cách phát hiện & Ngăn chặn
* **Phát hiện:**
    + Phương pháp tốt nhất để phát hiện các mô hình quá khớp là kiểm tra các mô hình học máy trên nhiều dữ liệu hơn.
    + Kiểm định chéo K-fold: chia tập huấn luyện thành K tập con có kích thước bằng nhau hoặc các tập mẫu được gọi là fold. Quá trình huấn luyện bao gồm một loạt các lần lặp. Trong mỗi lần lặp, các bước là:
        + Giữ lại một tập con làm dữ liệu kiểm định và huấn luyện mô hình học máy trên K-1 tập con còn lại.
        + Quan sát hiệu suất của mô hình trên mẫu kiểm định.
        + Đánh giá hiệu suất của mô hình dựa trên chất lượng dữ liệu đầu ra.


* **Ngăn chặn (Prevention):**
    - **Early stopping (Dừng sớm):** tạm dừng giai đoạn huấn luyện trước khi mô hình máy học xử lý nhiễu trong dữ liệu.
    - **Pruning (Cắt tỉa)**: xác định một số đặc trưng hoặc tham số ảnh hưởng đến dự đoán cuối cùng khi xây dựng mô hình. -> Giảm bớt đặc trưng (Remove features)
    - **Regularization (Điều chỉnh)**: tập hợp các kỹ thuật huấn luyện/tối ưu hóa nhằm giảm thiểu hiện tượng quá khớp. Các phương pháp này cố gắng loại bỏ những yếu tố không ảnh hưởng đến kết quả dự đoán bằng cách xếp hạng các đặc trưng dựa trên tầm quan trọng.
    - **Ensembling (Kết hợp mô hình)**: kết hợp các dự đoán từ nhiều thuật toán học máy riêng biệt. Một số mô hình được gọi là mô hình học yếu vì kết quả của chúng thường không chính xác. Phương pháp kết hợp mô hình kết hợp tất cả các mô hình học yếu để có được kết quả chính xác hơn. Chúng sử dụng nhiều mô hình để phân tích dữ liệu mẫu và chọn ra kết quả chính xác nhất.
    - **Data augmentation (Tăng cường dữ liệu)**: một kỹ thuật học máy thay đổi dữ liệu mẫu một chút mỗi khi mô hình xử lý nó.

### 3. Tham số huấn luyện quan trọng

- **batch_size:** Số lượng mẫu dữ liệu được xử lý trong một lần cập nhật trọng số.
    - **Batch Nhỏ:** Tốn ít RAM, hội tụ nhanh nhưng dao động nhiều (noise).
    - **Batch Lớn:** Hội tụ mượt hơn nhưng tốn RAM.
    - Nó biểu thị số lượng **samples** (mẫu) được xử lý trong một lần lặp.
    - Trong quá trình huấn luyện, toàn bộ tập dữ liệu được chia thành các lô (batch), và mỗi Batch được xử lý lần lượt từng mẫu một.
    - Trọng số của mô hình được cập nhật sau khi xử lý mỗi batch.
    - Kích thước **batch nhỏ** hơn thường được sử dụng cho thuật toán giảm độ dốc ngẫu nhiên (SGD), trong khi kích thước **batch lớn** hơn được sử dụng cho thuật toán giảm độ dốc theo Batch (Batch Gradient Descent).
    - Kích thước **batch nhỏ** hơn có thể gây nhiễu trong quá trình cập nhật trọng số nhưng có thể hội tụ nhanh hơn, trong khi kích thước **batch lớn** hơn có thể hội tụ mượt mà hơn nhưng yêu cầu nhiều bộ nhớ hơn.


* **steps_per_epoch:** Số lượng batch cần xử lý để coi là xong 1 epoch. Thường bằng `Tổng số ảnh / batch_size`.
    - Tham số này thể hiện số lượng Batch dữ liệu cần xử lý trước khi chuyển sang kỷ nguyên tiếp theo.
    - Tham số này hữu ích khi bạn có một tập dữ liệu lớn và muốn chỉ định số lượng Batch dữ liệu cần được xử lý trước khi coi một kỷ nguyên là hoàn tất.
    - Nếu không được chỉ định, hành vi mặc định là xử lý toàn bộ tập dữ liệu trong một kỷ nguyên.

![batch_sizes & step_per-epoch](/images/slot_04/03-epoch-in-machine-learning_.webp)
> Minh họa batch_sizes & step_per-epoch
---

## III. Transfer Learning (Học chuyển giao)

### 1. Khái niệm

* **Transfer Learning** 
    - Là kỹ thuật sử dụng một mô hình đã được huấn luyện cho một tác vụ (Source Task) làm điểm khởi đầu cho một tác vụ khác (Target Task).
    - Kỹ thuật này hữu ích khi nhiệm vụ thứ hai tương tự như nhiệm vụ đầu tiên, hoặc khi dữ liệu có sẵn cho nhiệm vụ thứ hai bị hạn chế.
    - **Transfer Learning**  là một phương pháp tối ưu hóa cho phép tiến bộ nhanh chóng hoặc cải thiện hiệu suất khi mô hình hóa nhiệm vụ thứ hai.
    - **Transfer Learning**  có liên quan đến các vấn đề như học đa nhiệm và sự thay đổi khái niệm, và không chỉ là lĩnh vực nghiên cứu riêng của học sâu.

![Transfer Learning](/images/slot_04/04-transfer-learning-16.jpg)
> The concepts of transfer learning

* **Tại sao cần dùng?**

* Các lớp đầu của mạng nơ-ron thường học các đặc trưng cơ bản (cạnh, màu sắc, hình khối) giống nhau ở mọi bài toán ảnh. Ta có thể tái sử dụng chúng.
* Giải quyết vấn đề thiếu dữ liệu hoặc cần train nhanh.

### 2. Quy trình thực hiện (Develop Model Approach)

1. Chọn **Source Task (Nhiệm vụ Nguồn)**: Chọn một bài toán mô hình dự đoán có liên quan với lượng dữ liệu dồi dào, trong đó có mối quan hệ nào đó giữa dữ liệu đầu vào, dữ liệu đầu ra và/hoặc các khái niệm được học trong quá trình ánh xạ từ dữ liệu đầu vào sang dữ liệu đầu ra.
2. **Develop Source Model (Phát triển Mô hình Nguồn)**: Phát triển một mô hình hiệu quả cho nhiệm vụ đầu tiên này. Mô hình phải tốt hơn một mô hình đơn giản để đảm bảo rằng việc học đặc trưng đã được thực hiện.
3. **Reuse Model (Tái sử dụng mô hình)**: Mô hình được xây dựng trên nhiệm vụ nguồn có thể được sử dụng làm điểm khởi đầu cho mô hình trên nhiệm vụ thứ hai cần quan tâm. Điều này có thể bao gồm việc sử dụng toàn bộ hoặc một phần mô hình, tùy thuộc vào kỹ thuật mô hình hóa được sử dụng.
4. **Tune Model (Tinh chỉnh mô hình)**: Tùy chọn, mô hình có thể cần được điều chỉnh hoặc tinh chỉnh dựa trên dữ liệu cặp đầu vào-đầu ra có sẵn cho nhiệm vụ cần quan tâm.


### 3. Quy trình thực hiện (Pre-trained Model Approach)

1. **Select Source Model:** Chọn mô hình đã train sẵn (VD: Inception, MobileNet, VGG...). Đây gọi là **Base Model**.
2. **Reuse Model (Transfer Layers):**
* Lấy các lớp Convolution (thường đóng băng/freeze các lớp này để không train lại).
* Loại bỏ lớp Output cũ (vì bài toán cũ có thể là phân loại 1000 lớp, còn bài mới chỉ cần 2 lớp Chó/Mèo).


3. **Tune Model (Fine-tuning):**
* Thêm các lớp mới (Dense layers) phù hợp với bài toán mới.
* Huấn luyện lại (Retrain) các lớp mới này với dữ liệu mới.



### 3. Ưu và Nhược điểm

* **Ưu điểm:** Tăng tốc độ huấn luyện, hiệu năng tốt hơn, xử lý được tập dữ liệu nhỏ.
* **Nhược điểm:** Có thể gặp vấn đề lệch miền dữ liệu (Domain mismatch) hoặc mô hình quá phức tạp so với bài toán.

### 4. Code minh họa (Logic Transfer Learning)

```python
# 1. Tải Base Model (VD: InceptionV3) đã train trên ImageNet
# include_top=False nghĩa là bỏ lớp Classification cuối cùng đi
base_model = tf.keras.applications.InceptionV3(input_shape=(150, 150, 3),
                                               include_top=False,
                                               weights='imagenet')

# 2. Đóng băng các lớp của Base Model (Không train lại)
for layer in base_model.layers:
    layer.trainable = False

# 3. Thêm các lớp mới
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x) # Thêm Dropout để giảm Overfitting
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# 4. Tạo mô hình hoàn chỉnh
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# 5. Compile và Train như bình thường
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

```

### 5. Tóm tắt chung về cách thức hoạt động của Transfer Learning:

- **Pre-trained Model (Mô hình được huấn luyện trước)**: Bắt đầu với một mô hình đã được huấn luyện trước đó cho một nhiệm vụ nhất định bằng cách sử dụng một tập dữ liệu lớn. Thường được huấn luyện trên các tập dữ liệu rộng lớn, mô hình này đã xác định được các đặc điểm và mẫu chung liên quan đến nhiều công việc tương tự.

- **Base Model (Mô hình cơ sở)**: Mô hình đã được huấn luyện trước được gọi là mô hình cơ sở. Nó được tạo thành từ các lớp đã sử dụng dữ liệu đầu vào để học các biểu diễn đặc trưng phân cấp.

- **Transfer Layers (Lớp chuyển giao)**: Trong mô hình được huấn luyện trước, hãy tìm một tập hợp các lớp nắm bắt thông tin chung liên quan đến nhiệm vụ mới cũng như nhiệm vụ trước đó. Vì chúng có khả năng học thông tin cấp thấp, các lớp này thường được tìm thấy gần đầu mạng.

- **Fine-Tuning (Tinh chỉnh)**: Sử dụng tập dữ liệu từ thử thách mới để huấn luyện lại các lớp đã chọn. Chúng ta định nghĩa quy trình này là tinh chỉnh. Mục tiêu là bảo toàn kiến ​​thức từ quá trình huấn luyện trước trong khi cho phép mô hình sửa đổi các tham số của nó để phù hợp hơn với yêu cầu của nhiệm vụ hiện tại.

**Advantages of transfer learning:**
- Tăng tốc quá trình huấn luyện.
- Hiệu suất tốt hơn.
- Xử lý được các tập dữ liệu nhỏ.

**Disadvantages of transfer learning:**

- Không phù hợp với miền dữ liệu.
- Quá khớp (Overfitting).
- Độ phức tạp cao.

![Tranditional ML & Transfer Learning](/images/slot_04/05.jpg)