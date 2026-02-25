# **Slot 11 - Natural Language Processing: Language Models & RNN**

**Môn học:** AI Development with TensorFlow
**Tài liệu gốc:** Slot_11_Language Model + RNN.pdf
**Mục tiêu bài học:**
1. Hiểu khái niệm Mô hình ngôn ngữ (Language Models) và N-Grams.
2. Nắm vững cấu trúc, cơ chế chia sẻ trọng số và phân loại Mạng nơ-ron hồi quy (RNN).
3. Giải quyết bài toán Vanishing/Exploding Gradient bằng BPTT và các kiến trúc nâng cao (LSTM, GRU, Bi-RNN).

---

## I. Mô hình ngôn ngữ (Language Models) & N-Grams

### 1. Language Model (Mô hình ngôn ngữ)
* **Định nghĩa:** Được thiết kế để đo lường phân phối xác suất của các đơn vị ngôn ngữ (từ, chữ).
* **Nhiệm vụ cốt lõi:** Tính xác suất của một từ $w$ khi biết trước các từ trong lịch sử $h$, ký hiệu là $P(w|h)$.
* **Ứng dụng:** Dự đoán từ tiếp theo (Google Search, Autocomplete), Dịch máy (Machine Translation), Sửa lỗi chính tả.

### 2. Mô hình N-Grams
* Là các nhóm gồm $n$ từ liên tiếp nhau.
* **Phân loại phổ biến:** Unigram (1 từ), Bigram (2 từ), Trigram (3 từ).

---

## II. Mạng nơ-ron hồi quy (Recurrent Neural Network - RNN)

### 1. Tại sao không dùng Feed Forward Network (FFN)?
* FFN không có bộ nhớ, không thể xử lý chuỗi dữ liệu (sequential data) mà thứ tự đóng vai trò quan trọng.
* FFN yêu cầu đầu vào và đầu ra phải có kích thước cố định (fixed size).

### 2. Cấu trúc và Cơ chế của RNN

* **Khái niệm:** RNN duyệt qua từng phần tử của chuỗi (ví dụ: từng từ trong câu) và duy trì một **trạng thái ẩn (hidden state - $h_t$)** chứa thông tin về những gì nó đã thấy trong quá khứ.
* **Chia sẻ trọng số (Weight Sharing):** RNN sử dụng chung một bộ trọng số ($W, U, V$) cho tất cả các bước thời gian (time steps). Điều này giúp mô hình giảm lượng tham số và có thể xử lý chuỗi có độ dài bất kỳ.
* **Công thức cốt lõi:**
  Trạng thái ẩn mới được tính dựa trên trạng thái ẩn cũ và đầu vào hiện tại, sử dụng hàm kích hoạt $\tanh$ để ép giá trị về khoảng [-1, 1]:
  $$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{hx} \cdot x_t + b_h)$$

### 3. Các dạng kiến trúc RNN

* **One to Many:** Một đầu vào, nhiều đầu ra (Ví dụ: Image Captioning - Nhìn ảnh sinh ra câu mô tả).
* **Many to One:** Nhiều đầu vào, một đầu ra (Ví dụ: Sentiment Classification - Phân loại cảm xúc câu nói).
* **Many to Many:** * *Dịch pha (Encoder-Decoder):* Dùng cho Machine Translation (Dịch máy).
    * *Đồng bộ (Synchronized):* Dùng cho phân loại video (mỗi frame sinh ra một nhãn).

---

## III. Huấn luyện RNN & Vấn đề Gradient

### 1. Backpropagation Through Time (BPTT)
* BPTT là thuật toán lan truyền ngược qua thời gian để cập nhật trọng số cho mạng RNN.
* Để tính được đạo hàm cho trọng số ở bước thời gian $t$, ta phải dùng *Quy tắc chuỗi (Chain Rule)* nhân các đạo hàm từ bước cuối cùng lùi về $t$.

### 2. Vấn đề Vanishing & Exploding Gradient
* Do việc nhân liên tiếp các ma trận đạo hàm, nếu các giá trị này nhỏ hơn 1, tích sẽ tiến dần về 0 -> **Vanishing Gradient** (Đạo hàm biến mất), mô hình không thể học được các phụ thuộc xa (long-term dependencies).
* Nếu giá trị lớn hơn 1, tích sẽ tiến tới vô cùng -> **Exploding Gradient** (Đạo hàm bùng nổ), làm mô hình mất ổn định.

### 3. Giải pháp khắc phục
* **Với Exploding Gradient:** Dùng kỹ thuật *Gradient Clipping* (cắt gọt gradient nếu nó vượt quá một ngưỡng nhất định).
* **Với Vanishing Gradient:** Sử dụng hàm kích hoạt ReLU, khởi tạo ma trận Identity, hoặc dùng các kiến trúc mạng có "cổng" như **LSTM** và **GRU**.
* **Với chuỗi quá dài:** Dùng *Truncated BPTT* (cắt ngắn quá trình lan truyền ngược thành các đoạn nhỏ).

---

## IV. Kiến trúc RNN nâng cao (LSTM, GRU, Bi-RNN)



### 1. Long Short-Term Memory (LSTM)
* LSTM giải quyết vấn đề Vanishing Gradient bằng cách thêm **Cell state ($C_t$)**, hoạt động như một băng chuyền chạy thẳng xuyên suốt chuỗi, giúp thông tin truyền đi dễ dàng mà không bị biến đổi nhiều.
* **3 cổng (Gates) kiểm soát thông tin:**
    * **Forget Gate ($f_t$):** Quyết định thông tin nào từ quá khứ sẽ bị ném đi (dùng hàm Sigmoid).
    * **Input Gate ($i_t$):** Quyết định thông tin mới nào sẽ được ghi vào Cell state.
    * **Output Gate ($o_t$):** Quyết định phần nào của Cell state sẽ được xuất ra làm Hidden state ($h_t$).

### 2. Gated Recurrent Unit (GRU)
* GRU là biến thể đơn giản hóa của LSTM, có cơ chế tương tự nhưng **ít tham số hơn** (chạy nhanh hơn) và không sử dụng riêng biệt Cell state (gộp chung vào Hidden state).

### 3. Bidirectional RNN (Mạng hồi quy 2 chiều)
* Giúp mạng hiểu ngữ cảnh tốt hơn bằng cách đọc câu theo 2 chiều: Từ trái sang phải ($h^{\rightarrow}$) và từ phải sang trái ($h^{\leftarrow}$).
* Đầu ra tại mỗi bước là sự kết hợp của cả 2 hidden state này.
