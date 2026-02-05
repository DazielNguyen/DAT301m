Dataset: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

## Bài 1:

Trong bài học, chúng ta biết Tokenizer tạo ra dictionary từ ngữ. Tuy nhiên, trong thực tế, bộ nhớ không đủ chứa tất cả từ vựng. Chúng ta phải giới hạn num_words. Giả sử chúng ta triển khai model lên thiết bị với bộ nhớ hạn chế, chỉ được phép giữ lại 2,000 từ quan trọng nhất.

Yêu cầu:
1.	Khởi tạo Tokenizer với num_words=2000 và oov_token="<OOV>".
2.	Fit trên toàn bộ sentences.
3.	Câu hỏi phân tích: Hãy tính toán "Tỷ lệ bao phủ thông tin" (Information Coverage Rate).
    - Gợi ý: Duyệt qua tất cả các câu trong sentences. Với mỗi câu, đếm xem có bao nhiêu từ nằm trong Top 2000 (được giữ lại) và bao nhiêu từ bị thay thế bằng <OOV>.
    - Output yêu cầu: Vẽ biểu đồ hoặc in ra % số từ bị mất (trở thành OOV) trên toàn bộ dataset.

## Bài 2:
Ví dụ về sequence cho thấy khi gặp từ lạ (như "manatee" trong slide), Tokenizer mặc định sẽ âm thầm bỏ qua từ đó, khiến câu bị ngắn lại mà không báo lỗi.

Yêu cầu:
1.	Khởi tạo lại một Tokenizer KHÔNG dùng oov_token (để mô phỏng hành vi mặc định trong bài giảng ).
2.	Đặt num_words=10000.
3.	Chuyển sentences sang sequences.
4.	Nhiệm vụ: Viết hàm find_corrupted_sentences(original_texts, sequences) để tìm ra các câu bị thay đổi ý nghĩa nghiêm trọng.
    - Tiêu chí: Một câu bị coi là "hỏng" (corrupted) nếu độ dài của sequence ngắn hơn độ dài số từ trong câu gốc quá 30% (tức là mất >30% lượng từ).
    - In ra 5 ví dụ: Câu gốc vs. Các từ còn lại sau khi tokenize.
5.	Output ví dụ

```
Plaintext
Original: "former versace store clerk sues over secret 'black code' for minority"
Tokenized Words: ['former', 'store', 'clerk', 'sues', 'over', 'secret', 'code']
Status: CORRUPTED (Lost words: 'versace', 'black', 'shoppers')

```

## Bài 3

Slide đã học giới thiệu word_index. Để debug model, ta cần dịch ngược từ số về chữ.

Yêu cầu: Viết hàm decode_sequence(sequence) thực hiện việc sau:
1.	Nhận vào một list các số nguyên (một dòng trong sequences).
2.	Trả về câu văn bản gốc.
3.	Lưu ý: Phải xử lý trường hợp số 0 và các token đặc biệt.
4.	Áp dụng hàm này để giải mã sequence tại index 100 trong dataset.

## Bài 4

Sinh viên không được dùng tensorflow.keras.preprocessing.text.Tokenizer.

Yêu cầu: Viết class MyLiteTokenizer bằng Python thuần:

```
Python
class MyLiteTokenizer:
    def _init__(self, num_words=None) :
        self. num_words = num_words
        self.word_index = {}
        self.word_counts = {} # Đếm tần suất
def fit_on_texts (self, texts) :
        # 1. Chuẩn hóa: Lowercase.
        # 2. Tách từ (Split by space).
        # 3. Loại bỏ dấu câu cơ bản: ! " # $ % & ( ) * + , - . / : ; < = > ? @[ # 4. Đếm tần suất xuất hiện của từng từ.
        # 5. Xây dựng self.word_index chỉ chứa Top 'num_words' từ xuất hiện nhiềt # (Lưu ý: index bắt đầu từ 1, giống Keras [cite: 80]) .
        pass
def texts_to_sequences (self, texts) :
        # 1. Chuyến đối list các câu text thành list các sequence (list of intege 
        # 2. Chỉ map các từ có trong self.word_index.
        # 3. Các từ không có trong index thì bỏ qua (giống hành vi slide 3.2 [cit pass

# Test code
my_tokenizer = MyLiteTokenizer(num_words=100)
my_tokenizer.fit_on_texts (sentences)
my_sequences = my_tokenizer. texts_to_sequences(sentences)
# So sánh kết quả với Keras Tokenizer thật để chấm điểm.

```