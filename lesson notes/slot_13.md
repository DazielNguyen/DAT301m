# **Slot 13 - The introduction of Time Series and It's application**
**Ngày học:** 02-03-2026

**Tài liệu gốc:** 

- 4.1 The introduction of Time series and its application.ppt

## Chủ đề **Phân tích dữ liệu chuỗi thời gian (Time Series Analysis)**.

**1. Khái niệm và Đặc trưng của Chuỗi thời gian**

* **Định nghĩa:** Dữ liệu chuỗi thời gian là tập hợp các quan sát được ghi nhận hoặc thu thập liên tục theo các mốc thời gian cố định (ví dụ: dữ liệu cảm biến đo nhiệt độ từ 9h đến 10h, mỗi 10 phút một lần).
* **Thứ tự thời gian (Temporal Ordering):** Các điểm dữ liệu bắt buộc phải tuân theo một trình tự thời gian nhất định; nếu cắt ngắt hoặc làm xáo trộn trình tự này, dữ liệu sẽ mất đi ý nghĩa.
* **Sự phụ thuộc vào thời gian (Dependency on Time):** Giá trị của dữ liệu ở thời điểm hiện tại thường phụ thuộc chặt chẽ vào các giá trị ở quá khứ (ví dụ: giá bán hôm nay bị ảnh hưởng bởi giá ngày hôm qua).

**2. Phân loại và Xử lý dữ liệu**

* Dữ liệu được chia thành hai loại: Đều đặn (Regular - ví dụ: ghi nhận đều đặn hàng giờ, hàng ngày) và Không đều đặn (Irregular).
* Khi xây dựng các mô hình Học máy (Machine Learning), hệ thống thường ưu tiên xử lý các tập dữ liệu mang tính đều đặn (Regular).

**3. Các thành phần phân tích (Decomposition)**
Khi phân tích chuỗi thời gian, ta thường bóc tách dữ liệu thành các thành phần:

* **Xu hướng (Trend):** Thể hiện hướng đi lên hoặc đi xuống của dữ liệu trong một thời gian dài.
* **Tính mùa vụ (Seasonality):** Biến động có tính chất lặp đi lặp lại theo chu kỳ (ví dụ: doanh số bán kem tăng mạnh vào mùa hè).
* **Tự tương quan (Auto-correlation):** Đo lường mức độ tương quan giữa dữ liệu trong quá khứ và sự kiện ở hiện tại/tương lai.

**4. Phương pháp Dự báo (Forecasting)**

* Các phương pháp thống kê truyền thống thường không đủ sức để giải quyết những tập dữ liệu quá lớn và phức tạp.
* Để giải quyết, các mô hình **Deep Learning** được khuyên dùng, đặc biệt tiêu biểu là mạng **LSTM (Long Short-Term Memory)** – một mô hình rất mạnh mẽ trong việc ghi nhớ và dự đoán chuỗi thời gian.

**5. Ứng dụng thực tế**
Phân tích chuỗi thời gian được ứng dụng rộng rãi trong nhiều ngành:

* **Tài chính & Kinh tế:** Dự đoán giá cổ phiếu, chứng khoán hoặc biến động GDP.
* **Năng lượng & Thời tiết:** Dự báo lượng điện năng tiêu thụ, dự báo thời tiết.
* **Viễn thông:** Phân tích lưu lượng mạng (network traffic) để dự đoán và phân bổ băng thông (bandwidth) nhằm chống nghẽn mạng.

Bên cạnh phần lý thuyết, giảng viên cũng gợi ý sinh viên có thể chủ động tìm các bộ dữ liệu (dataset) như dữ liệu nhiệt độ trên Kaggle để tiến hành kiểm tra (test) và thực hành xây dựng mô hình dự báo.