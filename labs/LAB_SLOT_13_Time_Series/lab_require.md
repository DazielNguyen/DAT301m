Dataset: https://www.kaggle.com/datasets/rakannimer/air-passengers

Thực hiện các công việc sau:

1. Đọc file CSV. Chuyển đổi cột Month sang định dạng datetime và đặt nó làm index (chỉ mục) cho DataFrame.
2. Vẽ biểu đồ đường (Line plot) thể hiện số lượng hành khách theo thời gian.
3. Nhận xét bằng mắt thường: Bạn có thấy xu hướng (trend) không? Có thấy tính mùa vụ (seasonality) không? Biên độ của mùa vụ thay đổi như thế nào theo thời gian?
4. Sử dụng hàm seasonal_decompose (của thư viện statsmodels) để tách chuỗi thành 3 thành phần: Trend, Seasonality và Residuals (nhiễu).
5. Tìm một phương trình miêu tả được sự biến thiên của dữ liệu này.
6. Chia ra Train / Test để thực hiện bài toán dự đoán và Tính toán các chỉ số lỗi như RMSE (Root Mean Squared Error).

### Đánh giá các chỉ số lỗi trên tập Test

| Chỉ số | Holt-Winters | Fourier Regression | Giải thích |
|---|---|---|---|
| **RMSE** | ~26.04 | ~38.03 | Sai số bình phương trung bình gốc — đơn vị giống dữ liệu gốc (nghìn HK), phạt nặng các sai lệch lớn |
| **MAE** | ~21.96 | ~28.83 | Sai số tuyệt đối trung bình — phản ánh độ lệch trung bình thực tế giữa dự đoán và thực tế |
| **MAPE** | ~4.82% | ~6.31% | Sai số phần trăm trung bình — cho phép so sánh giữa các mô hình có đơn vị khác nhau |

**Phân tích kết quả:**

- **Holt-Winters vượt trội ở cả 3 chỉ số** (RMSE, MAE, MAPE) so với Fourier Regression trên tập test. Lý do chính là Holt-Winters sử dụng mô hình nhân (multiplicative seasonality), phù hợp với bản chất của dữ liệu — biên độ mùa vụ tỉ lệ với xu hướng.

- **MAPE ~4.82%** của Holt-Winters là mức rất tốt trong dự báo chuỗi thời gian, cho thấy sai lệch dự đoán trung bình chỉ khoảng 5% so với giá trị thực tế.

- **Fourier Regression với MAPE ~6.31%** vẫn ở mức chấp nhận được, tuy nhiên do bản chất tuyến tính (additive), mô hình này không phản ánh được sự gia tăng biên độ mùa vụ theo thời gian, dẫn đến sai lệch lớn hơn ở những tháng cao điểm của các năm cuối chu kỳ.

**Kết luận:** Với dữ liệu chuỗi thời gian có xu hướng tăng và mùa vụ multiplicative như AirPassengers, **Holt-Winters Triple Exponential Smoothing** là lựa chọn phù hợp và hiệu quả hơn so với hồi quy Fourier đơn giản.