## Yêu cầu 1: 

- Sử dụng các chiến lược chia tách khác nhau để huấn luyện mô hình. (Có thế sử dụng tất cả các mô hình nào có sẵn.)
    + Chiến lược 1: **Fixed or Rolling Window:** Decide whether to use a fixed window approach (e.g., the first 80% for training, the next 10% for validation, and the last 10% for testing) or a rolling window approach (e.g., using a moving time window for training, validation, and testing).
    + Chiến lược 2: **Multiple Train-Validation-Test Splits:** In cases where the dataset is large, multiple train-validation-test splits may be performed for robust evaluation.
    + Chiến lược 3: **Cross-Validation:** Traditional k-fold cross-validation might not be directly applicable to time series data due to the temporal aspect. However, techniques like time series cross-validation (e.g., walk-forward validation) can be employed.
    
## Yêu cầu 2: 

- Sử dụng các tiêu chí đánh giá khác nhau. 
    + **The Errors:** The difference between the forecasted values from our model and the actual values over the evaluation period.
    + **The Mean Squared Error:** Square the errors and then calculate their mean. Measures the average squared difference between predicted and actual values. 
    + **The Root Mean Squared Error:** If we want the mean of our errors' calculation to be of the same scale as the original errors, then we just get its square root, giving us a root means squared error or rmse, The square root of MSE; provides an interpretable measure in the same units as the target variable.
    + **The Mean Absolute Error:** Measures the average absolute difference between predicted and actual values.
    + **The Mean Absolute Percentage Error:** Expresses errors as a percentage of the actual values, providing a relative measure of accuracy.
    + **The Forecast Bias:** Measures the average tendency of the forecast to be too high or too low.
    + **The Percentage Forecast Accuracy:** Measures the overall accuracy of the forecast as a percentage.
    