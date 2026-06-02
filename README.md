# ❤️ Hệ Thống Hỗ Trợ Chẩn Đoán Nguy Cơ Bệnh Tim Mạch Dựa Trên Học Máy

## 📝 1. Tổng Quan Đề Tài
Dự án tập trung vào việc nghiên cứu và ứng dụng các kỹ thuật học máy có giám sát (`Supervised Learning`) để phân loại, sàng lọc sớm nguy cơ mắc bệnh lý tim mạch dựa trên dữ liệu bảng lâm sàng đa trung tâm. Nhằm tối ưu tính thực tiễn, mô hình AI tốt nhất sau khi huấn luyện đã được đóng gói và tích hợp vào một ứng dụng giao diện Web hỗ trợ các bác sĩ chẩn đoán thời gian thực.

* **Bộ dữ liệu gốc:** Tích hợp đa trung tâm dữ liệu UCI (920 mẫu bệnh án lâm sàng với 13 đặc trưng).
* **Mô hình tối ưu:** Random Forest Classifier kết hợp kỹ thuật sàng lọc siêu tham số chuyên sâu (`GridSearchCV`) và kiểm định chéo phân tầng (`Stratified 5-Fold Cross-Validation`).
* **Hiệu năng:** Mô hình đạt diện tích dưới đường cong kiểm thử độc lập **ROC-AUC = 0.93** và độ nhạy lâm sàng **Recall = 89.2%**.

---

## ⚙️ 2. Kiến Trúc Hệ Thống & Luồng Tiền Xử Lý Dữ Liệu
Quy trình tiền xử lý dữ liệu được thiết kế chặt chẽ nhằm bảo toàn tri thức y tế mạch vành và chống rò rỉ thông tin (`Data Leakage`):
1. **Loại bỏ biến phi lâm sàng:** Triệt tiêu nhiễu từ các cột quản lý hành chính (`id`, `dataset`).
2. **Nhị phân hóa mục tiêu:** Chuyển đổi nhãn phân loại đa lớp nghiêm trọng (`num`) về biến nhị phân `target` (0: Bình thường, 1: Có bệnh).
3. **Mã hóa nhãn nâng cao:** Đồng bộ hóa các biến logic (`fbs`, `exang`) và biến phân loại chữ (`sex`, `cp`, `thal`,...) sang dạng số nguyên nhưng bắt buộc bảo toàn cấu hình ô trống `NaN`.
4. **Phân tách phân tầng (Stratified Split):** Chia tập dữ liệu theo tỷ lệ 80% Huấn luyện / 20% Kiểm thử trước khi áp dụng bất kỳ thuật toán nội suy nào nhằm cô lập tập Test tuyệt đối.
5. **Nội suy đa biến với KNN Imputer:** Tìm kiếm 5 láng giềng gần nhất ($k=5$) dựa trên khoảng cách Nan-Euclidean đa chiều để điền khuyết thiếu cho các chỉ số cận lâm sàng nặng (`ca` thiếu 66.4%, `thal` thiếu 52.8%).
6. **Chuẩn hóa thang đo (Scaling):** Áp dụng công thức Z-Score của `StandardScaler` để đưa các đặc trưng định lượng liên tục (`age`, `chol`, `trestbps`, `oldpeak`,...) về cùng một quy mô hình học.

---

## 📊 3. Kết Quả Thực Nghiệm Các Mô Hình Nền Tảng
Đánh giá song phẳng hiệu năng chẩn đoán của 3 trường phái thuật toán trên tập kiểm thử độc lập ($N=184$):

| Thước đo hiệu năng (Metrics) | KNN (K=5) | Logistic Regression | Random Forest (Baseline) |
| :--- | :---: | :---: | :---: |
| **Accuracy (Chính xác toàn cục)** | 84.2% | 83.2% | **85.9%** |
| **Precision (Độ chuẩn xác)** | 85.4% | 83.2% | **85.8%** |
| **Recall (Độ nhạy lâm sàng)** | 86.3% | 87.3% | **89.2%** |
| **F1-Score (Trung bình điều hòa)** | 85.9% | 85.2% | **87.5%** |

*Nhận xét:* Mô hình cấu trúc học kết hợp `Random Forest` cho kết quả áp đảo tuyệt đối ở mọi khía cạnh, đặc biệt là chỉ số **Recall (89.2%)** giúp hạn chế tối đa rủi ro bỏ sót ca bệnh trong y tế dự phòng.

---

## 💻 4. Hướng Dẫn Cài Đặt và Khởi Chạy Ứng Dụng

### Bước 1: Sao chép mã nguồn (Clone Repository) về máy tính
Mở Terminal/Command Prompt trên máy tính và thực hiện lệnh sao chép kho lưu trữ:
```bash
git clone [https://github.com/dungmy0411-/heart-disease-prediction.git](https://github.com/TEN_TAI_KHOAN_CUA_BAN/heart-disease-prediction.git)
cd heart-disease-prediction
pip install -r requirements.txt
streamlit run heart_app.py
