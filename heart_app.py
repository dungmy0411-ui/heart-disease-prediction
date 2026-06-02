import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. CẤU HÌNH TRANG WEB
st.set_page_config(
    page_title="Hệ thống Dự đoán Nguy cơ Bệnh tim",
    page_icon="❤️",
    layout="centered"
)

# 2. TẢI MÔ HÌNH ĐÃ HUẤN LUYỆN
@st.cache_resource
def load_model():
    # Hãy đảm bảo file .pkl này nằm cùng thư mục với file app.py
    return joblib.load('heart_disease_random_forest_model.pkl')

try:
    model = load_model()
except:
    st.error("❌ Không tìm thấy file 'heart_disease_random_forest_model.pkl'. Vui lòng đặt file mô hình vào cùng thư mục với file app.py!")
    st.stop()

# 3. GIAO DIỆN CHÍNH
st.title("❤️ Hệ thống Hỗ trợ Chẩn đoán Nguy cơ Bệnh tim")
st.write("Ứng dụng sử dụng mô hình học máy Random Forest để đánh giá xác suất nguy cơ dựa trên các chỉ số sức khỏe lâm sàng và cận lâm sàng.")

st.divider()

# Chia form nhập liệu thành 2 Tabs trực quan
tab1, tab2 = st.tabs(["🩺 1. Triệu chứng & Lâm sàng", "🧪 2. Chỉ số Cận lâm sàng (Xét nghiệm)"])

with tab1:
    st.subheader("Thông tin cơ bản & Triệu chứng ban đầu")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Tuổi của bệnh nhân:", min_value=1, max_value=120, value=50)
        sex_label = st.radio("Giới tính:", ["Nam", "Nữ"])
        sex = 1 if sex_label == "Nam" else 0
    with col2:
        cp_label = st.selectbox(
            "Kiểu đau ngực (cp):",
            [
                "0: Đau thắt ngực điển hình (Typical Angina)",
                "1: Đau thắt ngực không điển hình (Atypical Angina)",
                "2: Đau ngực không do bệnh tim (Non-anginal Pain)",
                "3: Không có triệu chứng đau (Asymptomatic)"
            ]
        )
        cp = int(cp_label.split(":")[0])
        
        exang_label = st.radio("Bị đau ngực khi vận động/tập thể dục? (exang):", ["Không", "Có"])
        exang = 1 if exang_label == "Có" else 0

    st.subheader("Chỉ số sinh lý dễ đo lường")
    thalach = st.slider("Nhịp tim tối đa đạt được (thalach):", min_value=60, max_value=220, value=150)

with tab2:
    st.subheader("Chỉ số Sinh hóa & Huyết động học")
    
    col3, col4 = st.columns(2)
    with col3:
        trestbps = st.number_input("Huyết áp lúc nghỉ ngơi (trestbps) - mmHg:", min_value=80, max_value=200, value=120)
        chol = st.number_input("Chỉ số Cholesterol trong máu (chol) - mg/dl:", min_value=100, max_value=600, value=200)
    with col4:
        fbs_label = st.radio("Đường huyết lúc đói > 120 mg/dl? (fbs):", ["Không", "Có"])
        fbs = 1 if fbs_label == "Có" else 0
        
        restecg_label = st.selectbox(
            "Kết quả điện tâm đồ lúc nghỉ (restecg):",
            ["0: Bình thường", "1: Có sóng ST-T bất thường", "2: Phì đại thất trái theo tiêu chuẩn Estes"]
        )
        restecg = int(restecg_label.split(":")[0])

    st.subheader("Chỉ số chuyên sâu từ Bác sĩ")
    col5, col6 = st.columns(2)
    with col5:
        oldpeak = st.number_input("Độ suy giảm đoạn ST khi vận động (oldpeak):", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope_label = st.selectbox("Độ dốc đoạn ST ở đỉnh điểm vận động (slope):", ["0: Dốc lên", "1: Phẳng", "2: Dốc xuống"])
        slope = int(slope_label.split(":")[0])
    with col6:
        ca = st.slider("Số lượng mạch máu lớn được nhuộm màu huỳnh quang (ca):", min_value=0, max_value=4, value=0)
        thal_label = st.selectbox("Kết quả kiểm tra Thalassemia (thal):", ["0: Bình thường", "1: Khuyết tật cố định", "2: Khuyết tật có thể đảo ngược", "3: Không xác định"])
        thal = int(thal_label.split(":")[0])

st.divider()

# 4. XỬ LÝ DỰ ĐOÁN KHI BẤM NÚT
if st.button("🚀 TIẾN HÀNH CHẨN ĐOÁN NGUY CƠ", type="primary", use_container_width=True):
    
    # Gom tất cả biến thành một DataFrame đúng cấu trúc 13 cột của X_train
    # Thứ tự các cột phải trùng khớp hoàn toàn với dữ liệu bạn đã huấn luyện
    input_data = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalch': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])
    
    # Thực hiện dự đoán xác suất
    pred_proba = model.predict_proba(input_data)[0]
    prob_healthy = pred_proba[0] * 100
    prob_disease = pred_proba[1] * 100
    
    st.subheader("📊 Kết quả phân tích từ Hệ thống:")
    
    # Hiển thị kết quả dựa trên nhãn chiếm tỷ lệ cao hơn
    if prob_disease > prob_healthy:
        st.error(f"⚠️ **CẢNH BÁO:** Bệnh nhân có nguy cơ mắc bệnh tim mạch ở mức **CAO**.")
        st.metric(label="Xác suất nhiễm bệnh", value=f"{prob_disease:.2f}%")
        st.warning("Khuyến nghị: Cần chuyển bệnh nhân thực hiện các chẩn đoán chuyên sâu lâm sàng sớm nhất có thể.")
    else:
        st.success(f"✅ **AN TÂM:** Bệnh nhân nằm trong nhóm có hệ tim mạch **AN TOÀN**.")
        st.metric(label="Xác suất an toàn", value=f"{prob_healthy:.2f}%")
        st.info("Khuyến nghị: Tiếp tục duy trì chế độ ăn uống lành mạnh và tập thể dục đều đặn.")