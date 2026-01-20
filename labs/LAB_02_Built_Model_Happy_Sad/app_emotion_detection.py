"""
Ứng dụng Web Interface cho Phân loại Cảm xúc
Sử dụng Streamlit để tạo giao diện người dùng
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Cấu hình trang
st.set_page_config(
    page_title="Phân loại Cảm xúc",
    page_icon="😊",
    layout="wide"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 16px;
        border-radius: 5px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Tiêu đề
st.title("🎭 Ứng dụng Phân loại Cảm xúc")
st.markdown("**Sinh viên:** Nguyễn Văn Anh Duy | **MSSV:** SE181823 | **Lớp:** AI1803")
st.markdown("---")

# Các lớp cảm xúc (phải khớp với thứ tự trong training)
EMOTION_CLASSES = {
    0: "😠 Tức giận (Anger)",
    1: "😒 Khinh bỉ (Contempt)",
    2: "🤢 Ghê tởm (Disgust)",
    3: "😨 Sợ hãi (Fear)",
    4: "😊 Vui vẻ (Happy)",
    5: "😢 Buồn bã (Sad)",
    6: "😲 Ngạc nhiên (Surprised)"
}

# Mapping tên tiếng Việt
EMOTION_NAMES_VI = {
    'anger': '😠 Tức giận',
    'contempt': '😒 Khinh bỉ',
    'disgust': '🤢 Ghê tởm',
    'fear': '😨 Sợ hãi',
    'happy': '😊 Vui vẻ',
    'sad': '😢 Buồn bã',
    'surprised': '😲 Ngạc nhiên'
}

@st.cache_resource
def load_models():
    """Tải các mô hình đã train"""
    # Tìm file model ở nhiều vị trí
    vgg16_paths = [
        'best_model_vgg16_trained.keras',
        '../best_model_vgg16_trained.keras',
        os.path.join(os.path.dirname(__file__), 'best_model_vgg16_trained.keras'),
        os.path.join(os.path.dirname(__file__), '..', 'best_model_vgg16_trained.keras')
    ]
    
    densenet_paths = [
        'best_model_densenet121.keras',
        '../best_model_densenet121.keras',
        os.path.join(os.path.dirname(__file__), 'best_model_densenet121.keras'),
        os.path.join(os.path.dirname(__file__), '..', 'best_model_densenet121.keras')
    ]
    
    # Tải VGG16
    vgg16_model = None
    for path in vgg16_paths:
        if os.path.exists(path):
            try:
                vgg16_model = load_model(path)
                st.success(f"✅ Đã tải mô hình VGG16 từ: {path}")
                break
            except Exception as e:
                continue
    
    if vgg16_model is None:
        st.warning("⚠️ Không tìm thấy mô hình VGG16. Vui lòng đảm bảo file 'best_model_vgg16_trained.keras' tồn tại.")
    
    # Tải DenseNet121
    densenet_model = None
    for path in densenet_paths:
        if os.path.exists(path):
            try:
                densenet_model = load_model(path)
                st.success(f"✅ Đã tải mô hình DenseNet121 từ: {path}")
                break
            except Exception as e:
                continue
    
    if densenet_model is None:
        st.warning("⚠️ Không tìm thấy mô hình DenseNet121. Vui lòng đảm bảo file 'best_model_densenet121.keras' tồn tại.")
    
    return vgg16_model, densenet_model

def preprocess_image(image, target_size=(224, 224)):
    """Tiền xử lý ảnh để phù hợp với model"""
    # Chuyển về RGB nếu cần
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize ảnh
    image = image.resize(target_size)
    
    # Chuyển thành numpy array
    img_array = np.array(image)
    
    # Chuẩn hóa về [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Thêm batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_emotion(image, model, model_name):
    """Dự đoán cảm xúc từ ảnh"""
    # Tiền xử lý ảnh
    processed_img = preprocess_image(image)
    
    # Dự đoán
    predictions = model.predict(processed_img, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    # Lấy top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [(EMOTION_CLASSES[i], predictions[0][i] * 100) for i in top_3_indices]
    
    return EMOTION_CLASSES[predicted_class], confidence, top_3_predictions

# Sidebar - Chọn nguồn ảnh
st.sidebar.header("⚙️ Cấu hình")

# Tải models
with st.spinner("Đang tải các mô hình..."):
    vgg16_model, densenet_model = load_models()

# Chọn model
st.sidebar.subheader("1️⃣ Chọn mô hình")
model_option = st.sidebar.selectbox(
    "Chọn mô hình để dự đoán:",
    ["VGG16", "DenseNet121"]
)

# Chọn nguồn ảnh
st.sidebar.subheader("2️⃣ Chọn nguồn ảnh")
input_option = st.sidebar.radio(
    "Chọn cách nhập ảnh:",
    ["📤 Upload ảnh", "📸 Chụp ảnh từ webcam", "🖼️ Ảnh mẫu"]
)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📷 Ảnh đầu vào")
    
    uploaded_image = None
    
    # Upload ảnh
    if input_option == "📤 Upload ảnh":
        uploaded_file = st.file_uploader(
            "Chọn ảnh khuôn mặt cần phân loại",
            type=['jpg', 'jpeg', 'png'],
            help="Hỗ trợ định dạng: JPG, JPEG, PNG"
        )
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Ảnh đã upload", use_container_width=True)
    
    # Chụp ảnh từ webcam
    elif input_option == "📸 Chụp ảnh từ webcam":
        camera_photo = st.camera_input("Chụp ảnh khuôn mặt")
        if camera_photo is not None:
            uploaded_image = Image.open(camera_photo)
            st.image(uploaded_image, caption="Ảnh đã chụp", use_container_width=True)
    
    # Ảnh mẫu
    elif input_option == "🖼️ Ảnh mẫu":
        # Tìm thư mục test ở nhiều vị trí
        test_dirs = ['test', '../test', os.path.join(os.path.dirname(__file__), 'test'), 
                     os.path.join(os.path.dirname(__file__), '..', 'test')]
        
        sample_dir = None
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                sample_dir = test_dir
                break
        
        if sample_dir and os.path.exists(sample_dir):
            # Lấy danh sách các thư mục cảm xúc
            emotions = [d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))]
            
            if emotions:
                selected_emotion = st.selectbox("Chọn loại cảm xúc:", emotions)
                emotion_path = os.path.join(sample_dir, selected_emotion)
                
                # Lấy danh sách ảnh
                images = [f for f in os.listdir(emotion_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if images:
                    selected_image = st.selectbox("Chọn ảnh:", images[:10])  # Giới hạn 10 ảnh
                    image_path = os.path.join(emotion_path, selected_image)
                    uploaded_image = Image.open(image_path)
                    st.image(uploaded_image, caption=f"Ảnh mẫu - {selected_emotion}", use_container_width=True)
                else:
                    st.warning("Không tìm thấy ảnh trong thư mục này")
            else:
                st.warning("Không tìm thấy thư mục cảm xúc")
        else:
            st.warning("⚠️ Không tìm thấy thư mục 'test'")

with col2:
    st.header("🎯 Kết quả dự đoán")
    
    if uploaded_image is not None:
        # Chọn model
        selected_model = vgg16_model if model_option == "VGG16" else densenet_model
        
        if selected_model is None:
            st.error(f"❌ Mô hình {model_option} chưa được tải!")
        else:
            # Nút dự đoán
            if st.button("🔍 Phân tích cảm xúc", type="primary"):
                with st.spinner(f"Đang phân tích bằng mô hình {model_option}..."):
                    # Dự đoán
                    emotion, confidence, top_3 = predict_emotion(
                        uploaded_image, 
                        selected_model,
                        model_option
                    )
                    
                    # Hiển thị kết quả chính
                    st.markdown("### 🎭 Cảm xúc được nhận diện:")
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h1 style="text-align: center; margin: 0;">{emotion}</h1>
                        <h3 style="text-align: center; color: #4CAF50; margin: 10px 0;">
                            Độ tin cậy: {confidence:.2f}%
                        </h3>
                        <p style="text-align: center; color: #666;">
                            Mô hình: {model_option}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Hiển thị Top 3 predictions
                    st.markdown("### 📊 Chi tiết dự đoán (Top 3):")
                    
                    for i, (emo, conf) in enumerate(top_3, 1):
                        st.progress(float(conf / 100), text=f"{i}. {emo}: {conf:.2f}%")
                    
                    # Hiển thị thông tin bổ sung
                    with st.expander("ℹ️ Thông tin chi tiết"):
                        st.markdown(f"""
                        - **Mô hình sử dụng:** {model_option}
                        - **Kích thước ảnh đầu vào:** 224x224
                        - **Số lớp cảm xúc:** 7
                        - **Phương pháp:** Transfer Learning
                        """)
    else:
        st.info("👆 Vui lòng chọn hoặc upload ảnh để bắt đầu phân tích")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>📚 LAB 02 - Phân loại Cảm xúc với Transfer Learning</p>
        <p>🎓 Trường Đại học FPT TP.HCM - 2026</p>
    </div>
""", unsafe_allow_html=True)
