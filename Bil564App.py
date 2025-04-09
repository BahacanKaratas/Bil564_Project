import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Sayfa ayarları
st.set_page_config(page_title="Akciğer Kanseri Tahmini", layout="centered")
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Akciğer Kanseri Sınıflandırıcı</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Yüklediğiniz akciğer görüntüsünü analiz ederek hem <b>Normal CNN</b> hem de <b>GAN destekli CNN</b> modellerinin tahmin sonuçlarını karşılaştırın.</p>", unsafe_allow_html=True)

# Model yolları
NORMAL_MODEL_PATH = "normal_model_best.h5"
GAN_MODEL_PATH = "gan_model_best.h5"
NORMAL_THRESHOLD_PATH = "normal_threshold.txt"
GAN_THRESHOLD_PATH = "gan_threshold.txt"

# Modelleri yükle
@st.cache_resource
def load_models():
    try:
        normal_model = tf.keras.models.load_model(NORMAL_MODEL_PATH)
        gan_model = tf.keras.models.load_model(GAN_MODEL_PATH)
        return normal_model, gan_model
    except Exception as e:
        st.error(f"Model yüklenemedi: {str(e)}")
        return None, None

# Eşik değerlerini yükle
def load_threshold(path, default=0.5):
    try:
        with open(path, "r") as f:
            return float(f.read().strip())
    except:
        return default

# Model ve eşikleri getir
normal_model, gan_model = load_models()
normal_threshold = load_threshold(NORMAL_THRESHOLD_PATH)
gan_threshold = load_threshold(GAN_THRESHOLD_PATH)

# Eğer model yüklenemezse dur
if normal_model is None or gan_model is None:
    st.stop()

# Görsel yükleme
uploaded_file = st.file_uploader("Bir akciğer görüntüsü yükleyin (JPG veya PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((150, 150))
    st.image(image, caption="Yüklenen Görüntü", use_column_width=False)

    # Görseli işleme
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 150, 150, 3)

    # Tahminler
    pred_normal_prob = float(normal_model.predict(img_array, verbose=0)[0][0])
    pred_gan_prob = float(gan_model.predict(img_array, verbose=0)[0][0])

    # Yorumlama
    def interpret(prob, threshold):
        if prob >= threshold:
            return '<span style="color: red; font-weight: bold; font-size: 20px;">CANCEROUS</span>'
        else:
            return '<span style="color: green; font-weight: bold; font-size: 20px;">NORMAL</span>'

    # Tahmin sonuçları
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #34495E;'>Tahmin Sonuçları</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>Normal CNN</h3>", unsafe_allow_html=True)
        st.markdown(interpret(pred_normal_prob, normal_threshold), unsafe_allow_html=True)
        st.markdown(f"Güven: <code>{pred_normal_prob:.2%}</code>", unsafe_allow_html=True)
        st.caption(f"Eşik Değeri: {normal_threshold:.2f}")

    with col2:
        st.markdown("<h3>GAN Destekli CNN</h3>", unsafe_allow_html=True)
        st.markdown(interpret(pred_gan_prob, gan_threshold), unsafe_allow_html=True)
        st.markdown(f"Güven: <code>{pred_gan_prob:.2%}</code>", unsafe_allow_html=True)
        st.caption(f"Eşik Değeri: {gan_threshold:.2f}")

else:
    st.warning("Lütfen bir görüntü dosyası yükleyin.")
