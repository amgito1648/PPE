import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Detector de PPE (Seguridad)", layout="wide")
st.title("🛡️ Sistema de Detección de Equipo de Seguridad")
st.write("Sube una imagen para verificar el cumplimiento de las normas de seguridad.")

# 1. Cargar el modelo
@st.cache_resource # Esto evita que el modelo se recargue en cada clic
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# 2. Subida de archivos
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convertir el archivo a una imagen de PIL
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Detección de IA")
        
        # 3. Realizar la predicción
        results = model.predict(image, conf=0.4) # conf=0.4 es el umbral de confianza
        
        # 4. Dibujar los resultados en la imagen
        # YOLO devuelve los resultados en un formato que se puede plotear directamente
        res_plotted = results[0].plot()
        
        # Mostrar la imagen con las cajitas (bounding boxes)
        st.image(res_plotted, caption='Resultados del Escaneo', use_column_width=True)

    # 5. Mostrar estadísticas en una tabla o lista
    st.divider()
    st.subheader("Detalle de Objetos Detectados")
    boxes = results[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            st.write(f"- ✅ Se detectó: **{label}** con una confianza del **{conf:.2%}**")
    else:
        st.warning("⚠️ No se detectaron elementos de seguridad en la imagen.")