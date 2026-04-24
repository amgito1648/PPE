import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# 1. Configuración de la página
st.set_page_config(
    page_title="Detector de EPP - Seguridad Industrial",
    page_icon="🛡️",
    layout="centered"
)

# Título y descripción
st.title("🛡️ Sistema de Detección de EPP")
st.write("Sube una imagen para verificar el uso de casco, chaleco y otros implementos de seguridad.")

# 2. Cargar el modelo con caché para que sea rápido
@st.cache_resource
def load_model():
    try:
        # Cargamos tu archivo best.pt
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo 'best.pt': {e}")
        return None

model = load_model()

# 3. Sidebar para ajustes
st.sidebar.header("Configuración")
conf_threshold = st.sidebar.slider("Umbral de confianza", 0.0, 1.0, 0.45)

# 4. Cargador de archivos
uploaded_file = st.file_uploader("Cargar imagen de inspección...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Convertir el archivo subido a una imagen de PIL
    image = Image.open(uploaded_file)
    
    # Mostrar la imagen original
    st.subheader("Imagen Original")
    st.image(image, use_container_width=True)
    
    # Botón para procesar
    if st.button("Realizar Detección"):
        with st.spinner('La IA está analizando la imagen...'):
            # Realizar la predicción
            # Convertimos a numpy array para evitar problemas de formato
            img_input = np.array(image)
            results = model.predict(source=img_input, conf=conf_threshold)

            # Obtener la imagen con las cajas dibujadas
            # IMPORTANTE: YOLO entrega esto en formato BGR (estándar de OpenCV)
            res_plotted = results[0].plot()

            # CORRECCIÓN DE COLOR: De BGR a RGB para que no se vea azul
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            # Mostrar resultado
            st.subheader("Resultado de la Detección")
            st.image(res_rgb, caption="Detecciones de la IA", use_container_width=True)

            # 5. Mostrar estadísticas de lo detectado
            st.divider()
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.success(f"Se encontraron {len(boxes)} elementos.")
                
                # Crear una lista de las etiquetas detectadas
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    prob = float(box.conf[0])
                    st.write(f"- **{label}**: {prob:.2%} de confianza")
            else:
                st.warning("No se detectaron elementos de seguridad con el umbral actual.")

elif model is None:
    st.error("El modelo no se pudo cargar. Asegúrate de que el archivo 'best.pt' esté en la misma carpeta que este script.")

# Pie de página
st.markdown("---")
st.caption("Desarrollado con YOLOv8 y Streamlit")