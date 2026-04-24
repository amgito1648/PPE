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
    
    if st.button("Realizar Detección"):
        with st.spinner('Analizando...'):
            # 1. Convertir imagen de PIL a formato OpenCV (BGR)
            # Esto es lo que YOLO espera recibir internamente
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # 2. Predecir
            results = model.predict(source=img_cv, conf=conf_threshold)

            # 3. Dibujar resultados
            # res_plotted SIEMPRE saldrá en BGR porque lo dibujó OpenCV
            res_plotted = results[0].plot()

            # 4. LA SOLUCIÓN DEFINITIVA:
            # Forzamos la conversión a RGB para Streamlit
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            # 5. Mostrar resultado
            st.subheader("Resultado de la Detección")
            st.image(res_rgb, caption="¡Ahora sí con colores reales!", use_container_width=True)

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

st.caption("Cristian Cala @Unab")