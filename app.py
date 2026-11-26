# app.py
# Detector de rostros: clasifica Hombre / Mujer con el modelo entrenado en Colab.
# Tiene 3 modos: cámara en vivo, tomar una foto y subir una imagen.
# Para probarlo en tu computadora: streamlit run app.py

import os

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_webrtc import webrtc_streamer

TAM = 224       # tamaño de entrada del modelo (el mismo que se usó al entrenar)
CADA_N = 4      # en la cámara en vivo, se vuelve a clasificar cada N cuadros (para que vaya fluido)
RUTA_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_genero.keras")

# Colores en formato RGB
COLOR_MUJER = (230, 50, 180)   # magenta
COLOR_HOMBRE = (0, 140, 255)   # azul

st.set_page_config(page_title="Detector de género", page_icon="🙂")


# ---------------- Carga del modelo y del detector ----------------
@st.cache_resource
def cargar_modelo():
    # Se carga una sola vez y queda en memoria.
    # compile=False porque aquí solo se usa para predecir, no para entrenar.
    from tensorflow import keras
    return keras.models.load_model(RUTA_MODELO, compile=False)


@st.cache_resource
def cargar_detector():
    # Detector de rostros de OpenCV (Haar Cascade): solo encuentra dónde hay una cara.
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ---------------- Funciones de apoyo ----------------
def mejorar_iluminacion(img_bgr):
    # Corrige contraluz con ecualización adaptativa CLAHE (solo se usa para DETECTAR el rostro).
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def recortar_para_modelo(img_rgb, caja):
    # Recorta el rostro igual que en el entrenamiento (ampliado 35%) y lo deja de 224x224.
    x, y, w, h = caja
    alto, ancho = img_rgb.shape[:2]
    pad_x, pad_y = int(w * 0.35), int(h * 0.35)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(ancho, x + w + pad_x), min(alto, y + h + pad_y)
    return cv2.resize(img_rgb[y1:y2, x1:x2], (TAM, TAM))


def etiqueta_y_confianza(prob_mujer):
    # Convierte la probabilidad de MUJER (0 a 1) en una etiqueta y un porcentaje de seguridad.
    if prob_mujer >= 0.5:
        return "Mujer", prob_mujer * 100, COLOR_MUJER
    return "Hombre", (1 - prob_mujer) * 100, COLOR_HOMBRE


def dibujar_rostro(img, caja, etiqueta, confianza, color):
    # Dibuja el cuadro y el texto sobre la imagen (el color debe venir en el orden de la imagen).
    x, y, w, h = caja
    grosor = max(2, img.shape[1] // 300)
    escala_texto = max(0.6, w / 300)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, grosor)
    posicion = (x, y - 10) if y > 30 else (x + 5, y + 30)
    cv2.putText(img, f"{etiqueta} {confianza:.0f}%", posicion,
                cv2.FONT_HERSHEY_SIMPLEX, escala_texto, color, 2)


# ---------------- Modo 1: cámara en vivo ----------------
def buscar_cercano(cx, cy, w, candidatos):
    # Entre los rostros vistos antes, busca el más cercano (se asume que es la misma persona).
    mejor, mejor_dist = None, 0.75 * w
    for c in candidatos:
        d = ((c["cx"] - cx) ** 2 + (c["cy"] - cy) ** 2) ** 0.5
        if d < mejor_dist:
            mejor, mejor_dist = c, d
    return mejor


def crear_callback(modelo, detector, estado):
    # Devuelve la función que procesa cada cuadro del video.
    # "estado" guarda los rostros de los cuadros anteriores para suavizar el resultado.
    def callback(frame):
        try:
            # Voltear como espejo (igual que en Colab) para que se vea natural
            img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
            estado["n"] += 1
            n = estado["n"]

            # 1) Detectar rostros
            gris = cv2.cvtColor(mejorar_iluminacion(img), cv2.COLOR_BGR2GRAY)
            rostros = detector.detectMultiScale(gris, 1.1, 5, minSize=(90, 90))

            # 2) Relacionar cada rostro con el del cuadro anterior (para conservar su porcentaje)
            disponibles = list(estado["rostros"])
            actuales = []
            for (x, y, w, h) in rostros:
                caja = (int(x), int(y), int(w), int(h))
                cx, cy = caja[0] + caja[2] / 2, caja[1] + caja[3] / 2
                previo = buscar_cercano(cx, cy, caja[2], disponibles)
                if previo is not None:
                    disponibles.remove(previo)
                actuales.append({"caja": caja, "cx": cx, "cy": cy, "n": n,
                                 "prob": previo["prob"] if previo else None})

            # 3) Clasificar: los rostros nuevos de inmediato, y todos cada CADA_N cuadros
            a_clasificar = [r for r in actuales if r["prob"] is None or n % CADA_N == 0]
            if a_clasificar:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                lote = np.array([recortar_para_modelo(img_rgb, r["caja"]) for r in a_clasificar],
                                dtype=np.float32)
                probs = modelo.predict(lote, verbose=0).ravel()
                for r, p in zip(a_clasificar, probs):
                    # Promedio con lo anterior para que la etiqueta no parpadee
                    r["prob"] = float(p) if r["prob"] is None else 0.6 * r["prob"] + 0.4 * float(p)

            # 4) Dibujar (la imagen del video está en BGR, por eso se invierte el color)
            for r in actuales:
                etiqueta, confianza, color = etiqueta_y_confianza(r["prob"])
                dibujar_rostro(img, r["caja"], etiqueta, confianza, color[::-1])

            # 5) Guardar para el siguiente cuadro (se conservan unos cuadros los rostros que se perdieron)
            estado["rostros"] = actuales + [d for d in disponibles if n - d["n"] <= 8]
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print("Error en el video:", e)
            return frame

    return callback


# ---------------- Modos 2 y 3: foto o imagen subida ----------------
def analizar_imagen(img_rgb, modelo, detector, usar_imagen_completa):
    # Recibe una imagen RGB. Devuelve la imagen con los rostros marcados y la lista de resultados.
    alto, ancho = img_rgb.shape[:2]
    if max(alto, ancho) > 1000:   # reducir imágenes muy grandes
        escala = 1000 / max(alto, ancho)
        img_rgb = cv2.resize(img_rgb, (int(ancho * escala), int(alto * escala)))
        alto, ancho = img_rgb.shape[:2]

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gris = cv2.cvtColor(mejorar_iluminacion(img_bgr), cv2.COLOR_BGR2GRAY)
    rostros = detector.detectMultiScale(gris, 1.1, 5, minSize=(60, 60))
    cajas = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rostros]

    # Si no hay rostros y el usuario lo permitió, se analiza la imagen completa
    if len(cajas) == 0 and usar_imagen_completa:
        cajas = [(0, 0, ancho, alto)]

    anotada = img_rgb.copy()
    resultados = []
    if len(cajas) == 0:
        return anotada, resultados

    lote = np.array([recortar_para_modelo(img_rgb, c) for c in cajas], dtype=np.float32)
    probs = modelo.predict(lote, verbose=0).ravel()

    for caja, p in zip(cajas, probs):
        etiqueta, confianza, color = etiqueta_y_confianza(float(p))
        dibujar_rostro(anotada, caja, etiqueta, confianza, color)
        resultados.append((etiqueta, confianza))
    return anotada, resultados


def mostrar_resultado_foto(entrada, modelo, detector, usar_completa):
    # Muestra el análisis de una foto tomada o subida.
    # exif_transpose corrige fotos de celular que vienen rotadas
    imagen = ImageOps.exif_transpose(Image.open(entrada)).convert("RGB")
    with st.spinner("Analizando..."):
        anotada, resultados = analizar_imagen(np.array(imagen), modelo, detector, usar_completa)
    st.image(anotada, caption="Resultado")
    if len(resultados) == 0:
        st.warning("No se detectó ningún rostro. Prueba con más luz, de frente y más cerca de la cámara.")
    else:
        for i, (etiqueta, confianza) in enumerate(resultados, start=1):
            st.write(f"Rostro {i}: **{etiqueta}** ({confianza:.0f}% de seguridad)")


# ---------------- Interfaz de la página ----------------
st.title("🙂 Detector de rostros: Hombre / Mujer")
st.write("El modelo fue entrenado en Google Colab. Elige cómo quieres probarlo.")

if not os.path.exists(RUTA_MODELO):
    st.error("No se encontró el archivo modelo_genero.keras. Súbelo al mismo repositorio que app.py.")
    st.stop()

with st.spinner("Cargando modelo..."):
    modelo = cargar_modelo()
    detector = cargar_detector()

tab_vivo, tab_foto, tab_subir = st.tabs(["🎥 Cámara en vivo", "📸 Tomar foto", "🖼️ Subir imagen"])

with tab_vivo:
    st.write("Presiona **START** y permite el acceso a la cámara. Los rostros se marcan en tiempo real.")
    # Estado propio de esta sesión (guarda los rostros de los cuadros anteriores)
    estado_video = st.session_state.setdefault("estado_video", {"n": 0, "rostros": []})
    webrtc_streamer(
        key="detector-rostros",
        video_frame_callback=crear_callback(modelo, detector, estado_video),
        # Servidor STUN gratuito de Google: necesario para conectar el video cuando la app está en la nube
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )
    st.caption("Si el video no conecta, usa la pestaña 📸 Tomar foto.")

usar_completa = st.checkbox("En fotos: si no detecta un rostro, analizar la imagen completa")

with tab_foto:
    foto = st.camera_input("Toma una foto")
    if foto is not None:
        mostrar_resultado_foto(foto, modelo, detector, usar_completa)

with tab_subir:
    archivo = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])
    if archivo is not None:
        mostrar_resultado_foto(archivo, modelo, detector, usar_completa)
