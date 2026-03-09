import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import io
import pandas as pd

st.set_page_config(page_title="Validador de fotos docentes", layout="wide")

SEM_STATUS = {
    "aprobada": "🟢 Aprobada",
    "observada": "🟡 Observada",
    "rechazada": "🔴 Rechazada",
}

# -----------------------------
# Configuración
# -----------------------------
MIN_WIDTH = 800
MIN_HEIGHT = 800
MIN_BRIGHTNESS = 70
MAX_BRIGHTNESS = 210
MIN_SHARPNESS = 80
CENTER_TOLERANCE = 0.12  # tolerancia para centro del rostro
FACE_MIN_RATIO = 0.12    # cara demasiado chica
FACE_MAX_RATIO = 0.38    # cara demasiado grande
BORDER_STD_MAX = 28      # menor = fondo más uniforme

# Cargamos clasificador Haar de OpenCV
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def pil_to_cv(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def image_format_ok(uploaded_file):
    if uploaded_file is None:
        return False, "No se cargó ningún archivo"
    valid_types = ["image/jpeg", "image/png"]
    if uploaded_file.type not in valid_types:
        return False, f"Formato no válido: {uploaded_file.type}. Debe ser JPG o PNG"
    return True, "Formato válido"


def is_square(width, height, tolerance=0.03):
    ratio = width / height
    return abs(ratio - 1.0) <= tolerance


def analyze_brightness(gray):
    return float(np.mean(gray))


def analyze_sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_face(gray):
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    # elegimos el rostro más grande
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return faces[0]


def face_centering(face, width, height):
    x, y, w, h = face
    cx = x + w / 2
    cy = y + h / 2
    dx = abs(cx / width - 0.5)
    dy = abs(cy / height - 0.38)  # el rostro idealmente algo arriba del centro
    return dx, dy


def face_ratio(face, width, height):
    _, _, w, h = face
    return max(w / width, h / height)


def background_uniformity(rgb_img):
    # evaluamos bordes de la imagen, donde idealmente debería verse fondo
    arr = np.array(rgb_img.convert("RGB"))
    h, w, _ = arr.shape
    bw = max(10, int(w * 0.08))
    bh = max(10, int(h * 0.08))

    borders = np.concatenate([
        arr[:bh, :, :].reshape(-1, 3),
        arr[-bh:, :, :].reshape(-1, 3),
        arr[:, :bw, :].reshape(-1, 3),
        arr[:, -bw:, :].reshape(-1, 3)
    ], axis=0)

    std = float(np.std(borders))
    return std


def crop_to_square(img: Image.Image):
    return ImageOps.fit(img, (1000, 1000), method=Image.Resampling.LANCZOS)


def draw_face_box(img_cv, face):
    annotated = img_cv.copy()
    if face is not None:
        x, y, w, h = face
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def criterion_row(label, passed, detail, severity="media"):
    icon = "✅" if passed else "⚠️"
    return {
        "Criterio": label,
        "Estado": icon,
        "Detalle": detail,
        "Severidad": severity,
    }


def classify_photo(rows):
    failed_high = sum(1 for r in rows if r["Estado"] == "⚠️" and r["Severidad"] == "alta")
    failed_medium = sum(1 for r in rows if r["Estado"] == "⚠️" and r["Severidad"] == "media")
    failed_low = sum(1 for r in rows if r["Estado"] == "⚠️" and r["Severidad"] == "baja")

    if failed_high >= 1 or failed_medium >= 3:
        return "rechazada"
    if failed_medium >= 1 or failed_low >= 2:
        return "observada"
    return "aprobada"


def recommendation_text(status_key):
    if status_key == "aprobada":
        return "La fotografía puede utilizarse en el perfil docente sin cambios adicionales."
    if status_key == "observada":
        return "La fotografía puede utilizarse solo si se aplican ajustes o se valida manualmente."
    return "La fotografía no cumple con los criterios mínimos y se recomienda solicitar una nueva imagen."


st.title("Validador de fotografías docentes")
st.write(
    "Subí una foto y el sistema revisará criterios técnicos y visuales para perfiles docentes en cursos virtuales."
)

with st.sidebar:
    st.header("Configuración institucional")
    min_resolution = st.selectbox(
        "Resolución mínima",
        options=[600, 800, 1000, 1200],
        index=1,
        help="Se aplicará tanto al ancho como al alto de la imagen original."
    )
    allow_auto_crop = st.checkbox(
        "Permitir recorte automático a cuadrado en la revisión",
        value=True
    )
    use_institutional_verdict = st.checkbox(
        "Usar dictamen institucional (Aprobada / Observada / Rechazada)",
        value=True
    )
    st.markdown("---")
    st.caption("Esta configuración permite adaptar el prototipo al criterio de tu equipo.")

with st.expander("Criterios evaluados"):
    st.markdown(
        """
- Formato JPG o PNG
- Resolución mínima sugerida: 800 x 800
- Relación cuadrada
- Iluminación suficiente
- Nitidez aceptable
- Presencia de rostro frontal detectable
- Rostro centrado
- Proporción de rostro compatible con retrato profesional
- Fondo relativamente uniforme

**Nota:** vestimenta discreta, postura recta exacta y plano desde la cintura no se pueden validar con total certeza usando reglas simples. Este prototipo los infiere parcialmente.
        """
    )

uploaded_file = st.file_uploader("Subir foto", type=["jpg", "jpeg", "png"])

auto_crop = st.checkbox(
    "Generar vista previa recortada a formato cuadrado",
    value=allow_auto_crop
)

observations_manual = st.text_area(
    "Observaciones manuales opcionales",
    placeholder="Ej.: fondo no institucional o encuadre muy cerrado."
)

if uploaded_file:
    ok_format, format_msg = image_format_ok(uploaded_file)
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size

    image_for_analysis = crop_to_square(image) if auto_crop else image

    img_cv = pil_to_cv(image_for_analysis)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    brightness = analyze_brightness(gray)
    sharpness = analyze_sharpness(gray)
    face = detect_face(gray)
    bg_std = background_uniformity(image_for_analysis)

    rows = []
    rows.append(criterion_row("Formato", ok_format, format_msg))

    res_ok = width >= min_resolution and height >= min_resolution
    rows.append(
        criterion_row(
            "Resolución",
            res_ok,
            f"Resolución original: {width} x {height} | mínimo esperado: {min_resolution} x {min_resolution}",
            severity="alta"
        )
    )

    square_ok = is_square(width, height)
    square_detail = "Formato cuadrado" if square_ok else f"Relación actual: {width}:{height}. Se recomienda 1:1"
    rows.append(criterion_row("Formato cuadrado", square_ok, square_detail, severity="media"))

    bright_ok = MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS
    rows.append(
        criterion_row(
            "Iluminación",
            bright_ok,
            f"Brillo promedio: {brightness:.1f}",
            severity="alta"
        )
    )

    sharp_ok = sharpness >= MIN_SHARPNESS
    rows.append(
        criterion_row(
            "Nitidez",
            sharp_ok,
            f"Índice de nitidez: {sharpness:.1f}",
            severity="alta"
        )
    )

    if face is None:
        rows.append(criterion_row("Rostro detectable", False, "No se detectó un rostro frontal con suficiente claridad", severity="alta"))
        rows.append(criterion_row("Rostro centrado", False, "No evaluable sin detección de rostro", severity="media"))
        rows.append(criterion_row("Encuadre del retrato", False, "No evaluable sin detección de rostro", severity="media"))
    else:
        rows.append(criterion_row("Rostro detectable", True, "Se detectó un rostro frontal", severity="alta"))

        dx, dy = face_centering(face, image_for_analysis.size[0], image_for_analysis.size[1])
        centered_ok = dx <= CENTER_TOLERANCE and dy <= 0.18
        rows.append(
            criterion_row(
                "Rostro centrado",
                centered_ok,
                f"Desvío horizontal: {dx:.2f} | desvío vertical: {dy:.2f}",
                severity="media"
            )
        )

        ratio = face_ratio(face, image_for_analysis.size[0], image_for_analysis.size[1])
        framing_ok = FACE_MIN_RATIO <= ratio <= FACE_MAX_RATIO
        detail = f"Proporción del rostro respecto a la imagen: {ratio:.2f}"
        rows.append(criterion_row("Encuadre del retrato", framing_ok, detail, severity="media"))

    bg_ok = bg_std <= BORDER_STD_MAX
    rows.append(
        criterion_row(
            "Fondo uniforme",
            bg_ok,
            f"Variación del borde: {bg_std:.1f} (menor suele indicar fondo más liso)",
            severity="media"
        )
    )

    rows.append(
        criterion_row(
            "Revisión manual de vestimenta y postura",
            False if not observations_manual else True,
            "Completar por el equipo revisor para confirmar vestimenta discreta, hombros visibles y postura adecuada.",
            severity="baja"
        )
    )

    passed = sum(1 for r in rows if r["Estado"] == "✅")
    total = len(rows)
    score = round((passed / total) * 100)

    status_key = classify_photo(rows) if use_institutional_verdict else (
        "aprobada" if score >= 85 else "observada" if score >= 65 else "rechazada"
    )
    verdict = SEM_STATUS[status_key]
    verdict_help = recommendation_text(status_key)
    verdict_type = "success" if status_key == "aprobada" else "warning" if status_key == "observada" else "error"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Vista previa")
        st.image(image, caption="Imagen original", use_container_width=True)
        if auto_crop:
            st.image(image_for_analysis, caption="Vista previa cuadrada", use_container_width=True)

    with col2:
        st.subheader("Detección")
        annotated = draw_face_box(img_cv, face)
        st.image(annotated, caption="Rostro detectado", use_container_width=True)

        st.subheader("Resultado general")
        if verdict_type == "success":
            st.success(f"{verdict} — {score}%")
        elif verdict_type == "warning":
            st.warning(f"{verdict} — {score}%")
        else:
            st.error(f"{verdict} — {score}%")
        st.write(verdict_help)

        st.subheader("Semáforo institucional")
        semaforo_cols = st.columns(3)
        with semaforo_cols[0]:
            st.metric("Aprobada", "Sí" if status_key == "aprobada" else "No")
        with semaforo_cols[1]:
            st.metric("Observada", "Sí" if status_key == "observada" else "No")
        with semaforo_cols[2]:
            st.metric("Rechazada", "Sí" if status_key == "rechazada" else "No")

    st.subheader("Checklist de validación")
    df_rows = pd.DataFrame(rows)
    st.dataframe(df_rows, use_container_width=True, hide_index=True)

    st.subheader("Estado final")
    st.write(f"**Estado sugerido por el sistema:** {SEM_STATUS[status_key]}")

    st.subheader("Sugerencias automáticas")
    suggestions = []
    if not square_ok:
        suggestions.append("Recortar la imagen a formato cuadrado 1:1.")
    if not bright_ok:
        suggestions.append("Mejorar la iluminación o ajustar exposición antes de publicar la foto.")
    if not sharp_ok:
        suggestions.append("Solicitar una imagen más nítida y en mayor calidad.")
    if face is None:
        suggestions.append("Usar una foto de frente, con el rostro bien visible y sin filtros.")
    if face is not None:
        if not centered_ok:
            suggestions.append("Reencuadrar la foto para centrar mejor a la persona.")
        if not framing_ok:
            suggestions.append("Ajustar el encuadre para un plano medio profesional.")
    if not bg_ok:
        suggestions.append("Remover el fondo con Remove.bg y colocar un fondo liso, preferiblemente blanco.")

    if suggestions:
        for s in suggestions:
            st.write(f"- {s}")
    else:
        st.write("No se detectaron ajustes importantes.")

    report = f"""
Estado automático: {SEM_STATUS[status_key]}
Puntaje: {score}%

Observaciones manuales:
{observations_manual if observations_manual else 'Sin observaciones manuales.'}
"""

    st.subheader("Reporte breve")
    st.text_area("Resumen exportable", value=report.strip(), height=180)

    # Descarga del recorte cuadrado
    if auto_crop:
        buffer = io.BytesIO()
        image_for_analysis.save(buffer, format="PNG")
        st.download_button(
            label="Descargar vista previa cuadrada",
            data=buffer.getvalue(),
            file_name="foto_docente_cuadrada.png",
            mime="image/png"
        )

    csv_buffer = io.StringIO()
    df_export = df_rows.copy()
    df_export["Estado_sistema"] = SEM_STATUS[status_key]
    df_export["Observaciones_manuales"] = observations_manual if observations_manual else ""
    df_export.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Descargar reporte CSV",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="reporte_validacion_foto_docente.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption(
    "Prototipo base. Para una versión más robusta se puede integrar segmentación de fondo, detección de hombros, reglas de vestimenta y revisión asistida por IA."
)
