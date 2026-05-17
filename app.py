"""
Streamlit App: Sistema de Identificación de Especies de Aves
P02 - Introducción a la IA — Clasificación mediante Random Forest + Atributos Descriptivos
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clasificador de Aves — Atributos",
    page_icon="🦅",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MODEL_PATH   = 'models/bird_rf_model.joblib'
COLUMNS_PATH = 'models/feature_columns.joblib'
CLASSES_PATH = 'CUB_200_2011/classes.txt'

# ---------------------------------------------------------------------------
# Carga de recursos (con caché para no recargar en cada interacción)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource
def cargar_columnas():
    if not os.path.exists(COLUMNS_PATH):
        return list(range(1, 313))
    return joblib.load(COLUMNS_PATH)

@st.cache_data
def cargar_clases():
    if not os.path.exists(CLASSES_PATH):
        return {i: f'Clase {i}' for i in range(1, 201)}
    df = pd.read_csv(CLASSES_PATH, sep=' ', header=None, names=['class_id', 'class_name'])
    # Limpiar prefijos numéricos del nombre (ej: "001.Black_footed_Albatross" → "Black footed Albatross")
    df['class_name'] = df['class_name'].str.replace(r'^\d+\.', '', regex=True).str.replace('_', ' ')
    return dict(zip(df['class_id'], df['class_name']))

model        = cargar_modelo()
feature_cols = cargar_columnas()
clases       = cargar_clases()

# ---------------------------------------------------------------------------
# Sidebar — información del modelo
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🦅 Clasificador de Aves")
    st.markdown("**Proyecto P02 — Introducción a la IA**")
    st.divider()

    if model is not None:
        st.success("Modelo cargado correctamente")
        st.markdown(f"""
        **Técnica**: Random Forest
        **Atributos**: {len(feature_cols)} características binarias
        **Especies**: {len(clases)} clases
        **Estimadores**: {model.n_estimators}
        **Max features**: {model.max_features}
        """)
    else:
        st.error("Modelo no encontrado")
        st.info("Ejecuta primero el notebook `P02_aves_random_forest.ipynb` para entrenar y guardar el modelo en `models/`.")

    st.divider()
    st.markdown("""
    **Cómo usar:**
    1. Marca los atributos que observas en el ave
    2. Pulsa **Clasificar**
    3. Obtén la especie predicha con sus probabilidades
    """)

# ---------------------------------------------------------------------------
# Título principal
# ---------------------------------------------------------------------------
st.title("🦅 Sistema de Clasificación de Especies de Aves")
st.markdown(
    "Introduce los atributos observables del ave (características de pico, color, "
    "tamaño, hábitat…) y el sistema predicirá la especie usando un **Random Forest**."
)
st.divider()

# ---------------------------------------------------------------------------
# Formulario de atributos
# ---------------------------------------------------------------------------
# Agrupamos los 312 atributos en categorías de ~30 para facilitar la navegación
GRUPOS = {
    "Forma del pico (1–30)":    list(range(1, 31)),
    "Color del pico (31–60)":   list(range(31, 61)),
    "Forma de las alas (61–90)": list(range(61, 91)),
    "Color del cuerpo (91–130)": list(range(91, 131)),
    "Partes superiores (131–170)": list(range(131, 171)),
    "Partes inferiores (171–210)": list(range(171, 211)),
    "Cola y patas (211–250)":   list(range(211, 251)),
    "Tamaño y forma (251–280)": list(range(251, 281)),
    "Hábitat y comportamiento (281–312)": list(range(281, 313)),
}

st.subheader("Atributos del ave")
st.caption("Marca con ✔ los atributos que presenta el ejemplar. Deja sin marcar los ausentes.")

valores = {}

tabs = st.tabs(list(GRUPOS.keys()))
for tab, (grupo, ids) in zip(tabs, GRUPOS.items()):
    with tab:
        # Columnas de 3 para mejor lectura
        cols = st.columns(3)
        for i, attr_id in enumerate(ids):
            # Solo mostrar atributos que existen en las columnas del modelo
            if attr_id in feature_cols:
                with cols[i % 3]:
                    valores[attr_id] = 1 if st.checkbox(f"Atributo {attr_id}", key=f"attr_{attr_id}") else 0

st.divider()

# ---------------------------------------------------------------------------
# Botón de clasificación
# ---------------------------------------------------------------------------
col_btn, col_reset = st.columns([1, 5])
with col_btn:
    clasificar = st.button("🔍 Clasificar", type="primary", use_container_width=True)
with col_reset:
    if st.button("Limpiar selección", use_container_width=False):
        st.rerun()

# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
if clasificar:
    if model is None:
        st.error("No hay modelo cargado. Entrena el modelo primero.")
    else:
        # Construir vector de entrada en el orden correcto
        x_input = np.array([[valores.get(col, 0) for col in feature_cols]])
        df_input = pd.DataFrame(x_input, columns=feature_cols)

        # Predicción
        pred_class  = model.predict(df_input)[0]
        pred_probas = model.predict_proba(df_input)[0]

        # Top 5 clases con mayor probabilidad
        top5_idx   = np.argsort(pred_probas)[-5:][::-1]
        top5_clases = model.classes_[top5_idx]
        top5_probs  = pred_probas[top5_idx]

        especie_predicha = clases.get(pred_class, f"Clase {pred_class}")
        confianza        = pred_probas[np.where(model.classes_ == pred_class)[0][0]] * 100

        # Resultado principal
        st.success(f"### Especie predicha: **{especie_predicha}**")
        st.metric("Confianza", f"{confianza:.1f}%")

        st.markdown("#### Top 5 especies más probables")
        for clase_id, prob in zip(top5_clases, top5_probs):
            nombre = clases.get(clase_id, f"Clase {clase_id}")
            st.progress(float(prob), text=f"**{nombre}** — {prob*100:.1f}%")

        # Atributos marcados
        n_marcados = sum(valores.values())
        st.caption(f"Clasificación basada en {n_marcados} atributo(s) marcado(s) de {len(feature_cols)} posibles.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("P02 — Introducción a la IA · Clasificación de Aves con Random Forest · Dataset CUB-200-2011")
