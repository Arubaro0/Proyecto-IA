"""
Streamlit App: Sistema de Identificación de Especies de Aves
P02 - Introducción a la IA — Clasificación mediante Random Forest + Atributos Descriptivos
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Clasificador de Aves — Atributos",
    page_icon="🦅",
    layout="wide"
)

MODEL_PATH   = 'models/bird_rf_model.joblib'
COLUMNS_PATH = 'models/feature_columns.joblib'
CLASSES_PATH = 'CUB_200_2011/classes.txt'

ATTR_NAMES = {
    # Forma del pico (1-9)
    1: "Pico: curvado (sin gancho)", 2: "Pico: tipo daga", 3: "Pico: con gancho",
    4: "Pico: tipo aguja", 5: "Pico: gancho marino", 6: "Pico: tipo espátula",
    7: "Pico: uso general", 8: "Pico: cónico", 9: "Pico: especializado",
    # Color del ala (10-24)
    10: "Color ala: azul", 11: "Color ala: marrón", 12: "Color ala: iridiscente",
    13: "Color ala: morado", 14: "Color ala: rojizo", 15: "Color ala: gris",
    16: "Color ala: amarillo", 17: "Color ala: oliva", 18: "Color ala: verde",
    19: "Color ala: rosa", 20: "Color ala: naranja", 21: "Color ala: negro",
    22: "Color ala: blanco", 23: "Color ala: rojo", 24: "Color ala: beige",
    # Color partes superiores (25-39)
    25: "Partes sup.: azul", 26: "Partes sup.: marrón", 27: "Partes sup.: iridiscente",
    28: "Partes sup.: morado", 29: "Partes sup.: rojizo", 30: "Partes sup.: gris",
    31: "Partes sup.: amarillo", 32: "Partes sup.: oliva", 33: "Partes sup.: verde",
    34: "Partes sup.: rosa", 35: "Partes sup.: naranja", 36: "Partes sup.: negro",
    37: "Partes sup.: blanco", 38: "Partes sup.: rojo", 39: "Partes sup.: beige",
    # Color partes inferiores (40-54)
    40: "Partes inf.: azul", 41: "Partes inf.: marrón", 42: "Partes inf.: iridiscente",
    43: "Partes inf.: morado", 44: "Partes inf.: rojizo", 45: "Partes inf.: gris",
    46: "Partes inf.: amarillo", 47: "Partes inf.: oliva", 48: "Partes inf.: verde",
    49: "Partes inf.: rosa", 50: "Partes inf.: naranja", 51: "Partes inf.: negro",
    52: "Partes inf.: blanco", 53: "Partes inf.: rojo", 54: "Partes inf.: beige",
    # Color pecho (55-69)
    55: "Color pecho: azul", 56: "Color pecho: marrón", 57: "Color pecho: iridiscente",
    58: "Color pecho: morado", 59: "Color pecho: rojizo", 60: "Color pecho: gris",
    61: "Color pecho: amarillo", 62: "Color pecho: oliva", 63: "Color pecho: verde",
    64: "Color pecho: rosa", 65: "Color pecho: naranja", 66: "Color pecho: negro",
    67: "Color pecho: blanco", 68: "Color pecho: rojo", 69: "Color pecho: beige",
    # Color espalda (70-84)
    70: "Color espalda: azul", 71: "Color espalda: marrón", 72: "Color espalda: iridiscente",
    73: "Color espalda: morado", 74: "Color espalda: rojizo", 75: "Color espalda: gris",
    76: "Color espalda: amarillo", 77: "Color espalda: oliva", 78: "Color espalda: verde",
    79: "Color espalda: rosa", 80: "Color espalda: naranja", 81: "Color espalda: negro",
    82: "Color espalda: blanco", 83: "Color espalda: rojo", 84: "Color espalda: beige",
    # Forma de la cola (85-90)
    85: "Cola: bifurcada", 86: "Cola: redondeada", 87: "Cola: escotada",
    88: "Cola: en abanico", 89: "Cola: puntiaguda", 90: "Cola: cuadrada",
    # Color cola superior (91-105)
    91: "Cola sup.: azul", 92: "Cola sup.: marrón", 93: "Cola sup.: iridiscente",
    94: "Cola sup.: morado", 95: "Cola sup.: rojizo", 96: "Cola sup.: gris",
    97: "Cola sup.: amarillo", 98: "Cola sup.: oliva", 99: "Cola sup.: verde",
    100: "Cola sup.: rosa", 101: "Cola sup.: naranja", 102: "Cola sup.: negro",
    103: "Cola sup.: blanco", 104: "Cola sup.: rojo", 105: "Cola sup.: beige",
    # Patrón cabeza (106-116)
    106: "Patrón cabeza: sólido", 107: "Patrón cabeza: rayado", 108: "Patrón cabeza: multicolor",
    109: "Patrón cabeza: moteado", 110: "Patrón cabeza: liso", 111: "Patrón cabeza: único",
    112: "Patrón cabeza: ceja superciliar", 113: "Patrón cabeza: anillo ocular",
    114: "Patrón cabeza: antifaz", 115: "Patrón cabeza: cresta", 116: "Patrón cabeza: bigotillo",
    # Patrón pecho (117-120)
    117: "Patrón pecho: sólido", 118: "Patrón pecho: moteado",
    119: "Patrón pecho: rayado", 120: "Patrón pecho: multicolor",
    # Patrón espalda (121-124)
    121: "Patrón espalda: sólido", 122: "Patrón espalda: moteado",
    123: "Patrón espalda: rayado", 124: "Patrón espalda: multicolor",
    # Patrón cola (125-128)
    125: "Patrón cola: sólido", 126: "Patrón cola: moteado",
    127: "Patrón cola: rayado", 128: "Patrón cola: multicolor",
    # Patrón ala (129-132)
    129: "Patrón ala: sólido", 130: "Patrón ala: moteado",
    131: "Patrón ala: rayado", 132: "Patrón ala: multicolor",
    # Tamaño (133-136)
    133: "Tamaño: grande (40-80 cm)", 134: "Tamaño: pequeño (13-23 cm)",
    135: "Tamaño: muy grande (80-180 cm)", 136: "Tamaño: mediano (23-40 cm)",
    # Forma corporal (137-150)
    137: "Forma: ave acuática erguida", 138: "Forma: ave de marisma",
    139: "Forma: ave zancuda", 140: "Forma: tipo pato",
    141: "Forma: tipo búho", 142: "Forma: gaviota costera",
    143: "Forma: tipo halcón", 144: "Forma: tipo correlimos",
    145: "Forma: ave terrestre", 146: "Forma: tipo golondrina",
    147: "Forma: ave de percha", 148: "Forma: tipo colibrí",
    149: "Forma: ave pelágica", 150: "Forma: tipo somormujo",
    # Color vientre (151-165)
    151: "Color vientre: azul", 152: "Color vientre: marrón", 153: "Color vientre: iridiscente",
    154: "Color vientre: morado", 155: "Color vientre: rojizo", 156: "Color vientre: gris",
    157: "Color vientre: amarillo", 158: "Color vientre: oliva", 159: "Color vientre: verde",
    160: "Color vientre: rosa", 161: "Color vientre: naranja", 162: "Color vientre: negro",
    163: "Color vientre: blanco", 164: "Color vientre: rojo", 165: "Color vientre: beige",
    # Longitud del pico (166-168)
    166: "Longitud pico: similar a la cabeza", 167: "Longitud pico: más largo que la cabeza",
    168: "Longitud pico: más corto que la cabeza",
    # Color frente (169-183)
    169: "Color frente: azul", 170: "Color frente: marrón", 171: "Color frente: iridiscente",
    172: "Color frente: morado", 173: "Color frente: rojizo", 174: "Color frente: gris",
    175: "Color frente: amarillo", 176: "Color frente: oliva", 177: "Color frente: verde",
    178: "Color frente: rosa", 179: "Color frente: naranja", 180: "Color frente: negro",
    181: "Color frente: blanco", 182: "Color frente: rojo", 183: "Color frente: beige",
    # Color cola inferior (184-198)
    184: "Cola inf.: azul", 185: "Cola inf.: marrón", 186: "Cola inf.: iridiscente",
    187: "Cola inf.: morado", 188: "Cola inf.: rojizo", 189: "Cola inf.: gris",
    190: "Cola inf.: amarillo", 191: "Cola inf.: oliva", 192: "Cola inf.: verde",
    193: "Cola inf.: rosa", 194: "Cola inf.: naranja", 195: "Cola inf.: negro",
    196: "Cola inf.: blanco", 197: "Cola inf.: rojo", 198: "Cola inf.: beige",
    # Color nuca (199-213)
    199: "Color nuca: azul", 200: "Color nuca: marrón", 201: "Color nuca: iridiscente",
    202: "Color nuca: morado", 203: "Color nuca: rojizo", 204: "Color nuca: gris",
    205: "Color nuca: amarillo", 206: "Color nuca: oliva", 207: "Color nuca: verde",
    208: "Color nuca: rosa", 209: "Color nuca: naranja", 210: "Color nuca: negro",
    211: "Color nuca: blanco", 212: "Color nuca: rojo", 213: "Color nuca: beige",
    # Patrón vientre (214-217)
    214: "Patrón vientre: sólido", 215: "Patrón vientre: moteado",
    216: "Patrón vientre: rayado", 217: "Patrón vientre: multicolor",
    # Color plumas primarias (218-232)
    218: "Plumas prim.: azul", 219: "Plumas prim.: marrón", 220: "Plumas prim.: iridiscente",
    221: "Plumas prim.: morado", 222: "Plumas prim.: rojizo", 223: "Plumas prim.: gris",
    224: "Plumas prim.: amarillo", 225: "Plumas prim.: oliva", 226: "Plumas prim.: verde",
    227: "Plumas prim.: rosa", 228: "Plumas prim.: naranja", 229: "Plumas prim.: negro",
    230: "Plumas prim.: blanco", 231: "Plumas prim.: rojo", 232: "Plumas prim.: beige",
    # Color patas (233-247)
    233: "Color patas: azul", 234: "Color patas: marrón", 235: "Color patas: iridiscente",
    236: "Color patas: morado", 237: "Color patas: rojizo", 238: "Color patas: gris",
    239: "Color patas: amarillo", 240: "Color patas: oliva", 241: "Color patas: verde",
    242: "Color patas: rosa", 243: "Color patas: naranja", 244: "Color patas: negro",
    245: "Color patas: blanco", 246: "Color patas: rojo", 247: "Color patas: beige",
    # Color pico (248-262)
    248: "Color pico: azul", 249: "Color pico: marrón", 250: "Color pico: iridiscente",
    251: "Color pico: morado", 252: "Color pico: rojizo", 253: "Color pico: gris",
    254: "Color pico: amarillo", 255: "Color pico: oliva", 256: "Color pico: verde",
    257: "Color pico: rosa", 258: "Color pico: naranja", 259: "Color pico: negro",
    260: "Color pico: blanco", 261: "Color pico: rojo", 262: "Color pico: beige",
    # Color corona (263-277)
    263: "Color corona: azul", 264: "Color corona: marrón", 265: "Color corona: iridiscente",
    266: "Color corona: morado", 267: "Color corona: rojizo", 268: "Color corona: gris",
    269: "Color corona: amarillo", 270: "Color corona: oliva", 271: "Color corona: verde",
    272: "Color corona: rosa", 273: "Color corona: naranja", 274: "Color corona: negro",
    275: "Color corona: blanco", 276: "Color corona: rojo", 277: "Color corona: beige",
    # Color garganta (278-292)
    278: "Color garganta: azul", 279: "Color garganta: marrón", 280: "Color garganta: iridiscente",
    281: "Color garganta: morado", 282: "Color garganta: rojizo", 283: "Color garganta: gris",
    284: "Color garganta: amarillo", 285: "Color garganta: oliva", 286: "Color garganta: verde",
    287: "Color garganta: rosa", 288: "Color garganta: naranja", 289: "Color garganta: negro",
    290: "Color garganta: blanco", 291: "Color garganta: rojo", 292: "Color garganta: beige",
    # Forma del ala (293-297)
    293: "Forma ala: puntiaguda", 294: "Forma ala: redondeada", 295: "Forma ala: ancha",
    296: "Forma ala: ahusada", 297: "Forma ala: larga",
    # Color ojos (298-312)
    298: "Color ojos: azul", 299: "Color ojos: marrón", 300: "Color ojos: iridiscente",
    301: "Color ojos: morado", 302: "Color ojos: rojizo", 303: "Color ojos: gris",
    304: "Color ojos: amarillo", 305: "Color ojos: oliva", 306: "Color ojos: verde",
    307: "Color ojos: rosa", 308: "Color ojos: naranja", 309: "Color ojos: negro",
    310: "Color ojos: blanco", 311: "Color ojos: rojo", 312: "Color ojos: beige",
}

GRUPOS = {
    "Pico":             list(range(1, 10))   + list(range(166, 169)) + list(range(248, 263)),
    "Alas y cola":      list(range(10, 25))  + list(range(85, 106))  + list(range(121, 133))
                      + list(range(184, 199)) + list(range(218, 233)) + list(range(293, 298)),
    "Cabeza":           list(range(106, 117)) + list(range(169, 184)) + list(range(199, 214))
                      + list(range(263, 293)),
    "Pecho y vientre":  list(range(55, 70))  + list(range(117, 121)) + list(range(151, 166))
                      + list(range(214, 218)),
    "Colores dorsales": list(range(25, 55))  + list(range(70, 85)),
    "Forma y tamaño":   list(range(133, 151)),
    "Patas y otros":    list(range(233, 248)) + list(range(298, 313)),
}

# ---------------------------------------------------------------------------
# Carga de recursos
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
    df['class_name'] = df['class_name'].str.replace(r'^\d+\.', '', regex=True).str.replace('_', ' ')
    return dict(zip(df['class_id'], df['class_name']))

model        = cargar_modelo()
feature_cols = cargar_columnas()
clases       = cargar_clases()

feature_set = set(feature_cols)

# ---------------------------------------------------------------------------
# Sidebar
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
    "tamaño, forma…) y el sistema predicirá la especie usando un **Random Forest**."
)
st.divider()

# ---------------------------------------------------------------------------
# Formulario de atributos
# ---------------------------------------------------------------------------
st.subheader("Atributos del ave")
st.caption("Marca con ✔ los atributos que presenta el ejemplar. Deja sin marcar los ausentes.")

valores = {}

tabs = st.tabs(list(GRUPOS.keys()))
for tab, (grupo, ids) in zip(tabs, GRUPOS.items()):
    with tab:
        cols = st.columns(3)
        for i, attr_id in enumerate(ids):
            if attr_id in feature_set:
                label = ATTR_NAMES.get(attr_id, f"Atributo {attr_id}")
                with cols[i % 3]:
                    valores[attr_id] = 1 if st.checkbox(label, key=f"attr_{attr_id}") else 0

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
        x_input = np.array([[valores.get(col, 0) for col in feature_cols]])
        df_input = pd.DataFrame(x_input, columns=feature_cols)

        pred_class  = model.predict(df_input)[0]
        pred_probas = model.predict_proba(df_input)[0]

        top5_idx    = np.argsort(pred_probas)[-5:][::-1]
        top5_clases = model.classes_[top5_idx]
        top5_probs  = pred_probas[top5_idx]

        especie_predicha = clases.get(pred_class, f"Clase {pred_class}")
        confianza        = pred_probas[np.where(model.classes_ == pred_class)[0][0]] * 100

        st.success(f"### Especie predicha: **{especie_predicha}**")
        st.metric("Confianza", f"{confianza:.1f}%")

        st.markdown("#### Top 5 especies más probables")
        for clase_id, prob in zip(top5_clases, top5_probs):
            nombre = clases.get(clase_id, f"Clase {clase_id}")
            st.progress(float(prob), text=f"**{nombre}** — {prob*100:.1f}%")

        n_marcados = sum(valores.values())
        st.caption(f"Clasificación basada en {n_marcados} atributo(s) marcado(s) de {len(feature_cols)} posibles.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("P02 — Introducción a la IA · Clasificación de Aves con Random Forest · Dataset CUB-200-2011")
