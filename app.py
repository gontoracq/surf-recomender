import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
import numpy as np

import requests
from bs4 import BeautifulSoup

# ---------------- CARGA DATOS ----------------

df = pd.read_csv('df_surf.csv')
df2 = pd.read_csv('responses_limpio.csv')
df3 = pd.read_csv('Formulario_limpio.csv', sep=';')
df4 = pd.read_csv('beginner.csv', sep=';')
df5 = pd.read_csv('first_timer.csv', sep=';')

# ---------------- FEATURE ENGINEERING ----------------

df["masa_corporal"] = df['surfer_weight'] / (df['surfer_height']*df['surfer_height'])

#df board_type es tipo de tabla de surf
#first_timer type es tipo de tabla de surf
#evolutivas type es tipo tabla de surf

df2.columns.values[1] = "board_length"
df2.columns.values[2] = "board_width"
df2.columns.values[3] = "board_thickness"
df2.columns.values[4] = "board_volume"
df2.columns.values[5] = "surfer_height"
df2.columns.values[6] = "surfer_weight"
df2.columns.values[7] = "surfer_experience"

df2["masa_corporal"] = df2['surfer_weight'] / (df2['surfer_height']*df2['surfer_height'])

df3.columns.values[0] = "board_length"
df3.columns.values[1] = "board_width"
df3.columns.values[2] = "board_thickness"
df3.columns.values[3] = "board_volume"
df3.columns.values[4] = "surfer_height"
df3.columns.values[5] = "surfer_weight"
df3.columns.values[6] = "surfer_experience"
df3["masa_corporal"] = df3['surfer_weight'] / (df3['surfer_height']*df3['surfer_height'])


df4.columns.values[0] = "board_length"
df4.columns.values[1] = "board_width"
df4.columns.values[2] = "board_thickness"
df4.columns.values[3] = "board_volume"
df4.columns.values[4] = "surfer_height"
df4.columns.values[5] = "surfer_weight"
df4.columns.values[6] = "surfer_experience"
df4["masa_corporal"] = df4['surfer_weight'] / (df4['surfer_height']*df4['surfer_height'])

df5.columns.values[0] = "board_length"
df5.columns.values[1] = "board_width"
df5.columns.values[2] = "board_thickness"
df5.columns.values[3] = "board_volume"
df5.columns.values[4] = "surfer_height"
df5.columns.values[5] = "surfer_weight"
df5.columns.values[6] = "surfer_experience"
df5["masa_corporal"] = df5['surfer_weight'] / (df5['surfer_height']*df5['surfer_height'])

feat = ["board_length", "board_width", "board_thickness","board_volume",
        "surfer_height", "surfer_weight", "surfer_experience", "masa_corporal"]

df = df[feat]
df2 = df2[feat]
df3 = df3.drop([7, 8])

df['board_length'] = df['board_length'] * 3.28084
df['surfer_height'] = df['surfer_height'] * 100

df = df.round(3)

feat = [
    "board_length",
    "board_width",
    "board_thickness",
    "board_volume",
    "surfer_height",
    "surfer_weight",
    "surfer_experience",
    "masa_corporal"
]

frames = [df,df2,df3,df4,df5]
result = pd.concat(frames)

features = [
    "surfer_height",
    "surfer_weight",
    "surfer_experience",
    "masa_corporal"
]

board_features = [
    "board_length",
    "board_thickness",
    "board_width",
    "board_volume"
]

d = {
    'First-timer': 0,
    'Beginner': 1,
    'Intermediate': 2,
    'Advanced': 3,
    'Pro': 4
}

result.surfer_experience = result.surfer_experience.map(d)

imputer = KNNImputer(n_neighbors=4)

result[features + board_features] = imputer.fit_transform(
    result[features + board_features]
)

pesos = np.array([1, 1, 100, 1])

X = result[features]
y = result[board_features]

X_weighted = X * pesos

knn = NearestNeighbors(
    n_neighbors=3,
    metric="cosine"
)

knn.fit(X_weighted)

# ---------------- ALGORITMO RECOMENDADOR ----------------

def recomendar_tabla(altura, peso, nivel, olas_grandes):

    masa_corporal = peso / ((altura) * (altura))

    user_input = np.array([
        [altura, peso, nivel, masa_corporal]
    ]) * pesos

    _, idx = knn.kneighbors(user_input)

    recomendacion = y.iloc[idx[0]].median()

    if olas_grandes:
        recomendacion[0] += 0.1

    tipo_tabla = "Hardboard"

    if recomendacion["board_volume"] > 60:
        tipo_tabla = 'Softboard'
    elif 50 <= recomendacion["board_volume"] <= 60:
        tipo_tabla = 'Softboard o Hardboard'

    return {
        "medidas": {
                "Largo": f"{round(recomendacion['board_length'], 2)} ft",
                "Grosor": f"{round(recomendacion['board_thickness'], 2)} in",
                "Ancho": f"{round(recomendacion['board_width'], 2)} in",
                "Volumen": f"{round(recomendacion['board_volume'], 2)} L",
        },
        "tipo": tipo_tabla
    }

# ------------------ SCRAPPING -------------------------

def generar_busqueda_decathlon(medidas, tipo):

    volumen = medidas["Volumen"].replace(" L", "")

    largo_raw = medidas["Largo"].replace(" ft", "")

    largo = convertir_pies_a_surf(float(largo_raw))

    query = f"surfboard {largo} {volumen}"

    if "Softboard" in tipo:
        query += " foam"

    query = query.replace(" ", "%20")

    return f"https://www.decathlon.co.uk/search?Ntt={query}"

# ---------------- UI ----------------

st.title("🏄 Recomendador de tablas de surf")

altura = st.slider("Altura (cm)", 140, 220, 175)
peso = st.slider("Peso (kg)", 40, 130, 75)

nivel = st.selectbox(
    "Nivel",
    [0,1,2,3,4],
    format_func=lambda x: [
        "First-timer",
        "Beginner",
        "Intermediate",
        "Advanced",
        "Pro"
    ][x]
)

olas = st.checkbox("¿Olas grandes?")

if st.button("Recomendar"):

    resultado = recomendar_tabla(
        altura,
        peso,
        nivel,
        olas
    )

    st.success("Tabla recomendada")

    st.write("### Medidas")

    st.write(resultado["medidas"])

    st.write("### Tipo")

    st.write(resultado["tipo"])
    
    # ------- RECOMIENDA TABLAS DE LA WEB SEGUN RESULTADO --------
    
    def convertir_pies_a_surf(feet_decimal):

        pies = int(feet_decimal)
    
        pulgadas = round((feet_decimal - pies) * 12)
    
        return f"{pies}'{pulgadas}"
    
    volumen = float(resultado["medidas"]["Volumen"].replace(" L", ""))
    
    url_decathlon = generar_busqueda_decathlon(
        resultado["medidas"],
        resultado["tipo"]
    )
    
    st.write("## 🏄 Tablas similares en Decathlon")
    
    st.markdown(
        f"""
        ### [🔎 Ver tablas recomendadas en Decathlon]({url_decathlon})
        """
    )