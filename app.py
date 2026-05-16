import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import requests
from bs4 import BeautifulSoup

import re

# ---------------- FUNCIONES AUXILIARES--------------

def parse_surf_length(x):
    if pd.isna(x):
        return np.nan
    
    x = str(x).strip()

    # Caso 5.10 1/2 o similares
    if " " in x:
        parts = x.split()

        base = parts[0]
        frac = parts[1] if len(parts) > 1 else "0"

        # 5.10 -> pies + decimales mal usados
        try:
            feet = float(base)
        except:
            return np.nan

        # fracción tipo 1/2
        if "/" in frac:
            num, den = frac.split("/")
            frac_val = float(num) / float(den)
        else:
            frac_val = 0

        return feet * 12 + frac_val * 12

    # Caso simple 5.10 o 5.1
    try:
        return float(x) * 12
    except:
        return np.nan
    
def inches_to_feet_str(inches):

    total_inches = round(inches)

    feet = total_inches // 12
    remaining_inches = total_inches % 12

    return f"{feet}'{remaining_inches}"

# ---------------- CARGA DATOS ----------------

df = pd.read_csv('df_surf.csv')
df2 = pd.read_csv('responses_limpio.csv')
df3 = pd.read_csv('Formulario_limpio.csv', sep=';')
df4 = pd.read_csv('beginner.csv', sep=';')
df5 = pd.read_csv('first_timer.csv', sep=';')

# ---------------- FEATURE ENGINEERING ----------------

df["masa_corporal"] = df['surfer_weight'] / ((df['surfer_height'] / 100) ** 2)

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

df2["masa_corporal"] = df2['surfer_weight'] / ((df2['surfer_height'] / 100) ** 2)

df3.columns.values[0] = "board_length"
df3.columns.values[1] = "board_width"
df3.columns.values[2] = "board_thickness"
df3.columns.values[3] = "board_volume"
df3.columns.values[4] = "surfer_height"
df3.columns.values[5] = "surfer_weight"
df3.columns.values[6] = "surfer_experience"

df3["masa_corporal"] = df3['surfer_weight'] / ((df3['surfer_height'] / 100) ** 2)

df4.columns.values[0] = "board_length"
df4.columns.values[1] = "board_width"
df4.columns.values[2] = "board_thickness"
df4.columns.values[3] = "board_volume"
df4.columns.values[4] = "surfer_height"
df4.columns.values[5] = "surfer_weight"
df4.columns.values[6] = "surfer_experience"

df4["masa_corporal"] = df4['surfer_weight'] / ((df4['surfer_height'] / 100) ** 2)

df5.columns.values[0] = "board_length"
df5.columns.values[1] = "board_width"
df5.columns.values[2] = "board_thickness"
df5.columns.values[3] = "board_volume"
df5.columns.values[4] = "surfer_height"
df5.columns.values[5] = "surfer_weight"
df5.columns.values[6] = "surfer_experience"

df5["masa_corporal"] = df5['surfer_weight'] / ((df5['surfer_height'] / 100) ** 2)

feat = ["board_length", "board_width", "board_thickness","board_volume",
        "surfer_height", "surfer_weight", "surfer_experience", "masa_corporal"]

df["board_length"] = df["board_length"].apply(parse_surf_length)
df2["board_length"] = df2["board_length"].apply(parse_surf_length)
df3["board_length"] = df3["board_length"].apply(parse_surf_length)
df4["board_length"] = df4["board_length"].apply(parse_surf_length)
df5["board_length"] = df5["board_length"].apply(parse_surf_length)

df = df[feat]
df2 = df2[feat]
df3 = df3.drop([7, 8])

#df['board_length'] = df['board_length'] * 3.28084
#df['surfer_height'] = df['surfer_height'] * 100

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

pesos = np.array([2, 2, 10, 1])

X = result[features]
y = result[board_features]

# Escalado
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Aplicar pesos
X_weighted = X_scaled * pesos

# ---------------- EVALUACION MODELO ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X_weighted,
    y,
    test_size=0.2,
    random_state=42
)

knn = NearestNeighbors(
    n_neighbors=3,
    metric="cosine"
)

knn.fit(X_train)

# Predicciones manuales usando vecinos
predicciones = []

for x in X_test:

    distancias, idx = knn.kneighbors([x])

    pred = y_train.iloc[idx[0]].median(numeric_only=True)

    predicciones.append(pred.values)

predicciones = np.array(predicciones)

# Score MAE
mae = mean_absolute_error(y_test, predicciones)

# Score R2
r2 = r2_score(y_test, predicciones)


# Reentrenar con TODOS los datos
knn.fit(X_weighted)

# ---------------- ALGORITMO RECOMENDADOR ----------------

def recomendar_tabla(altura, peso, nivel, olas_grandes):

    altura_m = altura / 100
    masa_corporal = peso / (altura_m ** 2)

    user_input = np.array([
        [altura, peso, nivel, masa_corporal]
    ])
    
    # Escalar igual que training
    user_scaled = scaler.transform(user_input)
    
    # Aplicar pesos
    user_weighted = user_scaled * pesos
    
    _, idx = knn.kneighbors(user_weighted)
    
    distancias, idx = knn.kneighbors(user_weighted)

    distancia_media = float(distancias.mean())
    
    # Score similitud
    score_similitud = max(0, (1 - distancia_media)) * 100
    
    recomendacion = y.iloc[idx[0]].median(numeric_only=True)
    
    tipo_tabla = "Hardboard"
    
    if distancia_media < 0.05:
        confianza = "Alta"
    
    elif distancia_media < 0.10:
        confianza = "Media"
    
    else:
        confianza = "Baja"
    
    if olas_grandes:
        recomendacion["board_length"] += 1
        tipo_tabla = "Hardboard"

    if recomendacion["board_volume"] > 50:
        tipo_tabla = 'Softboard'
    elif 45 <= recomendacion["board_volume"] <= 60:
        tipo_tabla = 'Softboard o Hardboard'

    return {
        "medidas": {
                "Largo": f"{inches_to_feet_str(recomendacion['board_length'])} ft",
                "Grosor": f"{round(recomendacion['board_thickness'], 2)} in",
                "Ancho": f"{round(recomendacion['board_width'], 2)} in",
                "Volumen": f"{round(recomendacion['board_volume'], 2)} L",
        },
        "tipo": tipo_tabla,
        "confianza": confianza,
        "score": round(score_similitud, 1)
    }

    
    # Habría que calcular el porcentaje de acierto sobre la predicción y sacarlo y si no el
    # del propio algoritmo una vez se ha entrenado mas arriba decir nuestras predicciones
    # son de tanto

# ------------------ SCRAPPING -------------------------

def generar_busqueda_decathlon(medidas, tipo):

    volumen = medidas["Volumen"].replace(" L", "")

    largo_raw = medidas["Largo"].replace(" ft", "")

    largo = convertir_formato_surf(largo_raw)

    query = f"surfboard {largo} {volumen}"

    if "Softboard" in tipo:
        query += " foam"

    query = query.replace(" ", "%20")

    return f"https://www.mundo-surf.com/es/?mot_q=={query}"

# ---------------- UI ----------------

st.title("🏄 Meet Your Surf")

col1, col2 = st.columns(2)

with col1:
    altura = st.number_input(
        "Altura (cm)",
        min_value=140,
        max_value=220,
        value=175
    )

with col2:
    peso = st.number_input(
        "Peso (kg)",
        min_value=40,
        max_value=130,
        value=75
    )

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
    
    st.write("### Porcentaje de similitud")

    st.progress(resultado["score"] / 100)

    st.write(f"{resultado['score']}% similitud con surfers reales")
    
    st.write("### Confianza recomendación")
    
    st.write(f"Similitud encontrada: {resultado['confianza']}")
    
    #st.write("### Precisión del algoritmo")

    #st.write(f"Error medio (MAE): {round(mae, 2)}")

    #st.write(f"Precisión R²: {round(r2 * 100, 2)}%")

    #st.write(f"{resultado['confianza']}% similitud con surfers reales")
    
    # ------- RECOMIENDA TABLAS DE LA WEB SEGUN RESULTADO --------
    
    def convertir_formato_surf(largo):

        return str(largo).replace(".", "'")
    
    volumen = float(resultado["medidas"]["Volumen"].replace(" L", ""))
    
    url_decathlon = generar_busqueda_decathlon(
        resultado["medidas"],
        resultado["tipo"]
    )
    
    st.write("## 🏄 Tablas similares:")
    
    st.markdown(
        f"""
        ### [🔎 Ver tablas recomendadas en Mundosurf]({url_decathlon})
        """
    )
      
    
#print(df['board_length'].head(20))