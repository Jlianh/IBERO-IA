# =========================================
# APRENDIZAJE SUPERVISADO - ÁRBOL DE DECISIÓN
# =========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1. Cargar dataset
# -----------------------------
df = pd.read_csv("dataset_transporte.csv")

# -----------------------------
# 2. Codificar variables categóricas
# -----------------------------
le_origen = LabelEncoder()
le_destino = LabelEncoder()
le_transporte = LabelEncoder()

df["origen"] = le_origen.fit_transform(df["origen"])
df["destino"] = le_destino.fit_transform(df["destino"])
df["transporte"] = le_transporte.fit_transform(df["transporte"])

# -----------------------------
# 3. Variables
# -----------------------------
X = df[["origen", "destino", "distancia_km", "transporte", "hora", "transbordos"]]
y = df["tiempo_real_min"]

# -----------------------------
# 4. División de datos
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Modelo
# -----------------------------
modelo = DecisionTreeRegressor(max_depth=5)
modelo.fit(X_train, y_train)

# -----------------------------
# 6. Predicciones
# -----------------------------
predicciones = modelo.predict(X_test)

# -----------------------------
# 7. Evaluación
# -----------------------------
error = mean_squared_error(y_test, predicciones)

print("Predicciones:", predicciones)
print("Error (MSE):", error)

# -----------------------------
# 8. Predicción nueva
# -----------------------------
# Ejemplo: Portal Norte → Calle 100
nuevo = pd.DataFrame([{
    "origen": le_origen.transform(["Portal Norte"])[0],
    "destino": le_destino.transform(["Calle 100"])[0],
    "distancia_km": 6,
    "transporte": le_transporte.transform(["transmilenio"])[0],
    "hora": 8,
    "transbordos": 0
}])

print("Tiempo estimado:", modelo.predict(nuevo))