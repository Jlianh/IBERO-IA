# =========================================
# APRENDIZAJE NO SUPERVISADO - K-MEANS
# =========================================

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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
# 3. Variables (SIN target)
# -----------------------------
X = df[["distancia_km", "hora", "transbordos", "transporte"]]

# -----------------------------
# 4. Modelo K-Means
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# -----------------------------
# 5. Resultados
# -----------------------------
print(df[["distancia_km", "hora", "transbordos", "transporte", "cluster"]].head())

# -----------------------------
# 6. Análisis de clusters
# -----------------------------
print("\nPromedio por cluster:")
print(df.groupby("cluster")[["distancia_km", "hora", "transbordos"]].mean())

# -----------------------------
# 7. Visualización
# -----------------------------
plt.scatter(df["distancia_km"], df["hora"], c=df["cluster"])
plt.xlabel("Distancia (km)")
plt.ylabel("Hora")
plt.title("Clusters de rutas - TransMilenio")
plt.show()