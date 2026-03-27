import heapq

grafo = {
    "A": [("B", 5, "bus"), ("C", 10, "metro")],
    "B": [("D", 3, "bus"), ("E", 7, "metro")],
    "C": [("E", 2, "metro")],
    "D": [("F", 4, "bus")],
    "E": [("F", 1, "metro")],
    "F": []
}

# Heurística (estimación al destino)
heuristica = {
    "A": 10,
    "B": 8,
    "C": 5,
    "D": 4,
    "E": 2,
    "F": 0
}

def costo_reglas(tipo):
    if tipo == "metro":
        return -1
    return 1

def a_ruta(origen, destino):
    cola = [(0, origen, [], None)]  # (costo, nodo, camino, tipo_anterior)
    visitados = set()

    while cola:
        costo, nodo, camino, tipo_anterior = heapq.heappop(cola)

        if nodo in visitados:
            continue

        camino = camino + [nodo]
        visitados.add(nodo)

        if nodo == destino:
            return costo, camino

        for vecino, peso, tipo in grafo[nodo]:
            if vecino not in visitados:
                penalizacion = costo_reglas(tipo)

                if tipo_anterior and tipo != tipo_anterior:
                    penalizacion += 2  # transbordo

                nuevo_costo = costo + peso + penalizacion + heuristica[vecino]

                heapq.heappush(
                    cola,
                    (nuevo_costo, vecino, camino, tipo)
                )

    return float("inf"), []

# Prueba
costo, ruta = a_ruta("A", "F")
print("Ruta óptima:", ruta)
print("Costo:", costo)




