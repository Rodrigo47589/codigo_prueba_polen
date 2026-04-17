import numpy as np


def similitud_coseno(vec1, vec2):
    norma1 = np.linalg.norm(vec1)
    norma2 = np.linalg.norm(vec2)

    if norma1 == 0 or norma2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norma1 * norma2)


def top_k_similares(vector_consulta, vectores_base, rutas_base, k=3):
    resultados = []

    for vector, ruta in zip(vectores_base, rutas_base):
        score = similitud_coseno(vector_consulta, vector)
        resultados.append((ruta, score))

    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados[:k]