import cv2
import numpy as np
from pathlib import Path


def circularidad(contorno):
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    if perimetro == 0:
        return 0
    return 4 * np.pi * area / (perimetro * perimetro)


def leer_imagen_segura(ruta_imagen):
    archivo_bytes = np.fromfile(str(ruta_imagen), dtype=np.uint8)
    if archivo_bytes.size == 0:
        return None
    return cv2.imdecode(archivo_bytes, cv2.IMREAD_COLOR)


def segmentar_una_imagen(ruta_imagen, ruta_salida="salida"):
    ruta_imagen = Path(ruta_imagen)
    ruta_salida = Path(ruta_salida)
    ruta_salida.mkdir(parents=True, exist_ok=True)

    img = leer_imagen_segura(ruta_imagen)
    if img is None:
        print(f"Error leyendo imagen: {ruta_imagen}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Detectar zona útil del microscopio
    _, thresh_circulo = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contornos_circulo, _ = cv2.findContours(
        thresh_circulo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos_circulo:
        print(f"No se detectó la zona del microscopio: {ruta_imagen.name}")
        return None

    contorno_circulo = max(contornos_circulo, key=cv2.contourArea)

    mask_microscopio = np.zeros_like(gray)
    cv2.drawContours(mask_microscopio, [contorno_circulo], -1, 255, -1)

    # 2. Detectar color rosado
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_pink = np.array([140, 30, 80])
    upper_pink = np.array([179, 255, 255])

    mask_color = cv2.inRange(hsv, lower_pink, upper_pink)
    mask_color = cv2.bitwise_and(mask_color, mask_color, mask=mask_microscopio)

    # 3. Limpieza morfológica
    kernel = np.ones((5, 5), np.uint8)
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel)
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel)

    # 4. Buscar candidatos
    contornos, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        print(f"No se detectó polen: {ruta_imagen.name}")
        return None

    # 5. Elegir mejor contorno: grande + circular
    mejor_contorno = None
    mejor_score = 0

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 150:
            continue

        circ = circularidad(cnt)
        score = area * circ

        if score > mejor_score:
            mejor_score = score
            mejor_contorno = cnt

    if mejor_contorno is None:
        print(f"No se encontró un candidato válido: {ruta_imagen.name}")
        return None

    # 6. Recorte usando círculo envolvente
    (cx, cy), radio = cv2.minEnclosingCircle(mejor_contorno)

    cx = int(cx)
    cy = int(cy)
    radio = int(radio)

    pad = 25  # aquí controlas el margen del recorte
    x1 = max(cx - radio - pad, 0)
    y1 = max(cy - radio - pad, 0)
    x2 = min(cx + radio + pad, img.shape[1])
    y2 = min(cy + radio + pad, img.shape[0])

    recorte = img[y1:y2, x1:x2]

    # 7. Imagen debug
    debug = img.copy()
    cv2.drawContours(debug, [mejor_contorno], -1, (0, 255, 0), 2)
    cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 8. Guardar
    nombre_base = ruta_imagen.stem

    cv2.imwrite(str(ruta_salida / f"{nombre_base}_mask_microscopio.png"), mask_microscopio)
    cv2.imwrite(str(ruta_salida / f"{nombre_base}_mask.png"), mask_color)
    cv2.imwrite(str(ruta_salida / f"{nombre_base}_debug.png"), debug)
    cv2.imwrite(str(ruta_salida / f"{nombre_base}_recorte.png"), recorte)

    print(f"Segmentado: {ruta_imagen.name}")

    return {
        "ruta_original": str(ruta_imagen),
        "recorte": str(ruta_salida / f"{nombre_base}_recorte.png"),
        "debug": str(ruta_salida / f"{nombre_base}_debug.png"),
        "bbox": (x1, y1, x2, y2)
    }