
import cv2
import os
import csv
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURACION DEL SISTEMA
# ==========================================

print("INICIANDO SISTEMA DE CONTEO DE PERSONAS...")

# Cargar inteligencia artificial
modelo = YOLO("yolo11n.pt")

# Abrir video
video = cv2.VideoCapture("video 1.mp4")

if not video.isOpened():
    print("ERROR: No se pudo abrir el video.")
    print("Revisa que video 1.mp4 este en la misma carpeta.")
    exit()

# ==========================================
# 2. CREAR CARPETAS Y ARCHIVOS
# ==========================================

carpeta_personas = "Personas_Detectadas"
carpeta_fotos = os.path.join(carpeta_personas, "Fotografias")

os.makedirs(carpeta_fotos, exist_ok=True)

archivo_csv = os.path.join(
    carpeta_personas,
    "registro_personas.csv"
)

# Crear archivo CSV si no existe
if not os.path.exists(archivo_csv):

    with open(archivo_csv, "w", newline="", encoding="utf-8-sig") as f:

        escritor = csv.writer(f)

        escritor.writerow([
            "Numero",
            "ID de seguimiento",
            "Fecha y hora",
            "Fotografia"
        ])

# ==========================================
# 3. VARIABLES
# ==========================================

personas_contadas = 0

posiciones_anteriores = {}

personas_registradas = set()

linea_x = None

# ==========================================
# 4. PROCESAR VIDEO
# ==========================================

while True:

    ret, frame = video.read()

    if not ret:
        break

    alto, ancho = frame.shape[:2]

    # Crear linea en el centro
    if linea_x is None:
        linea_x = ancho // 2

    # Detectar y seguir personas
    resultados = modelo.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    # Dibujar linea amarilla
    cv2.line(
        frame,
        (linea_x, 0),
        (linea_x, alto),
        (0, 255, 255),
        3
    )

    cajas = resultados[0].boxes

    # ==========================================
    # 5. DETECTAR PERSONAS
    # ==========================================

    if cajas is not None and cajas.id is not None:

        coordenadas = cajas.xyxy.cpu().numpy()

        ids = cajas.id.int().cpu().tolist()

        for caja, persona_id in zip(coordenadas, ids):

            x1, y1, x2, y2 = map(int, caja)

            # Centro de la persona
            centro_x = (x1 + x2) // 2
            centro_y = (y1 + y2) // 2

            # Dibujar rectangulo
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Mostrar ID
            cv2.putText(
                frame,
                f"Persona ID: {persona_id}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Dibujar centro
            cv2.circle(
                frame,
                (centro_x, centro_y),
                5,
                (0, 0, 255),
                -1
            )

            # ==========================================
            # 6. COMPROBAR CRUCE DE LINEA
            # ==========================================

            if persona_id in posiciones_anteriores:

                anterior = posiciones_anteriores[persona_id]

                cruzo_derecha = (
                    anterior < linea_x <= centro_x
                )

                cruzo_izquierda = (
                    anterior > linea_x >= centro_x
                )

                if (
                    (cruzo_derecha or cruzo_izquierda)
                    and persona_id not in personas_registradas
                ):

                    personas_contadas += 1

                    personas_registradas.add(persona_id)

                    # ==========================================
                    # 7. GUARDAR FECHA Y FOTOGRAFIA
                    # ==========================================

                    fecha_hora = datetime.now().strftime(
                        "%d-%m-%Y_%H-%M-%S"
                    )

                    nombre_foto = (
                        f"persona_{personas_contadas}_"
                        f"ID_{persona_id}.jpg"
                    )

                    ruta_foto = os.path.join(
                        carpeta_fotos,
                        nombre_foto
                    )

                    # Recortar persona detectada
                    margen = 15

                    px1 = max(0, x1 - margen)
                    py1 = max(0, y1 - margen)
                    px2 = min(ancho, x2 + margen)
                    py2 = min(alto, y2 + margen)

                    recorte = frame[py1:py2, px1:px2]

                    # Guardar fotografia
                    foto_guardada = False

                    if recorte.size > 0:
                        foto_guardada = cv2.imwrite(
                            ruta_foto,
                            recorte
                        )

                    # Guardar registro CSV
                    with open(
                        archivo_csv,
                        "a",
                        newline="",
                        encoding="utf-8-sig"
                    ) as f:

                        escritor = csv.writer(f)

                        escritor.writerow([
                            personas_contadas,
                            persona_id,
                            fecha_hora,
                            nombre_foto if foto_guardada else "No guardada"
                        ])

                    print(
                        f"PERSONA {personas_contadas} REGISTRADA "
                        f"- ID: {persona_id}"
                    )

            # Actualizar posicion
            posiciones_anteriores[persona_id] = centro_x

    # ==========================================
    # 8. MOSTRAR CONTADOR EN PANTALLA
    # ==========================================

    cv2.rectangle(
        frame,
        (10, 10),
        (370, 75),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"PERSONAS: {personas_contadas}",
        (20, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # Mostrar video
    cv2.imshow(
        "SISTEMA DE CONTEO DE PERSONAS",
        frame
    )

    # Salir presionando Q
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# ==========================================
# 9. FINALIZAR SISTEMA
# ==========================================

video.release()

cv2.destroyAllWindows()

print("\n================================")
print("VIDEO FINALIZADO")
print("TOTAL DE PERSONAS:", personas_contadas)
print("REGISTROS GUARDADOS EN:", archivo_csv)
print("FOTOGRAFIAS GUARDADAS EN:", carpeta_fotos)
print("================================")
