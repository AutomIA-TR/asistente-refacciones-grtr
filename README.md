# Asistente de refacciones GRTR — fases 1 y 2

Chatbot basado en reglas para atender a clientes de una distribuidora de refacciones automotrices:
busca piezas por tipo y vehículo, da precios, revisa compatibilidad, arma una cotización preliminar
y responde dudas de envío, garantía, formas de pago y contacto.

Proyecto de la Maestría en Inteligencia Artificial (Universidad Tecmilenio), curso *Desarrollo de
aplicaciones con inteligencia artificial*. Autor: Jaime Eduardo Silva Navarro.

## Fase 2: PLN, visión y primera integración

`asistente_fase2.ipynb` agrega un clasificador de intenciones (preprocesamiento con NLTK, bolsa de palabras o TF-IDF,
Naive Bayes o SVM lineal), un clasificador de imágenes de piezas (ResNet50 con transferencia de aprendizaje y
*fine-tuning*) y un asistente que atiende texto y fotos con el mismo estado de conversación.

1. Abre `asistente_fase2.ipynb` en Google Colab con un entorno de **GPU** (Entorno de ejecución → Cambiar tipo).
2. Las fotos se bajan de Kaggle con `kagglehub` («50 Types of Car Parts», `gpiosenka/car-parts-40-classes`, licencia
   Apache 2.0); no se guardan en este repositorio. Si Kaggle pide credenciales, guárdalas en los Secretos de Colab
   (`KAGGLE_USERNAME` y `KAGGLE_KEY`, o `KAGGLE_API_TOKEN`) con acceso del notebook activado.
3. Entorno de ejecución → Ejecutar todo. La sección 15 resume las métricas de texto, de imagen y de la integración.

## Cómo correr la fase 1

1. Abre `asistente_refacciones.ipynb` en Google Colab (Archivo → Abrir cuaderno → GitHub →
   `AutomIA-TR/asistente-refacciones-grtr`). La celda 2 clona este repositorio para tener los datos.
2. Entorno de ejecución → Ejecutar todo. La celda 8 corre las diez conversaciones de prueba y
   termina con `Todo bien.` o `FALLARON N`; la 9, las pruebas de límite; la 10 abre un chat interactivo.

Requiere Python 3.10 o superior y NLTK (`punkt_tab` y `stopwords`, que el notebook descarga).

## Contenido

| Archivo | Qué es |
|---|---|
| `asistente_refacciones.ipynb` | Preprocesamiento con NLTK, reglas de intención, respuestas y pruebas |
| `datos/base_conocimiento.json` | 14 intenciones con sus palabras clave, frases, prioridad y respuestas; familias, vehículos |
| `datos/catalogo_sintetico.csv` | 248 productos de 16 familias |
| `pruebas/conversaciones.json` | 10 conversaciones funcionales y 3 pruebas de límite, con la intención esperada por turno |
| `asistente_fase2.ipynb` | Fase 2: preprocesamiento, cuatro clasificadores de texto, ResNet50, integración texto-imagen |
| `asistente_fase1.py` | El chatbot de la fase 1, copiado sin cambios de su notebook para que la fase 2 lo importe |
| `datos/intenciones_entrenamiento.json` | 420 frases sintéticas de clientes, 30 por intención |
| `pruebas/frases_limite_fase2.json` | 23 frases de límite: ortografía, lenguaje coloquial, choques de palabras y piezas fuera del catálogo |

## Sobre los datos

**El catálogo es sintético.** Referencias, aplicaciones vehiculares y precios son fabricados; conserva
sólo la estructura de familias y líneas del catálogo real y niveles de precio aproximados. Ninguna
referencia coincide con una real. Los textos de envíos, garantía, formas de pago y horario son de
ejemplo (marcados `"ejemplo": true` en la base) hasta validarlos con la política de la empresa.
Las frases de entrenamiento de la fase 2 también son sintéticas: no vienen de conversaciones reales.

## Fases siguientes

Integración web con el sitio de la empresa para la auto-cotización.
