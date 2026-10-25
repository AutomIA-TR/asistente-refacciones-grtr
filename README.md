# Asistente de refacciones GRTR — fase 1

Chatbot basado en reglas para atender a clientes de una distribuidora de refacciones automotrices:
busca piezas por tipo y vehículo, da precios, revisa compatibilidad, arma una cotización preliminar
y responde dudas de envío, garantía, formas de pago y contacto.

Proyecto de la Maestría en Inteligencia Artificial (Universidad Tecmilenio), curso *Desarrollo de
aplicaciones con inteligencia artificial*. Autor: Jaime Eduardo Silva Navarro.

## Cómo correrlo

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

## Sobre los datos

**El catálogo es sintético.** Referencias, aplicaciones vehiculares y precios son fabricados; conserva
sólo la estructura de familias y líneas del catálogo real y niveles de precio aproximados. Ninguna
referencia coincide con una real. Los textos de envíos, garantía, formas de pago y horario son de
ejemplo (marcados `"ejemplo": true` en la base) hasta validarlos con la política de la empresa.

## Fases siguientes

Procesamiento de lenguaje natural para la identificación de intenciones, clasificación de imágenes
(foto de la pieza → familia) e integración web con el sitio de la empresa para la auto-cotización.
