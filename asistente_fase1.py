"""Código del chatbot de reglas de la fase 1 (Avance de proyecto 1), copiado sin cambios de las celdas 6, 8, 10 y 12
de asistente_refacciones.ipynb para que la fase 2 lo importe. El notebook de la fase 1 no se modifica."""

import re
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


def sin_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


class Preprocesador:
    """Convierte un mensaje en raíces comparables con las reglas."""

    def __init__(self, protegidas=()):
        self.stemmer = SnowballStemmer("spanish")
        vacias = {sin_acentos(p) for p in stopwords.words("spanish")}
        self.vacias = vacias - set(protegidas)

    def normalizar(self, texto: str) -> str:
        return re.sub(r"[^a-z0-9\s-]", " ", sin_acentos(texto.lower()))

    def tokens(self, texto: str) -> list[str]:
        return [t for t in word_tokenize(self.normalizar(texto), language="spanish") if re.search(r"[a-z0-9]", t)]

    def raices(self, texto: str, quitar_vacias: bool = True) -> list[str]:
        return [self.stemmer.stem(t) for t in self.tokens(texto) if not (quitar_vacias and t in self.vacias)]


import csv
import json


class BaseConocimiento:
    def __init__(self, ruta_base="datos/base_conocimiento.json", ruta_catalogo="datos/catalogo_sintetico.csv"):
        self.ruta_base = ruta_base
        self.ruta_catalogo = ruta_catalogo

    def leer(self):
        with open(self.ruta_base, encoding="utf-8") as f:
            base = json.load(f)
        with open(self.ruta_catalogo, newline="", encoding="utf-8") as f:
            catalogo = {}
            for p in csv.DictReader(f):
                p["precio"] = float(p["precio"]) if p["precio"] else None   # sin precio: None, nunca 0
                p["aplicaciones"] = [
                    {"marca": m, "modelo": mo, "desde": int(d), "hasta": int(h)}
                    for m, mo, d, h in (a.split("|") for a in p["aplicaciones"].split(";") if a)
                ]
                catalogo[p["referencia"]] = p
        return base, catalogo


NUMEROS = {"un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
           "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}


class ClasificadorReglas:
    def __init__(self, base: dict, catalogo: dict):
        self.base, self.catalogo = base, catalogo
        # las palabras clave no se tratan como vacías aunque estén en la lista de NLTK ("hay", "tienen")
        palabras = {w for i in base["intenciones"] for w in i["palabras"]}
        self.pre = Preprocesador(protegidas=palabras)
        self.reglas = {
            i["nombre"]: {
                "prioridad": i["prioridad"],
                "palabras": {self.pre.stemmer.stem(w) for w in i["palabras"]},
                "frases": [tuple(self.pre.raices(fr, quitar_vacias=False)) for fr in i["frases"]],
                "dispara_con": set(i["dispara_con"]),
                "requiere": set(i["requiere"]),
            }
            for i in base["intenciones"]
        }
        # sinónimos de familia como secuencias de raíces, del más largo al más corto
        self.sinonimos = sorted(
            ((tuple(self.pre.raices(s)), cod) for cod, lista in base["familias"].items() for s in lista),
            key=lambda x: -len(x[0]))
        self.genericos = {self.pre.stemmer.stem(g): cods for g, cods in base["genericos"].items()}
        self.vehiculos = {v: clave for clave, variantes in base["vehiculos"].items() for v in variantes}
        self.prefijos = {r.split("-")[0].lower() for r in catalogo}

    @staticmethod
    def contiene(secuencia, sub) -> bool:
        n = len(sub)
        return n > 0 and any(tuple(secuencia[i:i + n]) == sub for i in range(len(secuencia) - n + 1))

    def entidades(self, texto: str) -> dict:
        norm = self.pre.normalizar(texto)
        refs = []
        patron = r"(?:\b(\d{1,3})|\b(" + "|".join(NUMEROS) + r"))?\s*(?:pzas?|piezas?|unidades?)?\s*" \
                 r"(?:de\s+(?:la|el|las|los)\s+)?\b([a-z]{2})-?(\d{4})\b"
        for m in re.finditer(patron, norm):
            if m.group(3) in self.prefijos:
                cantidad = int(m.group(1)) if m.group(1) else NUMEROS.get(m.group(2), 1)
                refs.append((cantidad, f"{m.group(3).upper()}-{m.group(4)}"))
        resto = re.sub(r"\b[a-z]{2}-?\d{4}\b", " ", norm)          # sin referencias para buscar año
        anio = re.search(r"\b(19[89]\d|20[0-3]\d)\b", resto)
        tokens = self.pre.tokens(resto)
        vehiculo = next((self.vehiculos[t] for t in tokens if t in self.vehiculos), None)
        # la familia se busca sin el nombre del vehículo ("march" es modelo y también raíz de "marcha")
        raices = [self.pre.stemmer.stem(t) for t in tokens if t not in self.pre.vacias and t not in self.vehiculos]
        familia = next((cod for sec, cod in self.sinonimos if self.contiene(raices, sec)), None)
        generico = None if familia else next((c for r, c in self.genericos.items() if r in raices), None)
        cantidad_suelta = next((int(t) for t in tokens if t.isdigit() and int(t) < 100), None)
        return {"refs": refs, "familia": familia, "generico": generico, "vehiculo": vehiculo,
                "anio": int(anio.group(1)) if anio else None, "cantidad": cantidad_suelta}

    def identificar(self, texto: str):
        ent = self.entidades(texto)
        contenido = self.pre.raices(texto)
        todas = self.pre.raices(texto, quitar_vacias=False)
        presentes = {k for k, v in (("referencia", ent["refs"]), ("familia", ent["familia"]),
                                    ("generico", ent["generico"]), ("vehiculo", ent["vehiculo"])) if v}
        candidatas = []
        for nombre, r in self.reglas.items():
            puntaje = sum(1 for x in contenido if x in r["palabras"]) + 2 * sum(1 for fr in r["frases"] if self.contiene(todas, fr))
            if (puntaje > 0 or r["dispara_con"] & presentes) and r["requiere"] <= presentes:
                candidatas.append((r["prioridad"], -puntaje, nombre))
        intencion = min(candidatas)[2] if candidatas else "fuera_de_alcance"
        return intencion, ent


def pesos(x):
    return f"${x:,.2f}"


class Chatbot:
    def __init__(self, base: dict, catalogo: dict):
        self.base, self.catalogo = base, catalogo
        self.clasificador = ClasificadorReglas(base, catalogo)
        self.textos = {i["nombre"]: i.get("respuesta", "") for i in base["intenciones"]}
        self.nombres = base["nombres"]
        self.reiniciar()

    def reiniciar(self):
        self.ultima_ref = None
        self.cotizacion = {}

    # --- utilidades -------------------------------------------------------
    def ficha(self, p):
        precio = pesos(p["precio"]) if p["precio"] else "precio por confirmar con un asesor"
        return f"{p['referencia']} · {p['descripcion']} · {p['linea']} · {precio}"

    def vehiculo_txt(self, ent):
        if not ent["vehiculo"]:
            return ""
        marca, modelo = ent["vehiculo"].split("|")
        return f"{marca} {modelo}" + (f" {ent['anio']}" if ent["anio"] else "")

    def aplica(self, p, ent):
        marca, modelo = ent["vehiculo"].split("|")
        return any(a["modelo"] == modelo and (ent["anio"] is None or a["desde"] <= ent["anio"] <= a["hasta"])
                   for a in p["aplicaciones"])

    def filtrar(self, ent):
        lista = [p for p in self.catalogo.values() if p["familia"] == ent["familia"]]
        if ent["vehiculo"]:
            lista = [p for p in lista if self.aplica(p, ent)]
        return sorted(lista, key=lambda p: (p["precio"] is None, p["precio"] or 0))

    # --- intenciones que consultan el catálogo ---------------------------
    def r_buscar_pieza(self, ent):
        if ent["generico"]:
            nombres = [self.nombres[c] for c in ent["generico"]]
            opciones = ", ".join(nombres[:-1]) + " o " + nombres[-1]
            return f"Tengo varios tipos. ¿Cuál buscas: {opciones}?"
        if not ent["familia"]:
            return f"¿Qué pieza buscas para tu {self.vehiculo_txt(ent)}? Por ejemplo: bomba de gasolina, bobina, sensor de oxígeno."
        lista = self.filtrar(ent)
        nombre = self.nombres[ent["familia"]]
        if ent["vehiculo"]:
            if not lista:
                return f"No encontré {nombre} para {self.vehiculo_txt(ent)} en el catálogo. Un asesor puede revisarlo contigo."
            if len(lista) == 1:
                self.ultima_ref = lista[0]["referencia"]
            lineas = "\n".join("  • " + self.ficha(p) for p in lista[:3])
            mas = f"\n  …y {len(lista) - 3} más." if len(lista) > 3 else ""
            cuantas = "1 opción" if len(lista) == 1 else f"{len(lista)} opciones"
            return f"Encontré {cuantas} de {nombre} para {self.vehiculo_txt(ent)}:\n{lineas}{mas}"
        lineas = "\n".join("  • " + self.ficha(p) for p in lista[:3])
        return (f"Manejo {len(lista)} referencias de {nombre}. Algunas:\n{lineas}\n"
                f"Dime el modelo y año de tu auto para mostrarte las que le quedan.")

    def r_consultar_referencia(self, ent):
        if not ent["refs"]:
            return "Compárteme la referencia (por ejemplo, BG-1234) o dime qué pieza buscas y para qué vehículo."
        _, ref = ent["refs"][0]
        p = self.catalogo.get(ref)
        if not p:
            return f"No encontré la referencia {ref}. Revisa el código, o dime qué pieza y para qué vehículo la buscas."
        self.ultima_ref = ref
        apps = "; ".join(f"{a['marca']} {a['modelo']} {a['desde']}-{a['hasta']}" for a in p["aplicaciones"])
        return f"{self.ficha(p)}\nAplica para: {apps}."

    def r_precio(self, ent):
        if ent["refs"] or (not ent["familia"] and self.ultima_ref):
            ref = ent["refs"][0][1] if ent["refs"] else self.ultima_ref
            p = self.catalogo.get(ref)
            if not p:
                return f"No encontré la referencia {ref}. Revisa el código, o dime qué pieza y para qué vehículo la buscas."
            self.ultima_ref = ref
            if p["precio"] is None:
                return f"La {ref} no tiene precio publicado; un asesor te la cotiza. ({p['descripcion']})"
            return f"La {ref} ({p['descripcion']}, {p['linea']}) cuesta {pesos(p['precio'])}."
        if ent["familia"]:
            lista = [p for p in self.filtrar(ent) if p["precio"] is not None]
            nombre = self.nombres[ent["familia"]]
            donde = f" para {self.vehiculo_txt(ent)}" if ent["vehiculo"] else ""
            if not lista:
                return f"No tengo precios publicados de {nombre}{donde}. Un asesor puede cotizarte."
            minimo, maximo = lista[0]["precio"], lista[-1]["precio"]
            if len(lista) == 1:
                self.ultima_ref = lista[0]["referencia"]
                return f"Hay una opción de {nombre}{donde}: {self.ficha(lista[0])}."
            return (f"{nombre.capitalize()}{donde}: de {pesos(minimo)} a {pesos(maximo)} ({len(lista)} referencias con precio). "
                    f"La más económica: {self.ficha(lista[0])}.")
        return "¿De qué pieza quieres el precio? Dame la referencia o el tipo de pieza y tu vehículo."

    def r_compatibilidad(self, ent):
        ref = ent["refs"][0][1] if ent["refs"] else self.ultima_ref
        if ent["familia"] and not ent["refs"]:
            return self.r_buscar_pieza(ent)
        if not ref:
            return f"¿Qué pieza quieres revisar para tu {self.vehiculo_txt(ent)}? Dame la referencia o el tipo de pieza."
        p = self.catalogo.get(ref)
        if not p:
            return f"No encontré la referencia {ref}. Revisa el código, o dime qué pieza y para qué vehículo la buscas."
        self.ultima_ref = ref
        apps = "; ".join(f"{a['marca']} {a['modelo']} {a['desde']}-{a['hasta']}" for a in p["aplicaciones"])
        if self.aplica(p, ent):
            return (f"Sí: según el catálogo, la {ref} aplica para {self.vehiculo_txt(ent)} ({apps}). "
                    f"Confírmalo con el número de parte de tu pieza o con un asesor.")
        nombre = self.nombres[p["familia"]]
        return (f"Según el catálogo, la {ref} no aparece para {self.vehiculo_txt(ent)}; aplica para {apps}. "
                f"Escribe, por ejemplo, «busco {nombre} para {self.vehiculo_txt(ent)}».")

    def r_cotizar(self, ent, texto):
        norm = self.clasificador.pre.normalizar(texto)
        if re.search(r"\b(borra|borrar|cancela|cancelar|vacia|vaciar|limpia|limpiar)\b", norm):
            self.cotizacion = {}
            return "Listo, borré tu cotización."
        avisos = []
        # sin referencia en el mensaje, sólo un verbo de agregar usa la última pieza consultada
        verbo = re.search(r"\b(cotiza(?!cion)\w*|agrega\w*|anade\w*|llevo|compro|comprar)\b", norm)
        implicita = [(ent["cantidad"] or 1, self.ultima_ref)] if (verbo and self.ultima_ref) else []
        agregar = ent["refs"] or implicita
        for cantidad, ref in agregar:
            p = self.catalogo.get(ref)
            if not p:
                avisos.append(f"No encontré la {ref}; no la agregué.")
            elif p["precio"] is None:
                avisos.append(f"La {ref} no tiene precio publicado; no la agregué y un asesor te la cotiza.")
            else:
                self.cotizacion[ref] = self.cotizacion.get(ref, 0) + cantidad
                self.ultima_ref = ref
        if not self.cotizacion:
            texto_avisos = (" ".join(avisos) + " ") if avisos else ""
            return texto_avisos + "Tu cotización está vacía. Escribe, por ejemplo, «cotízame 2 BE-1126»."
        lineas, total = [], 0.0
        for ref, cantidad in self.cotizacion.items():
            p = self.catalogo[ref]
            importe = cantidad * p["precio"]
            total += importe
            lineas.append(f"  {cantidad} × {ref} {p['descripcion'][:40]} · {pesos(p['precio'])} c/u = {pesos(importe)}")
        partes = (avisos and [" ".join(avisos)]) or []
        partes += ["Cotización preliminar:", *lineas, f"  Total estimado: {pesos(total)}", self.base["nota_cotizacion"]]
        return "\n".join(partes)

    # --- orquestación -----------------------------------------------------
    def responder(self, texto: str):
        intencion, ent = self.clasificador.identificar(texto)
        if intencion == "cotizar":
            respuesta = self.r_cotizar(ent, texto)
        elif hasattr(self, "r_" + intencion):
            respuesta = getattr(self, "r_" + intencion)(ent)
        elif intencion == "catalogo":
            conteo = {}
            for p in self.catalogo.values():
                conteo[p["familia"]] = conteo.get(p["familia"], 0) + 1
            lista = ", ".join(f"{self.nombres[c]} ({n})" for c, n in conteo.items())
            respuesta = f"Manejo refacciones de motor, inyección, encendido y dirección: {lista}."
        else:
            respuesta = self.textos[intencion].format(familias=", ".join(self.nombres.values()))
        return intencion, respuesta
