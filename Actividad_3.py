import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# 1. Base de Conocimiento Jurídica (Corpus de 8 Documentos Normativos)
base_conocimiento_justia = {
    "doc_ley1010": "Ley 1010 de 2006 (Acoso Laboral): Define el acoso laboral como toda conducta persistente y demostrable ejercida sobre un empleado para infundir miedo, intimidación, terror y angustia. Otorga al trabajador el derecho a denunciar ante el Comité de Convivencia, solicitar fuero de estabilidad laboral por 6 meses tras la denuncia y exigir indemnización equivalente a 140 salarios mínimos.",
    "doc_ley1257": "Ley 1257 de 2008 (Protección a la Mujer): Garantiza el derecho de las mujeres a una vida libre de violencia. Establece subsidios de habitación, estabilidad laboral reforzada en casos de violencia intrafamiliar y la obligación de las autoridades de brindar medidas de atención médica y psicológica inmediata.",
    "doc_cst_art57": "Código Sustantivo del Trabajo - Artículo 57: Obliga al empleador a conceder al trabajador las licencias necesarias para el ejercicio del sufragio, atención de calamidad doméstica debidamente comprobada y comisión sindical.",
    "doc_ley2191": "Ley 2191 de 2022 (Desconexión Laboral): Garantiza el derecho de los trabajadores a no recibir órdenes, llamadas o correos fuera de la jornada laboral establecida. Su vulneración constituye presunción de acoso laboral.",
    "doc_codigo_infancia": "Código de Infancia y Adolescencia - Artículo 129: Establece que la cuota alimentaria para menores de edad incluye vivienda, educación, salud, vestuario y recreación, calculada sobre el total de los ingresos del progenitor obligado.",
    "doc_ley2114": "Ley 2114 de 2021 (Licencia de Paternidad): Amplía la licencia remunerada de paternidad a dos semanas a cargo del sistema de seguridad social en salud.",
    "doc_decreto1072": "Decreto 1072 de 2015 (Reglamento del Sector Trabajo): Establece la obligación de implementar el Sistema de Gestión de Seguridad y Salud en el Trabajo (SG-SST) para prevenir riesgos laborales psicosociales.",
    "doc_sentencia_t045": "Sentencia T-045 de 2020 (Corte Constitucional): Reitera que las personas pertenecientes a la población migrante o en condición de vulnerabilidad extrema tienen derecho a la atención médica de urgencias e inscripción prioritaria en sistemas de protección social."
}

# 2. Carga directa de Modelo y Tokenizer
model_name = "deepset/xlm-roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 3. Función de Extracción de Respuestas con Inferencia Directa
def consultar_asistente_justia(pregunta_usuario, corpus):
    contexto_completo = " ".join(corpus.values())
    
    # Tokenización de la pregunta y el contexto
    inputs = tokenizer(pregunta_usuario, contexto_completo, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Obtención de los índices de inicio y fin con mayor probabilidad
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    
    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    respuesta_extraida = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    if not respuesta_extraida:
        respuesta_extraida = "No se encontró respuesta en la base de datos."

    # Identificación del documento fuente
    fuente_oficial = "Documento no identificado"
    for contenido in corpus.values():
        if respuesta_extraida and respuesta_extraida in contenido:
            fuente_oficial = contenido.split(":")[0]
            break

    return {
        "pregunta": pregunta_usuario,
        "respuesta_extraida": respuesta_extraida,
        "fuente_legal": fuente_oficial
    }

# 4. Prueba del Asistente Jurídico
pregunta_ejemplo = "¿Qué derechos tengo si soy víctima de acoso laboral?"
respuesta = consultar_asistente_justia(pregunta_ejemplo, base_conocimiento_justia)

print("=== CONSULTA AL ASISTENTE VIRTUAL JURÍDICO ===")
print(f"Pregunta Usuario:   {respuesta['pregunta']}")
print(f"Respuesta Generada: {respuesta['respuesta_extraida']}")
print(f"Fuente Citada:      {respuesta['fuente_legal']}")