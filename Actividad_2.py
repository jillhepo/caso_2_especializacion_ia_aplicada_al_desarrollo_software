import spacy

# 1. Carga del modelo en español de spaCy
nlp = spacy.load("es_core_news_md")

# 2. Configuración de reglas personalizadas con EntityRuler
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    # Normas Jurídicas
    {"label": "NORMA_JURIDICA", "pattern": [{"LOWER": "ley"}, {"IS_DIGIT": True}, {"LOWER": "de"}, {"IS_DIGIT": True}]},
    {"label": "NORMA_JURIDICA", "pattern": [{"LOWER": "decreto"}, {"IS_DIGIT": True}, {"LOWER": "de"}, {"IS_DIGIT": True}]},
    {"label": "NORMA_JURIDICA", "pattern": [{"LOWER": "código"}, {"LOWER": "sustantivo"}, {"LOWER": "del"}, {"LOWER": "trabajo"}]},
    {"label": "NORMA_JURIDICA", "pattern": [{"LOWER": "código"}, {"LOWER": "de"}, {"LOWER": "infancia"}, {"LOWER": "y"}, {"LOWER": "adolescencia"}]},
    # Tipos de Violencia Sensibles (Desambiguación)
    {"label": "TIPO_VIOLENCIA", "pattern": [{"LOWER": "violencia"}, {"LOWER": "económica"}]},
    {"label": "TIPO_VIOLENCIA", "pattern": [{"LOWER": "violencia"}, {"LOWER": "patrimonial"}]},
    {"label": "TIPO_VIOLENCIA", "pattern": [{"LOWER": "violencia"}, {"LOWER": "intrafamiliar"}]},
    {"label": "TIPO_VIOLENCIA", "pattern": [{"LOWER": "violencia"}, {"LOWER": "física"}]},
    {"label": "TIPO_VIOLENCIA", "pattern": [{"LOWER": "violencia"}, {"LOWER": "psicológica"}]}
]
ruler.add_patterns(patterns)

# 3. Conjunto de 10 Sentencias Breves y Resoluciones Administrativas Simuladas
sentencias_dataset = [
    "1. En Bogotá el 15 de marzo de 2024, la Jueza María Rodríguez resolvió el caso dictando amparo bajo la Ley 1257 de 2008 a favor de la ciudadana Ana Martínez.",
    "2. Mediante Resolución 045 de 2023, el Ministerio del Trabajo impuso sanción por actos constitutivos de acoso laboral e incumplimiento del Código Sustantivo del Trabajo.",
    "3. La demandante alegó ser víctima de violencia económica y violencia patrimonial por parte de su cónyuge, el señor Carlos Benítez, el pasado 10 de enero de 2024.",
    "4. El Juzgado Tercero Penal decretó medida de aseguramiento invocando el Código de Procedimiento Penal por el delito de violencia intrafamiliar reiterada.",
    "5. El señor Juan Pérez solicitó la revisión de la liquidación de sus prestaciones sociales amparado en la Ley 100 de 1993 ante el Tribunal Superior.",
    "6. En fallo del 22 de agosto de 2023, la Corte observó vulneración a los derechos fundamentales debido a violencia física y violencia psicológica comprobadas.",
    "7. La Comisaría de Familia de Suba emitió medida de protección urgente regulada en la Ley 294 de 1996 ordenando el desalojo del agresor.",
    "8. La abogada Sofía Morales interpuso acción de tutela contra el Decreto 1072 de 2015 solicitando la reubicación laboral inmediata de un trabajador perjudicado.",
    "9. El 05 de mayo de 2024, el señor Roberto Gómez fue notificado de la apertura de proceso ejecutivo alimentario bajo el Código de Infancia y Adolescencia.",
    "10. La resolución administrativa determinó la ocurrencia de violencia patrimonial tras el embargo arbitrario de los bienes propios de la señora Luisa Fernández."
]

# 4. Procesamiento e Inspección de Entidades Extraídas
print("=== RESULTADOS DE EXTRACCIÓN NER SOBRE LAS 10 SENTENCIAS ===\n")
for idx, texto in enumerate(sentencias_dataset, 1):
    doc = nlp(texto)
    print(f"--- Texto {idx} ---")
    for ent in doc.ents:
        print(f"  Entidad: {ent.text:<35} | Etiqueta: {ent.label_}")
    print()