# ruff: noqa: I001
from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# 1. Generación de los 200 Fragmentos Jurídicos
plantillas_juridicas = {
    "Familia": [
        "Se solicita la fijación de cuota alimentaria para el menor de edad según el Código de Infancia y Adolescencia.",
        "Demanda de divorcio por la causal de cesación de los efectos civiles de matrimonio religioso.",
        "Solicitud de regulación del régimen de visitas y custodia compartida de los hijos menores.",
        "Proceso de impugnación de la paternidad y filiación extramatrimonial presentado por la parte demandante.",
        "Reclamación de embargos sobre el salario por mora e incumplimiento en la obligación alimentaria."
    ],
    "Laboral": [
        "El trabajador demanda la liquidación de prestaciones sociales, cesantías e intereses no cancelados.",
        "Demanda ordinaria laboral por despido injustificado de empleado en fuero de maternidad o fuero de salud.",
        "Solicitud de reconocimiento y reajuste pensional conforme al régimen de prima media de la Ley 100 de 1993.",
        "Reclamación de indemnización moratoria por falta de pago de las acreencias laborales a la terminación del contrato.",
        "Acción legal por acoso laboral reiterado y persecución sistemática en el puesto de trabajo."
    ],
    "Civil": [
        "Demanda de restitución de inmueble arrendado por mora consecutiva en el pago de los cánones de arrendamiento.",
        "Proceso declarativo de pertenencia por prescripción adquisitiva extraordinaria de dominio sobre bien inmueble.",
        "Acción de resolución de contrato de compraventa por vicios ocultos detectados en la estructura de la propiedad.",
        "Proceso ejecutivo para el cobro de título valor tipo letra de cambio incumplido en la fecha de vencimiento.",
        "Demanda por responsabilidad civil extracontractual por daños y perjuicios ocasionados en accidente de tránsito."
    ],
    "Penal": [
        "Imposición de medida de aseguramiento en centro carcelario por la presunta comisión del delito de hurto calificado.",
        "Audiencia de imputación de cargos y legalización de captura por el delito de homicidio en grado de tentativa.",
        "Denuncia penal formal por el delito de violencia intrafamiliar y agresiones físicas sistemáticas.",
        "Solicitud de libertad condicional por cumplimiento de las dos terceras partes de la pena privativa de la libertad.",
        "Proceso penal por defraudación y estafa agravada mediante el uso de documentos falsos en transacción bancaria."
    ]
}

conectores_contexto = [
    "En la ciudad de Bogotá, ", "Ante el despacho judicial competente, ", "En la revisión preliminar del caso, ",
    "Mediante apoderado judicial, ", "En la audiencia de conciliación, ", "Dentro del expediente radicado, "
]

textos, etiquetas = [], []
mapeo_clases = {"Familia": 0, "Laboral": 1, "Civil": 2, "Penal": 3}

# Generación iterativa hasta alcanzar exactamente 200 fragmentos únicos
count = 0
np.random.seed(42)

while count < 200:
    for cat, frases in plantillas_juridicas.items():
        if count >= 200:
            break
        base = np.random.choice(frases)
        prefijo = np.random.choice(conectores_contexto) if np.random.rand() > 0.4 else ""
        sufijo = f" (Radicado N° {1000 + count})."
        
        texto_final = f"{prefijo}{base}{sufijo}"
        textos.append(texto_final)
        etiquetas.append(mapeo_clases[cat])
        count += 1

df_dataset = pd.DataFrame({"text": textos, "label": etiquetas})
print(f"Dataset generado exitosamente con {len(df_dataset)} fragmentos jurídicos.")

# 2. Tokenización y Preprocesamiento
model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

dataset_hf = Dataset.from_pandas(df_dataset)
tokenized_dataset = dataset_hf.map(tokenize_function, batched=True)

# División en Entrenamiento (80%) y Evaluación (20%) -> 160 entrenamiento / 40 prueba
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

# 3. Función para Métricas (Accuracy, F1, Precision, Recall)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision": precision,
        "recall": recall
    }

# 4. Configuración del Modelo y Fine-Tuning
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

training_args = TrainingArguments(
    output_dir="./results_beto_200_fragmentos",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    #logging_dir='./logs',
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    compute_metrics=compute_metrics,
)

# Para ejecutar el entrenamiento en el entorno local/Colab:
trainer.train()
print("Modelo BETO preparado y entrenable sobre el dataset de 200 muestras.")