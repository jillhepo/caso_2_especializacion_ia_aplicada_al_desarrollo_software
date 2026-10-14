import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# 1. Generación Sintáctica Controlada de 100 Fragmentos Jurídicos
np.random.seed(42)

semillas_tematicas = {
    "Familia": [
        "Solicitud de fijación de cuota alimentaria para menores de edad.",
        "Demanda de divorcio por la causal de separación de cuerpos por más de dos años.",
        "Proceso de custodia, cuidado personal y regulación de visitas del menor.",
        "Acción de impugnación de la paternidad y prueba genética de ADN."
    ],
    "Laboral": [
        "Reclamación de liquidación de prestaciones sociales, cesantías e intereses.",
        "Demanda por despido injustificado de trabajador con fuero de salud.",
        "Cobro de indemnización moratoria por falta de pago del salario a la terminación.",
        "Acción tutelar por acoso laboral e intromisión en la jornada de descanso."
    ],
    "Civil": [
        "Proceso de restitución de inmueble arrendado por falta de pago del canon.",
        "Demanda de pertenencia por prescripción adquisitiva extraordinaria de dominio.",
        "Acción por responsabilidad civil extracontractual derivada de accidente.",
        "Proceso ejecutivo singular para el cobro de pagaré e intereses de mora."
    ],
    "Penal": [
        "Audiencia de imputación de cargos por el delito de hurto calificado y agravado.",
        "Denuncia penal por violencia intrafamiliar y solicitud de medida de protección.",
        "Solicitud de libertad condicional por cumplimiento del porcentaje de la pena.",
        "Proceso por fraude procesal y falsedad material en documento público."
    ]
}

fragmentos, etiquetas_reales = [], []
categorias = list(semillas_tematicas.keys())

# Generación de los 100 textos únicos
for i in range(100):
    cat_actual = categorias[i % 4]
    frase_base = np.random.choice(semillas_tematicas[cat_actual])
    variacion = f" (Expediente Interno N° {2024000 + i})"
    fragmentos.append(frase_base + variacion)
    etiquetas_reales.append(cat_actual)

print(f"Dataset de {len(fragmentos)} fragmentos jurídicos preparado para clustering.")

# 2. Extracción de Embeddings con Sentence Transformers
model_sentence = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model_sentence.encode(fragmentos, show_progress_bar=False)

# 3. Aplicación de K-Means Clustering (4 Cúmulos)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_predichos = kmeans.fit_predict(embeddings)

# 4. Reducción de Dimensionalidad con t-SNE a 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
embeddings_2d = tsne.fit_transform(embeddings)

# 5. Visualización de los Agrupamientos Temáticos
plt.figure(figsize=(9, 6))
colores = ['purple', 'green', 'blue', 'orange']

for i in range(n_clusters):
    puntos = embeddings_2d[cluster_predichos == i]
    plt.scatter(puntos[:, 0], puntos[:, 1], label=f'Cluster {i+1}', alpha=0.7, c=colores[i])

plt.title("Visualización t-SNE del Agrupamiento Semántico de Casos (100 Fragmentos)")
plt.xlabel("Dimensión t-SNE 1")
plt.ylabel("Dimensión t-SNE 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
# plt.show()
print("Gráfico t-SNE generado exitosamente.")