from datasets import load_dataset
import pandas as pd

# 1. Cargar dataset WikiQA desde Hugging Face
dataset = load_dataset("wiki_qa")

# 2. Convertir el conjunto de prueba a DataFrame
df = pd.DataFrame(dataset["test"])

# 3. Ver columnas disponibles
print("Columnas del dataset:")
print(df.columns)

# 4. Ver primeras filas
print("\nPrimeras filas:")
print(df.head())

# 5. Filtrar preguntas que tienen una respuesta correcta
df_validas = df[df["label"] == 1].copy()

# 6. Eliminar preguntas repetidas
df_validas = df_validas.drop_duplicates(subset=["question"])

# 7. Seleccionar 50 preguntas aleatorias
df_50 = df_validas.sample(n=50, random_state=42)

# 8. Crear tabla final
preguntas_eval = df_50[["question", "answer"]].reset_index(drop=True)
preguntas_eval.index = preguntas_eval.index + 1
preguntas_eval = preguntas_eval.reset_index()
preguntas_eval.columns = ["id", "question", "expected_answer"]

# 9. Guardar archivo CSV
preguntas_eval.to_csv("50_preguntas_wikiqa.csv", index=False, encoding="utf-8-sig")

print("\nArchivo generado correctamente: 50_preguntas_wikiqa.csv")
print("\nPrimeras 10 preguntas seleccionadas:")
print(preguntas_eval.head(10))