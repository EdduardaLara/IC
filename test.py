import os
import tensorflow as tf
import numpy as np
import psutil
import time
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Seleção do modelo
print("Selecione o modelo para teste:")
print("1 - modelo_PlantVillage_melhorado.h5")
print("2 - modelo_Coffee_melhorado.h5")
escolha = input("Digite 1 ou 2: ")

if escolha == "1":
    modelo_nome = "modelo_PlantVillage_melhorado.h5"
    dataset_nome = "PlantVillage"
elif escolha == "2":
    modelo_nome = "modelo_Coffee_melhorado.h5"
    dataset_nome = "Coffee"
else:
    raise ValueError("Escolha inválida. Por favor, execute novamente e digite 1 ou 2.")

# Carrega o modelo treinado
model = load_model(modelo_nome)

# Diretório de teste
test_dir = f"datasets/{dataset_nome}/dataset/test"
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Diretório de teste '{test_dir}' não encontrado. Verifique se o dataset foi dividido corretamente.")

# Classes
class_names = sorted(os.listdir(test_dir))
num_classes = len(class_names)
print(f"Classes detectadas: {class_names}")

# Começa a contagem de tempo e uso de memória
start_time = time.time()
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024**2  # em MB

# Gerador de imagem para o conjunto de teste
datagen_test = ImageDataGenerator(rescale=1.0 / 255)

test_generator = datagen_test.flow_from_directory(
    test_dir,
    target_size=(64, 64),  # Deve ser o mesmo tamanho usado no treinamento
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Avaliação
loss, accuracy = model.evaluate(test_generator)
print(f"\nAcurácia no conjunto de teste: {accuracy * 100:.2f}%\n")

# Processamento das previsões
total_images = 0
resultados = []
for class_folder in class_names:
    class_path = os.path.join(test_dir, class_folder)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = image.load_img(img_path, target_size=(64, 64))  # Ajuste o tamanho aqui também
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Alteração: Adicionar um "Flatten" para adequar o modelo
        prediction = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        resultados.append([img_name, class_folder, predicted_class, f"{confidence:.2f}%"])
        total_images += 1

# Tempo total e uso de memória
end_time = time.time()
execution_time = end_time - start_time
time_per_image = (execution_time / total_images) * 1000  # em ms
mem_after = process.memory_info().rss / 1024**2  # em MB
mem_used = mem_after - mem_before

# Salva resultados de previsão
with open("resultados_previsoes.txt", "w", encoding="utf-8") as f:
    f.write(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%\n\n")
    f.write("Previsões para todas as imagens de teste:\n")
    for r in resultados:
        f.write(
            f"Imagem: {r[0]} | Classe real: {r[1]} | Classe prevista: {r[2]} | Confiança: {r[3]}\n"
        )

# Geração de planilha
df_resultado = pd.DataFrame([{
    "Dataset": dataset_nome,
    "Number class": num_classes,
    "Number total de imagens do dataset usado": total_images,
    "Acuracia": round(accuracy * 100, 2),
    "Time total": round(execution_time, 2),
    "Time ms/image": round(time_per_image, 2),
    "Memo usage [MB]": round(mem_used, 2),
    "Method": "CNN 3xConv + Pooling + Dense(128) + Softmax"
}])

df_resultado.to_excel("resultado_teste_modelo.xlsx", index=False)
print("\n Planilha 'resultado_teste_modelo.xlsx' gerada com sucesso.")
print(f"Tempo total: {execution_time:.2f} segundos | Uso de memória: {mem_used:.2f} MB")

import numpy as np
import pandas as pd
from tqdm import tqdm

# Obtemos os caminhos das imagens testadas
caminhos_imagens = test_generator.filepaths

# Verdadeiros e nomes das classes
classes_reais = test_generator.classes
nomes_classes = list(test_generator.class_indices.keys())

# Faz predições (em lote)
predicoes = model.predict(test_generator, verbose=1)
classes_preditas = np.argmax(predicoes, axis=1)

# Cria o DataFrame detalhado
dados_resultado = []

for caminho, real, pred in zip(caminhos_imagens, classes_reais, classes_preditas):
    dados_resultado.append({
        "imagem": caminho,
        "classe_real": nomes_classes[real],
        "classe_predita": nomes_classes[pred],
        "acertou": nomes_classes[real] == nomes_classes[pred]
    })

df_detalhado = pd.DataFrame(dados_resultado)
df_detalhado.to_csv("relatorio_detalhado_teste.csv", index=False)

print("\nRelatório salvo em 'relatorio_detalhado_teste.csv'")
