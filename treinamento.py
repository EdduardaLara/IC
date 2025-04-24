import os
import time
import psutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Configurações
DATASET_PATH = "datasets"

print("Selecione o dataset para treinamento:")
print("1 - PlantVillage")
print("2 - Coffee")
escolha = input("Digite 1 ou 2: ")

if escolha == "1":
    DATASET_NOME = "PlantVillage"
elif escolha == "2":
    DATASET_NOME = "coffee"
else:
    print("Opção inválida. Usando 'PlantVillage' por padrão.")
    DATASET_NOME = "PlantVillage"

INPUT_SHAPE = (64, 64, 3)
EPOCHS = 20
BATCH_SIZE = 32
MODEL_NAME = f"modelo_{DATASET_NOME.capitalize()}_melhorado.h5"

# Caminho final do dataset
CAMINHO_COMPLETO = os.path.join(DATASET_PATH, DATASET_NOME)

# Diretórios
train_dir = os.path.join(DATASET_PATH, DATASET_NOME, "dataset", "train")
val_dir = os.path.join(DATASET_PATH, DATASET_NOME, "dataset", "validation")

# Geradores de imagem
datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Construção do modelo
num_classes = len(train_generator.class_indices)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento
inicio = time.time()
process = psutil.Process(os.getpid())

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop],
    verbose=1
)

tempo_total = time.time() - inicio
memoria_usada = process.memory_info().rss / (1024 ** 2)

# Salvando o modelo treinado
model.save(MODEL_NAME)

# Relatório final
print(f"\nTreinamento finalizado!")
print(f"Dataset usado: {DATASET_NOME}")
print(f"Número de classes: {num_classes}")
print(f"Tempo total de treinamento: {tempo_total:.2f} segundos")
print(f"Uso de memória: {memoria_usada:.2f} MB")
print(f"Modelo salvo como: {MODEL_NAME}")
