import tensorflow as tf

# Caminho para os dados (ajuste conforme seu diretório)
DATASET_PATH = './data'

# Função para carregar o dataset
def load_dataset(dataset_path):
    # Carrega o dataset de treinamento
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,  # 20% para validação
        subset="training",  # Subconjunto para treinamento
        seed=123,  # Semente para aleatoriedade
        image_size=(256, 256),  # Tamanho das imagens
        batch_size=32,  # Tamanho do batch
    )
    
    # Carrega o dataset de validação
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,  # 20% para validação
        subset="validation",  # Subconjunto para validação
        seed=123,  # Semente para aleatoriedade
        image_size=(256, 256),  # Tamanho das imagens
        batch_size=32,  # Tamanho do batch
    )
    
    # Acessando as classes e contando o número de classes
    class_names = train_ds.class_names  # Acessa as classes do dataset
    num_classes = len(class_names)  # Conta o número de classes
    
    # Retorna o dataset de treinamento, validação e o número de classes
    return train_ds, val_ds, num_classes

# Chamando a função para carregar o dataset
train_ds, val_ds, num_classes = load_dataset(DATASET_PATH)

# Exibindo o número de classes
print("Número de classes:", num_classes)

# Aqui você pode continuar com o restante do código, como treinamento do modelo
# Exemplo de criação de um modelo simples
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(256, 256, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Saída com número de classes
])

# Compilando o modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Para classes inteiras
    metrics=['accuracy']
)

# Treinando o modelo
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
