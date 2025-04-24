import os
import random
import shutil
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils import class_weight
from logger import Logger


# Função para escolher o dataset
def escolher_dataset():
    print("Escolha o dataset que você deseja usar:")
    print("1: Coffee")
    print("2: PlantVillage")
    escolha = input("Digite o número correspondente ao dataset (1 ou 2): ")

    if escolha == '1':
        return 'Coffee'
    elif escolha == '2':
        return 'PlantVillage'
    else:
        print("Opção inválida. Usando 'Coffee' por padrão.")
        return 'Coffee'


# Função para configurar os diretórios e paths
class Config:
    def __init__(self, dataset_type='PlantVillage', plant_culture='Tomato', balance_classes=False):
        """
        Inicializa a configuração com os parâmetros do dataset.

        Args:
            dataset_type: Tipo de dataset ('Coffee' ou 'PlantVillage')
            plant_culture: Nome da cultura (usado apenas para PlantVillage)
            balance_classes: Se deve balancear as classes por sub-amostragem
        """
        self.SEED_VALUE = 42
        random.seed(self.SEED_VALUE)
        os.environ['PYTHONHASHSEED'] = str(self.SEED_VALUE)

        self.DATASET_TYPE = dataset_type
        self.PLANT_CULTURE = plant_culture
        self.BALANCE_CLASSES = balance_classes

        self.root_path = os.getcwd()
        # Atualizar o caminho para garantir que a pasta 'datasets' seja usada corretamente
        self.project_path = os.path.join(self.root_path, 'datasets', self.DATASET_TYPE)
        
        self.DATA_PATH = os.path.join(self.project_path, 'data')
        self.DATASET_PATH = os.path.join(self.project_path, 'dataset')
        self.MODEL_PATH = os.path.join(self.project_path, 'model')
        self.LOG_PATH = os.path.join(self.project_path, 'log')

        self.TRAIN_PATH = os.path.join(self.DATASET_PATH, 'train')
        self.VALIDATION_PATH = os.path.join(self.DATASET_PATH, 'validation')
        self.TEST_PATH = os.path.join(self.DATASET_PATH, 'test')

    def create_directories(self):
        """Cria todos os diretórios necessários caso não existam."""
        for dir_path in [self.project_path, self.DATASET_PATH, self.MODEL_PATH, self.LOG_PATH,
                         self.TRAIN_PATH, self.VALIDATION_PATH, self.TEST_PATH]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Diretório criado: {dir_path}")


# Função para análise de distribuição das classes no dataset
def analyze_class_distribution(source_data_path, filter_prefix=None):
    class_counts = {}
    for class_name in os.listdir(source_data_path):
        if filter_prefix and filter_prefix not in class_name:
            continue

        class_dir = os.path.join(source_data_path, class_name)
        if os.path.isdir(class_dir):
            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                           and os.path.isfile(os.path.join(class_dir, f))]
            class_counts[class_name] = len(image_files)

    return class_counts


# Função para encontrar a classe com o menor número de imagens
def find_min_class_size(class_counts):
    if not class_counts:
        return 0, None
    min_class_size = min(class_counts.values())
    min_class_name = min(class_counts, key=class_counts.get)
    return min_class_size, min_class_name


# Função para dividir os dados com balanceamento automático
def split_data_from_drive_balanced(config, logger, train_ratio=0.6, validation_ratio=0.2):
    for dir_path in [config.TRAIN_PATH, config.VALIDATION_PATH, config.TEST_PATH]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info(f"Iniciando a divisão dos dados de {config.DATA_PATH} com balanceamento={config.BALANCE_CLASSES}")

    # Verifica se o diretório fonte existe
    if not os.path.exists(config.DATA_PATH):
        logger.error(f"Erro: Caminho de origem {config.DATA_PATH} não existe!")
        return

    # Analisa a distribuição das classes
    if config.DATASET_TYPE == 'PlantVillage':
        class_counts = analyze_class_distribution(config.DATA_PATH, filter_prefix=config.PLANT_CULTURE)
    else:
        class_counts = analyze_class_distribution(config.DATA_PATH)

    # Exibe informações sobre a distribuição
    total_images = sum(class_counts.values())
    min_class_size, min_class_name = find_min_class_size(class_counts)
    max_class_size = max(class_counts.values()) if class_counts else 0
    logger.info(f"Distribuição original das classes: {class_counts}")

    # Balanceamento por sub-amostragem
    samples_per_class = min_class_size if config.BALANCE_CLASSES else None
    logger.info(f"Sub-amostragem aplicada: {samples_per_class} imagens por classe" if config.BALANCE_CLASSES else "Sem balanceamento aplicado")

    # Processa cada classe
    for class_name in os.listdir(config.DATA_PATH):
        class_dir = os.path.join(config.DATA_PATH, class_name)
        if not os.path.isdir(class_dir):
            continue

        logger.info(f"Processando a classe: {class_name}")
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                       and os.path.isfile(os.path.join(class_dir, f))]

        if config.BALANCE_CLASSES:
            random.shuffle(image_files)
            image_files = image_files[:samples_per_class]

        # Divisão dos dados
        num_images = len(image_files)
        num_train = int(num_images * train_ratio)
        num_validation = int(num_images * validation_ratio)
        train_images, validation_images, test_images = image_files[:num_train], image_files[num_train:num_train + num_validation], image_files[num_train + num_validation:]

        # Criação de diretórios para as divisões
        dst_train_class_dir = os.path.join(config.TRAIN_PATH, class_name)
        dst_validation_class_dir = os.path.join(config.VALIDATION_PATH, class_name)
        dst_test_class_dir = os.path.join(config.TEST_PATH, class_name)

        for dir_path in [dst_train_class_dir, dst_validation_class_dir, dst_test_class_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # Copia as imagens para os respectivos diretórios
        for image_set, dst_dir in [(train_images, dst_train_class_dir),
                                   (validation_images, dst_validation_class_dir),
                                   (test_images, dst_test_class_dir)]:
            for image_file in image_set:
                src_path = os.path.join(class_dir, image_file)
                dst_path = os.path.join(dst_dir, image_file)
                shutil.copy2(src_path, dst_path)

        logger.info(f"Classe {class_name}: {len(train_images)} treino, {len(validation_images)} validação, {len(test_images)} teste")


# Função para verificar a qualidade do balanceamento
def check_balance_quality(class_counts):
    values = list(class_counts.values())
    mean = sum(values) / len(values)
    if mean == 0:
        return 0, 0, 0

    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = variance ** 0.5
    cv = std_dev / mean
    return cv, mean, std_dev


# Função principal
def main():
    print("Iniciando a divisão automática do dataset com balanceamento")

    dataset_escolhido = escolher_dataset()
    dataset_dir = os.path.join("datasets", dataset_escolhido)

    config = Config(dataset_type=dataset_escolhido)
    config.create_directories()

    logger = Logger(config.LOG_PATH, f"{config.PLANT_CULTURE}_dataset_split")
    logger.info("Iniciando a divisão automática do dataset")

    split_data_from_drive_balanced(config, logger)


if __name__ == "__main__":
    main()
