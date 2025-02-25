# pre_processor.py
import pandas as pd
import numpy as np

def load_ionosphere_data(filepath: str):
    """
    Carrega e pré-processa o dataset Ionosphere.
    
    Parâmetros:
        filepath (str): Caminho para o arquivo CSV do dataset.
    
    Retorna:
        X (np.ndarray): Matriz de dados (amostras x features).
        y (np.ndarray): Vetor de rótulos.
    """
    # O dataset não possui cabeçalho e a última coluna contém a classe ('g' ou 'b')
    df = pd.read_csv(filepath, header=None)
    
    # Converter todas as colunas, exceto a última, para float
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values
    return X, y
