import numpy as np

def knn_predict(X_train, y_train, X_test, k=3):
    """
    Implementa o classificador k-NN de forma otimizada.
    - Calcula a matriz de distâncias entre X_test e X_train de forma vetorizada.
    - Utiliza np.argpartition para selecionar os k vizinhos mais próximos sem ordenar completamente.
    - Para cada amostra de teste, realiza a votação majoritária para definir a classe.
    """
    # Calcula a soma dos quadrados de cada linha (amostra) para X_train e X_test
    X_train_sq = np.sum(X_train**2, axis=1)
    X_test_sq = np.sum(X_test**2, axis=1)
    
    # Computa a matriz de distâncias: d(i,j) = ||X_test[i] - X_train[j]||
    # A operação abaixo utiliza a expansão de (a - b)^2 = a^2 + b^2 - 2ab
    distances = np.sqrt(X_test_sq[:, np.newaxis] + X_train_sq - 2 * np.dot(X_test, X_train.T))
    
    num_test = X_test.shape[0]
    y_pred = np.empty(num_test, dtype=int)
    
    # Para cada amostra de teste, encontra os k vizinhos mais próximos e vota
    for i in range(num_test):
        # np.argpartition retorna os índices dos k menores valores, sem ordem completa
        knn_idx = np.argpartition(distances[i], k)[:k]
        # Realiza a votação majoritária
        # Usando np.unique com return_counts para identificar o rótulo mais frequente
        labels, counts = np.unique(y_train[knn_idx], return_counts=True)
        majority_label = labels[np.argmax(counts)]
        y_pred[i] = majority_label

    return y_pred
