import numpy as np
from collections import Counter

def kmeans_train(X_train, y_train, n_clusters=10, max_iter=100):
    """
    Aplica o algoritmo de k-means otimizado:
      - Inicializa os centróides a partir de amostras aleatórias.
      - Atribui cada amostra ao centróide mais próximo (usando distâncias ao quadrado).
      - Atualiza os centróides com a média das amostras atribuídas, utilizando list comprehension.
    Retorna:
      - centroids: array com os centróides finais.
      - cluster_labels: rótulo associado a cada cluster, determinado pela votação majoritária dos rótulos originais.
    """
    n_samples, n_features = X_train.shape
    np.random.seed(42)
    initial_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = X_train[initial_indices]
    
    for it in range(max_iter):
        # Calcula distâncias ao quadrado (evita o sqrt)
        distances = np.sum((X_train[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        clusters = np.argmin(distances, axis=1)
        
        # Atualiza os centróides de forma vetorizada via list comprehension
        new_centroids = np.array([
            X_train[clusters == i].mean(axis=0) if np.any(clusters == i) 
            else X_train[np.random.choice(n_samples)]
            for i in range(n_clusters)
        ])
        
        # Critério de parada: se os centróides não mudarem significativamente
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    # Associação dos clusters a classes via votação majoritária
    cluster_labels = np.array([
        Counter(y_train[clusters == i]).most_common(1)[0][0] if np.any(clusters == i) 
        else -1  # ou qualquer valor que indique ausência de amostras
        for i in range(n_clusters)
    ])
    
    return centroids, cluster_labels

def kmeans_predict(X, centroids, cluster_labels):
    """
    Prediz rótulos para os dados X com base nos centróides e nos rótulos dos clusters.
    Utiliza distâncias ao quadrado para identificar o centróide mais próximo.
    """
    # Calcula as distâncias ao quadrado entre X e cada centróide
    distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    clusters = np.argmin(distances, axis=1)
    y_pred = cluster_labels[clusters]
    return y_pred
