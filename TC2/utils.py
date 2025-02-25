import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist():
    """
    Carrega o dataset MNIST e normaliza os pixels para [0,1].
    """
    print("Carregando MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0  # Normalização
    y = mnist.target.astype(np.int32)
    return X, y

def pca(X, n_components=100):
    """
    Aplica PCA manualmente usando a decomposição em autovalores.
    
    Parâmetros:
      X: array de dados com shape (n_samples, n_features)
      n_components: número de componentes principais a reter
    
    Retorna:
      X_reduced: dados transformados com shape (n_samples, n_components)
      components: autovetores (componentes principais)
      explained_variance: autovalores correspondentes
      X_mean: média dos dados originais (para projeção de novos dados)
    """
    # Calcula a média dos dados originais
    X_mean = np.mean(X, axis=0)
    # Centraliza os dados
    X_centered = X - X_mean
    # Calcula a matriz de covariância
    covariance_matrix = np.cov(X_centered, rowvar=False)
    # Obtém os autovalores e autovetores
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Ordena do maior para o menor
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    # Seleciona os n_components primeiros
    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]
    # Projeta os dados
    X_reduced = np.dot(X_centered, components)
    return X_reduced, components, explained_variance, X_mean


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """
    Calcula a matriz de confusão.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def classification_metrics_from_confusion(cm):
    """
    Dada uma matriz de confusão (num_classes x num_classes), calcula as métricas
    de desempenho por classe: precision, recall, F1 e suporte.
    
    Retorna um dicionário onde cada chave é a classe (0 a num_classes-1) e os valores
    são dicionários com 'precision', 'recall', 'f1' e 'support'.
    """
    num_classes = cm.shape[0]
    metrics = {}
    for c in range(num_classes):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        support = cm[c, :].sum()
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[c] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': support}
    return metrics

def sample_training_set(X, y, samples_per_class=100):
    """
    Retorna um subconjunto de treinamento balanceado, selecionando aleatoriamente
    'samples_per_class' amostras para cada classe.
    """
    import numpy as np
    unique_classes = np.unique(y)
    indices = []
    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        # Se houver menos amostras que samples_per_class, utiliza todas
        n_samples = min(samples_per_class, len(cls_idx))
        sampled_idx = np.random.choice(cls_idx, n_samples, replace=False)
        indices.extend(sampled_idx)
    indices = np.array(indices)
    return X[indices], y[indices]