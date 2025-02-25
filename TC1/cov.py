import numpy as np

def compute_correlation_eq152(X: np.ndarray) -> np.ndarray:
    """
    Implementação da Equação 152:
        R̂_x = 1/N * Σ[n=1 to N] x(n)x(n)^T
    Realiza a soma das contribuições de cada amostra utilizando um loop.
    """
    N, d = X.shape
    R = np.zeros((d, d))
    for n in range(N):
        x_n = X[n].reshape(d, 1)  # Converter a amostra para vetor coluna
        R += x_n @ x_n.T
    return R / N

def compute_correlation_eq153(X: np.ndarray) -> np.ndarray:
    """
    Implementação da Equação 153:
        R̂_x = 1/N * (X.T @ X)
    Utiliza operações vetorizadas para obter a mesma estimativa.
    """
    N = X.shape[0]
    return (X.T @ X) / N

def compute_covariance_eq151(correlation_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Implementa a Equação 151:
        Ĉ_x = R̂_x - m m^T
    onde m é o vetor médio dos dados.
    """
    m = np.mean(X, axis=0, keepdims=True)
    return correlation_matrix - (m.T @ m)

def compute_covariance_eq152(X: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de covariância usando:
      - A Equação 152 para obter R̂_x (loop)
      - A Equação 151 para obter Ĉ_x = R̂_x - mm^T
    """
    R = compute_correlation_eq152(X)
    return compute_covariance_eq151(R, X)

def compute_covariance_eq153(X: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de covariância usando:
      - A Equação 153 para obter R̂_x (vetorizada)
      - A Equação 151 para obter Ĉ_x = R̂_x - mm^T
    """
    R = compute_correlation_eq153(X)
    return compute_covariance_eq151(R, X)

def recursive_cov_estimator(X: np.ndarray) -> np.ndarray:
    """
    Implementação da Equação 155:
        R̂_x(n) = (n-1)/n * R̂_x(n-1) + 1/n * x(n)x(n)^T, com R̂_x(0) = I_d.
    Calcula recursivamente a estimativa da matriz de correlação.
    """
    N, d = X.shape
    R_rec = np.eye(d)  # Inicializa com a identidade
    for n in range(1, N + 1):
        alpha = (n - 1) / n
        R_rec = alpha * R_rec + (1 - alpha) * np.outer(X[n-1], X[n-1])
    return R_rec

def compute_covariance_recursive(X: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de covariância utilizando a estimação recursiva (Eq. 155)
    seguida da Equação 151: Ĉ_x = R̂_x - m m^T.
    """
    R_rec = recursive_cov_estimator(X)
    m = np.mean(X, axis=0, keepdims=True)
    return R_rec - (m.T @ m)

def compute_class_covariance(X: np.ndarray, y: np.ndarray, target_class: str) -> tuple:
    """
    Calcula a matriz de covariância para os dados de uma classe específica.

    Parâmetros:
        X (np.ndarray): Matriz de dados (N x d)
        y (np.ndarray): Vetor de rótulos
        target_class (str): Rótulo da classe desejada ('g' ou 'b')

    Retorna:
        (C_class, m_class, N_class)
    """
    X_class = X[y == target_class]
    N_class = X_class.shape[0]
    m_class = np.mean(X_class, axis=0, keepdims=True)
    R_class = (X_class.T @ X_class) / N_class
    C_class = R_class - (m_class.T @ m_class)
    return C_class, m_class, N_class
