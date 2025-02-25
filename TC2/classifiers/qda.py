import numpy as np

def qda_train(X_train, y_train, num_classes=10, reg=1e-5):
    """
    Treina o classificador QDA:
      - Calcula a média e a matriz de covariância para cada classe.
      - reg é um valor de regularização para evitar que as covariâncias fiquem singulares.
    Retorna:
      - means: array de shape (num_classes, n_features)
      - cov_invs: lista de matrizes inversas (uma para cada classe)
      - cov_dets: lista com os determinantes das matrizes de covariância
    """
    n_features = X_train.shape[1]
    means = np.zeros((num_classes, n_features))
    cov_invs = []
    cov_dets = []
    
    for c in range(num_classes):
        X_c = X_train[y_train == c]
        means[c] = np.mean(X_c, axis=0)
        # Covariância com regularização (adição de reg*I)
        cov = np.cov(X_c, rowvar=False) + reg * np.eye(n_features)
        cov_inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        cov_invs.append(cov_inv)
        cov_dets.append(det)
    
    return means, cov_invs, cov_dets

def qda_predict(X, means, cov_invs, cov_dets, num_classes=10):
    """
    Prediz rótulos usando os parâmetros do QDA.
    Calcula a função discriminante para cada classe:
       g_c(x) = -0.5*log(det(Sigma_c)) - 0.5*(x-mu_c)^T*Sigma_c^-1*(x-mu_c)
    """
    num_samples = X.shape[0]
    scores = np.zeros((num_samples, num_classes))
    
    for c in range(num_classes):
        diff = X - means[c]
        quad_form = np.sum(diff @ cov_invs[c] * diff, axis=1)
        scores[:, c] = -0.5 * np.log(cov_dets[c] + 1e-10) - 0.5 * quad_form
    y_pred = np.argmax(scores, axis=1)
    return y_pred
