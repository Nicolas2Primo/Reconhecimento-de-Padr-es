import numpy as np

def one_hot_encode(y, num_classes=10):
    """
    Codifica os rótulos em one-hot.
    """
    return np.eye(num_classes)[y]

def least_squares_train(X_train, y_train, num_classes=10):
    """
    Treina o classificador de mínimos quadrados.
    Acrescenta uma coluna de bias e calcula os pesos via pseudoinversa.
    """
    n_samples = X_train.shape[0]
    X_bias = np.hstack([np.ones((n_samples, 1)), X_train])
    Y_onehot = one_hot_encode(y_train, num_classes)
    # Cálculo dos pesos: W = pinv(X_bias) * Y_onehot
    W = np.linalg.pinv(X_bias) @ Y_onehot
    return W

def least_squares_predict(X, W):
    """
    Prediz os rótulos utilizando os pesos aprendidos.
    """
    n_samples = X.shape[0]
    X_bias = np.hstack([np.ones((n_samples, 1)), X])
    scores = X_bias @ W
    y_pred = np.argmax(scores, axis=1)
    return y_pred
