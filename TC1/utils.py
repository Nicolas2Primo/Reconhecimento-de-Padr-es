import numpy as np
import matplotlib.pyplot as plt
import time

def measure_time(func, *args, **kwargs):
    """
    Mede o tempo de execução de uma função.
    
    Retorna:
        resultado, tempo (em segundos)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def analyze_matrix(A: np.ndarray, name: str = "Matriz"):
    """
    Analisa a matriz exibindo o rank e o número de condição.
    
    Parâmetros:
        A (np.ndarray): Matriz a ser analisada.
        name (str): Nome da matriz para exibição.
    
    Retorna:
        rank (int) e condição (float)
    """
    rank = np.linalg.matrix_rank(A)
    cond_num = np.linalg.cond(A)
    print(f"{name}: Rank = {rank}, Número de condição = {cond_num:.6e}")
    return rank, cond_num

def svd_inverse(A: np.ndarray, reg_threshold: float = 1e-12) -> np.ndarray:
    """
    Calcula a inversa de uma matriz A usando SVD com regularização.
    
    Parâmetros:
        A (np.ndarray): Matriz a ser invertida.
        reg_threshold (float): Limiar para regularização dos valores singulares.
    
    Retorna:
        A_inv (np.ndarray): Inversa regularizada de A.
    """
    U, s, Vh = np.linalg.svd(A)
    s_inv = np.array([1/si if si > reg_threshold else 0 for si in s])
    A_inv = Vh.T @ np.diag(s_inv) @ U.T
    return A_inv

def plot_singular_values(A: np.ndarray, title: str = "Valores Singulares"):
    """
    Plota os valores singulares da matriz A em escala logarítmica.
    """
    U, s, Vh = np.linalg.svd(A)
    plt.figure()
    plt.semilogy(s, 'o-')
    plt.title(title)
    plt.xlabel("Índice")
    plt.ylabel("Valor Singular (escala logarítmica)")
    plt.grid(True)
    plt.show()
