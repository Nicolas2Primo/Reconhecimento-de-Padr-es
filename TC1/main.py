# main.py
import numpy as np
import matplotlib.pyplot as plt
from pre_processor import load_ionosphere_data
import cov
import utils

def print_all_submatrices(A: np.ndarray, label: str):
    """
    Imprime as submatrizes 3x3 dos cantos superior esquerdo, superior direito,
    inferior esquerdo e inferior direito de A.
    """
    tl = A[:3, :3]
    tr = A[:3, -3:]
    bl = A[-3:, :3]
    br = A[-3:, -3:]
    
    print(f"\nSubmatriz 3x3 - {label} (Superior Esquerdo):")
    print(tl)
    print(f"\nSubmatriz 3x3 - {label} (Superior Direito):")
    print(tr)
    print(f"\nSubmatriz 3x3 - {label} (Inferior Esquerdo):")
    print(bl)
    print(f"\nSubmatriz 3x3 - {label} (Inferior Direito):")
    print(br)

def repeat_experiment(func, X, n_rep=10, **kwargs):
    """
    Repete a função 'func' n_rep vezes para medir o tempo de execução e retorna
    a média e o desvio padrão dos tempos, bem como o resultado obtido na última repetição.
    """
    times = []
    result = None
    for _ in range(n_rep):
        result, t = utils.measure_time(func, X, **kwargs)
        times.append(t)
    avg_time = np.mean(times)
    std_time = np.std(times)
    return result, avg_time, std_time

def main():
    # Caminho para o dataset (ajuste conforme necessário)
    filepath = 'dataset/ionosphere.data'
    X, y = load_ionosphere_data(filepath)
    N, d = X.shape
    print(f"Número de amostras: {N}, Dimensão dos dados: {d}")
    
    # Flag de normalização: 'N' para divisão por N ou 'N-1' para divisão por (N-1)
    norm_flag = 'N'
    
    # Método de referência: np.cov (bias=True para divisão por N; bias=False para N-1)
    if norm_flag == 'N':
        C_native = np.cov(X, rowvar=False, bias=True)
    else:
        C_native = np.cov(X, rowvar=False, bias=False)
    
    # Repetição dos experimentos (10 repetições) para cada método
    n_rep = 10
    
    # --- np.cov (método nativo) ---
    _, time_native, _ = repeat_experiment(lambda X: C_native, X, n_rep=n_rep)
    error_native = 0.0  # referência
    print("\n--- Método np.cov (Referência) ---")
    print_all_submatrices(C_native, "np.cov")
    print(f"\nTempo médio np.cov: {time_native*1e6:.2f} µs, Erro: {error_native:.6f}")
    
    # --- Método Eq. 152 (Loop) ---
    C_152, time_152_avg, time_152_std = repeat_experiment(cov.compute_covariance_eq152, X, n_rep=n_rep)
    error_152 = np.linalg.norm(C_native - C_152, 'fro')
    print("\n--- Método Eq. 152 (Loop) ---")
    print_all_submatrices(C_152, "Eq. 152")
    print(f"\nTempo médio Eq. 152: {time_152_avg*1e6:.2f} µs (± {time_152_std*1e6:.2f} µs), ||E||_F = {error_152:.6f}")
    
    # --- Método Eq. 153 (Vetorizada) ---
    C_153, time_153_avg, time_153_std = repeat_experiment(cov.compute_covariance_eq153, X, n_rep=n_rep)
    error_153 = np.linalg.norm(C_native - C_153, 'fro')
    print("\n--- Método Eq. 153 (Vetorizada) ---")
    print_all_submatrices(C_153, "Eq. 153")
    print(f"\nTempo médio Eq. 153: {time_153_avg*1e6:.2f} µs (± {time_153_std*1e6:.2f} µs), ||E||_F = {error_153:.6f}")
    
    # --- Método Recursivo (Eq. 155 + Eq. 151) ---
    C_rec, time_rec_avg, time_rec_std = repeat_experiment(cov.compute_covariance_recursive, X, n_rep=n_rep)
    error_rec = np.linalg.norm(C_native - C_rec, 'fro')
    print("\n--- Método Recursivo (Eq. 155 + Eq. 151) ---")
    print_all_submatrices(C_rec, "Recursivo")
    print(f"\nTempo médio Recursivo: {time_rec_avg*1e6:.2f} µs (± {time_rec_std*1e6:.2f} µs), ||E||_F = {error_rec:.6f}")
    
    # --- Covariância Local por Classe ---
    print("\n--- Covariância Local por Classe ---")
    for target in ['g', 'b']:
        C_class, m_class, N_class = cov.compute_class_covariance(X, y, target)
        print(f"\nClasse '{target}' ({N_class} amostras):")
        print_all_submatrices(C_class, f"Classe '{target}'")
        utils.analyze_matrix(C_class, name=f"Covariância Classe '{target}'")
    
    # --- Análise de Invertibilidade da Matriz Global ---
    print("\n--- Análise de Invertibilidade da Covariância Global (Eq. 153) ---")
    rank_global, cond_global = utils.analyze_matrix(C_153, name="Matriz de Covariância Global")
    if cond_global > 1e15:
        print("Observação: A matriz apresenta alto número de condicionamento, indicando mal-condicionamento e possível não-invertibilidade exata.")
    C_global_inv = utils.svd_inverse(C_153)
    print("Inversa (pseudo-inversa via SVD com regularização) calculada.")
    
    # Plot dos valores singulares da matriz global
    utils.plot_singular_values(C_153, title="Valores Singulares - Covariância Global")
    
    # --- Gráfico Comparativo dos Tempos de Execução ---
    methods = ['np.cov', 'Eq. 152 (Loop)', 'Eq. 153 (Vetorizada)', 'Recursivo (Eq. 155)']
    times = [time_native*1e6, time_152_avg*1e6, time_153_avg*1e6, time_rec_avg*1e6]
    errors = [0, time_152_std*1e6, time_153_std*1e6, time_rec_std*1e6]
    
    plt.figure(figsize=(8, 6))
    plt.bar(methods, times, yerr=errors, capsize=5, color=['blue', 'green', 'orange', 'red'])
    plt.ylabel("Tempo médio (µs)")
    plt.title("Comparação de Tempo de Execução entre Implementações")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == '__main__':
    main()
