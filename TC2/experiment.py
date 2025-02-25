import time
import numpy as np
from utils import pca, compute_confusion_matrix
from classifiers.knn import knn_predict
from classifiers.qda import qda_train, qda_predict
from classifiers.least_squares import least_squares_train, least_squares_predict
from classifiers.kmeans import kmeans_train, kmeans_predict
from classifiers.ensemble import ensemble_weighted_predict

def run_experiment(X, y, use_pca=True, n_components=100, k=3):
    """
    Executa uma rodada de treinamento e teste para cada classificador.
    Divide os dados em treinamento e teste, aplica PCA (se solicitado) e mede:
      - Acurácia
      - Tempo de execução
      - Matriz de confusão
      - Predições (usadas no ensemble)
    
    Também implementa um ensemble ponderado, que combina as predições de todos os modelos
    usando os seus desempenhos (acurácias) como pesos.
    """
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    split = int(0.8 * n_samples)
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # (Opcional) Estratégia de amostragem balanceada – descomente se necessário
    # from utils import sample_training_set
    # X_train, y_train = sample_training_set(X_train, y_train, samples_per_class=200)
    
    # Aplica PCA se solicitado
    if use_pca:
        print("Aplicando PCA...")
        X_train, components, _, X_mean = pca(X_train, n_components)
        X_test = np.dot(X_test - X_mean, components)
    
    results = {}
    individual = {}
    
    # k-NN
    print("Treinando k-NN...")
    start = time.time()
    y_pred_knn = knn_predict(X_train, y_train, X_test, k=k)
    t_knn = time.time() - start
    acc_knn = np.mean(y_pred_knn == y_test)
    cm_knn = compute_confusion_matrix(y_test, y_pred_knn)
    individual['knn'] = {'accuracy': acc_knn, 'time': t_knn, 'confusion_matrix': cm_knn, 'pred': y_pred_knn}
    print("k-NN concluído.")
    
    # QDA
    print("Treinando QDA...")
    start = time.time()
    means, cov_invs, cov_dets = qda_train(X_train, y_train)
    y_pred_qda = qda_predict(X_test, means, cov_invs, cov_dets)
    t_qda = time.time() - start
    acc_qda = np.mean(y_pred_qda == y_test)
    cm_qda = compute_confusion_matrix(y_test, y_pred_qda)
    individual['qda'] = {'accuracy': acc_qda, 'time': t_qda, 'confusion_matrix': cm_qda, 'pred': y_pred_qda}
    print("QDA concluído.")
    
    # Linear de Mínimos Quadrados
    print("Treinando Linear de Mínimos Quadrados...")
    start = time.time()
    W = least_squares_train(X_train, y_train)
    y_pred_ls = least_squares_predict(X_test, W)
    t_ls = time.time() - start
    acc_ls = np.mean(y_pred_ls == y_test)
    cm_ls = compute_confusion_matrix(y_test, y_pred_ls)
    individual['least_squares'] = {'accuracy': acc_ls, 'time': t_ls, 'confusion_matrix': cm_ls, 'pred': y_pred_ls}
    print("Linear de Mínimos Quadrados concluído.")
    
    # K-Médias
    print("Treinando K-Médias...")
    start = time.time()
    centroids, cluster_labels = kmeans_train(X_train, y_train)
    y_pred_km = kmeans_predict(X_test, centroids, cluster_labels)
    t_km = time.time() - start
    acc_km = np.mean(y_pred_km == y_test)
    cm_km = compute_confusion_matrix(y_test, y_pred_km)
    individual['kmeans'] = {'accuracy': acc_km, 'time': t_km, 'confusion_matrix': cm_km, 'pred': y_pred_km}
    print("K-Médias concluído.")
    
    # Ensemble Ponderado
    print("Treinando Ensemble ponderado (votação ponderada pelos desempenhos)...")
    y_pred_ensemble = ensemble_weighted_predict(individual)
    acc_ensemble = np.mean(y_pred_ensemble == y_test)
    cm_ensemble = compute_confusion_matrix(y_test, y_pred_ensemble)
    results['ensemble'] = {
         'accuracy': acc_ensemble,
         'time': None,  # O tempo do ensemble não é medido separadamente
         'confusion_matrix': cm_ensemble,
         'selected_classifier': 'ensemble_weighted'
    }
    
    # Inclui os resultados individuais
    results.update(individual)
    results['y_test'] = y_test
    
    return results

def multiple_iterations(X, y, n_iter=100, use_pca=True, n_components=100, k=3):
    """
    Executa várias iterações para avaliar estatisticamente o desempenho dos modelos.
    Armazena para cada iteração os resultados completos de cada classificador.
    
    Retorna:
      - metrics: dicionário com listas de acurácias por método.
      - all_results: lista de resultados (um por iteração) contendo todas as informações.
    """
    metrics = { 'knn': [], 'qda': [], 'least_squares': [],
                'kmeans': [], 'ensemble': [] }
    all_results = []  # lista de resultados completos para cada iteração
    
    for i in range(n_iter):
        print(f"Iteração {i+1}/{n_iter}")
        res = run_experiment(X, y, use_pca=use_pca, n_components=n_components, k=k)
        all_results.append(res)
        for method in metrics.keys():
            metrics[method].append(res[method]['accuracy'])
    return metrics, all_results
