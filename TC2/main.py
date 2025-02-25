import numpy as np
from utils import load_mnist, classification_metrics_from_confusion
from experiment import multiple_iterations

if __name__ == '__main__':
    # Carrega os dados MNIST
    X, y = load_mnist()
    
    # Opcional: para testes rápidos, reduzir o número de amostras
    # idx = np.random.choice(X.shape[0], 10000, replace=False)
    # X, y = X[idx], y[idx]
    
    # Parâmetros
    use_pca = False
    n_components = 100
    k = 3         # para k-NN
    n_iter = 100  # número de iterações
    
    # Executa as iterações e coleta métricas completas
    metrics, all_results = multiple_iterations(X, y, n_iter=n_iter, use_pca=use_pca, n_components=n_components, k=k)
    
    # Exibe estatísticas globais (acurácia)
    print("\nEstatísticas Globais (Acurácia):")
    for method, acc_list in metrics.items():
        acc_array = np.array(acc_list)
        print(f"\nMétodo: {method}")
        print(f"Média: {acc_array.mean():.4f}")
        print(f"Desvio Padrão: {acc_array.std():.4f}")
        print(f"Máxima: {acc_array.max():.4f}")
        print(f"Mínima: {acc_array.min():.4f}")
        print(f"Mediana: {np.median(acc_array):.4f}")
    
    # Para cada método, identifica a iteração com melhor e pior acurácia e mostra as matrizes de confusão
    for method in ['knn', 'qda', 'least_squares', 'kmeans', 'ensemble']:
        acc_array = np.array(metrics[method])
        best_iter = int(np.argmax(acc_array))
        worst_iter = int(np.argmin(acc_array))
        best_cm = all_results[best_iter][method]['confusion_matrix']
        worst_cm = all_results[worst_iter][method]['confusion_matrix']
        print(f"\nMétodo: {method}")
        print(f"Melhor iteração (#{best_iter+1}) - Acurácia: {acc_array[best_iter]:.4f}")
        print("Matriz de Confusão (Melhor):")
        print(best_cm)
        print("Métricas por Classe (Melhor):")
        best_metrics = classification_metrics_from_confusion(best_cm)
        for cls, met in best_metrics.items():
            print(f"  Classe {cls}: {met}")
        print(f"\nPior iteração (#{worst_iter+1}) - Acurácia: {acc_array[worst_iter]:.4f}")
        print("Matriz de Confusão (Pior):")
        print(worst_cm)
        print("Métricas por Classe (Pior):")
        worst_metrics = classification_metrics_from_confusion(worst_cm)
        for cls, met in worst_metrics.items():
            print(f"  Classe {cls}: {met}")
