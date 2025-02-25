import numpy as np

def ensemble_weighted_predict(individual):
    """
    Realiza a votação ponderada entre os classificadores.
    
    Parâmetros:
      individual: dicionário com chaves para cada classificador (por exemplo, 'knn', 'qda', etc.),
                  cada um contendo:
                    - 'pred': array com as predições (shape: [n_test])
                    - 'accuracy': acurácia obtida nessa rodada (usada como peso)
                  Nota: a chave 'y_test' deve ser ignorada.
    
    Retorna:
      ensemble_pred: array com as predições do ensemble, onde para cada amostra a classe é escolhida
                     com base na soma dos pesos dos classificadores que votaram nela.
    """
    # Filtra apenas as entradas dos classificadores
    classifier_keys = [key for key in individual.keys() if key != 'y_test']
    n_classifiers = len(classifier_keys)
    n_test = individual[classifier_keys[0]]['pred'].shape[0]
    
    # Recolhe predições e pesos (acurácias) de cada classificador
    preds = []
    weights = []
    for key in classifier_keys:
        preds.append(individual[key]['pred'])
        weights.append(individual[key]['accuracy'])
    
    preds = np.array(preds)      # shape: (n_classifiers, n_test)
    weights = np.array(weights)  # shape: (n_classifiers,)
    
    ensemble_pred = np.zeros(n_test, dtype=int)
    
    # Para cada amostra, acumula os pesos para cada classe e escolhe a classe com maior soma
    for i in range(n_test):
        vote_sum = {}
        for j in range(n_classifiers):
            vote = preds[j, i]
            vote_sum[vote] = vote_sum.get(vote, 0) + weights[j]
        ensemble_pred[i] = max(vote_sum, key=vote_sum.get)
    
    return ensemble_pred
