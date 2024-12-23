# -*- coding: utf-8 -*
import pandas as pd
from utils.MyFairness import Fairness
from utils.MyUtils import MyPipeline
from utils.MeLogSingle import MeLogger
from utils.MyPreprocessing import PreprocessingDatasets
from sklearn.model_selection import StratifiedKFold

from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset

import os               
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

sess = tf.compat.v1.Session()

def processa_imputados_fairness(classifier:str,
                                tabela_resultados:dict):

    # Cria diretorios para salvar os resultados do experimento
    os.makedirs(f"./ResultadosFairML/Baseline/", exist_ok=True)

    _logger = MeLogger()
    fairness_class = Fairness()
    
    for dados, nome in zip(tabela_resultados['datasets'], tabela_resultados['nome_datasets']):
        (features_sensitive,
         privileged_groups,
         unprivileged_groups) = Fairness.retorna_featuresFairness(nome)  
                  
        df = dados.copy()
        # Start the Cross-Validation to classification task
        fold = 0
        cv = StratifiedKFold()
        X = df.drop(columns='target')
        y = df['target'].values
        x_cv = X.values

        for train_index, test_index in cv.split(x_cv, y):
            _logger.info(f"{nome} Fold = {fold} classifier = {classifier}")
            x_treino, x_teste = x_cv[train_index], x_cv[test_index]
            y_treino, y_teste = y[train_index], y[test_index]
        
            X_treino = pd.DataFrame(x_treino, columns=X.columns)
            X_teste = pd.DataFrame(x_teste, columns=X.columns)

            # Inicializando o normalizador (scaler)
            scaler = PreprocessingDatasets.inicializa_normalizacao(X_treino)

            # Normalizando os dados
            X_treino_norm = PreprocessingDatasets.normaliza_dados(
                scaler, X_treino
            )
            X_treino_norm["target"] = y_treino

            X_teste_norm = PreprocessingDatasets.normaliza_dados(scaler, X_teste) 
            X_teste_norm["target"] = y_teste 

            # Type para biblioteca AI Fairness 360 da IBM
            df_aif360_train = BinaryLabelDataset(df=X_treino_norm,
                                            label_names = ["target"],
                                            protected_attribute_names = features_sensitive
                                            )
            df_aif360_test = BinaryLabelDataset(df=X_teste_norm,
                                            label_names = ["target"],
                                            protected_attribute_names = features_sensitive
                                            )

            # Resetar o gráfico TensorFlow
            tf.compat.v1.reset_default_graph()

            # Criar uma nova sessão TensorFlow dentro do bloco 'with'
            with tf.compat.v1.Session() as sess:
                # Model to mitigation unfairness
                clf = fairness_class.choose_model_fair(classifier=classifier,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups, 
                                                    session = sess)
                clf.fit(df_aif360_train)

                data_debiasing_test  = clf.predict(df_aif360_test)
                predict_labels = data_debiasing_test.labels
                
                # Métricas
                classification_metric = ClassificationMetric(df_aif360_test,
                                                            data_debiasing_test,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)

                fairness_class.calculate_metrics(nome=nome,
                                                fold=fold,
                                                classification_metric=classification_metric,
                                                y_true=y_teste,
                                                y_pred=predict_labels)


            fold += 1

    results_final_tab = pd.DataFrame(fairness_class.results_metrics)
    results_final_tab.to_csv(f"./ResultadosFairML/Baseline/classification_fairness_baseline_{classifier}.csv", index=False)
    
    return _logger.info(f"Resultados salvos com sucesso: {classifier}")

if __name__ == "__main__":
    diretorio = "./DatasetsFairness"
    datasets = MyPipeline.carrega_datasets(diretorio)

    fairness = Fairness()
    tabela_resultados = fairness.cria_tabela_fairness(datasets)

    processa_imputados_fairness("adversarial", tabela_resultados) 
    # gerry, adversarial