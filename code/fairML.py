# -*- coding: utf-8 -*
import pandas as pd
from utils.MyFairness import Fairness

from utils.MeLogSingle import MeLogger
from utils.MyPreprocessing import PreprocessingDatasets
from sklearn.model_selection import StratifiedKFold

from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset

import os
import multiprocessing
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

sess = tf.compat.v1.Session()

def processa_imputados_fairness(path:str, model_impt:str, nome_datasets:list, mecanismo:str, classifier:str):

    # Cria diretorios para salvar os resultados do experimento
    os.makedirs(f"./ResultadosFairML/{mecanismo}/{classifier}", exist_ok=True)

    _logger = MeLogger()
    fairness_class = Fairness()

    for nome in nome_datasets: 
        (features_sensitive,
         privileged_groups,
         unprivileged_groups) = Fairness.retorna_featuresFairness(nome) 
        
        for mr in [10,20,40,60]:
            # Read the imputed dataset saved
            imputed_dataset = pd.read_csv(path + f"/{nome}_{model_impt}_md{mr}.csv")
            df = imputed_dataset.copy()
            # Start the Cross-Validation to classification task
            fold = 0
            cv = StratifiedKFold()
            X = df.drop(columns='target')
            y = df['target'].values
            x_cv = X.values

            for train_index, test_index in cv.split(x_cv, y):
                _logger.info(f"{model_impt} = {nome} com md = {mr} no Fold = {fold} classifier = {classifier}")
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
    results_final_tab.to_csv(f"./ResultadosFairML/{mecanismo}/{classifier}/classification_fairness_{model_impt}.csv", index=False)
    
    return _logger.info(f"Resultados salvos com sucesso: {classifier}-{model_impt}")
    

if __name__ == "__main__":
       
    datasets = ["adult",
                "kdd",
                "german_credit", 
                "dutch",
                "bank",                          
                "credit_card",
                "compass_7k",
                "compass_4k",
                "diabetes",
                "ricci",
                "student_math",
                "student_port",
                "law"              
                ]

    mecanismo = "MNAR-determisticFalse"
    classifier_str = "gerry" # "adversarial", "gerry"
    path = f"./DatasetsImputados/{mecanismo}_Multivariado"

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:

            args_list = [(path,"mean",datasets,mecanismo,classifier_str),
                        (path,"customKNN",datasets,mecanismo,classifier_str),
                        (path,"mice",datasets,mecanismo,classifier_str),
                        (path,"softImpute",datasets,mecanismo,classifier_str),
                        (path,"gain",datasets,mecanismo,classifier_str)
                        ]
            
            pool.starmap(processa_imputados_fairness,args_list)

     
    

                