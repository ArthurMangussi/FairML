# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'


import os
import pandas as pd
from scipy.io import arff
from io import StringIO


# Bibliotecas
import pandas as pd
from scipy.stats import norm
import warnings

# Ignorar todos os avisos
warnings.filterwarnings("ignore")


# ==========================================================================
class MyPipeline:   
    # ------------------------------------------------------------------------
    @staticmethod
    def pre_imputed_dataset(data):
        fill_na = {}
        for col_missing in data[data.isna()]:
            media = data[col_missing].mean()
            std = data[col_missing].std()
            tam_sample = data[col_missing].isna().sum()
            index_nan = data[col_missing][data[col_missing].isna()].index

            valores_preencher_miss = norm.rvs(loc=media,
                                            scale=std,
                                            size=tam_sample)
            
            
            dict_nan = dict(zip(index_nan, valores_preencher_miss))
            fill_na[col_missing] = dict_nan
            
        dataset_pre_imputed = data.fillna(fill_na)
        return dataset_pre_imputed 
    # ------------------------------------------------------------------------
    @staticmethod
    def create_dirs(mechanism:str, approach:str):
        """
        Função para criar os diretórios de armazenamento para analisar os resultados.

        Args:
            mechanism (str): nome do mecanismo que os missing serão gerados
            approach (str): Abordagem Multivariado ou Univariado
        """
        os.makedirs(f'./Análises Resultados/Tempos/{mechanism}_{approach}', exist_ok=True)
        os.makedirs(f'./Análises Resultados/Classificação/{mechanism}_{approach}', exist_ok=True)
        os.makedirs(f'./Resultados Parciais Multivariado/{mechanism}_{approach}', exist_ok=True)
        os.makedirs(f"./Análises Resultados/Complexidade/{mechanism}_{approach}/baseline", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
    # ------------------------------------------------------------------------
    @staticmethod
    def cria_dataframe(df:pd.DataFrame) -> pd.DataFrame:
        """
        Função para criar um pandas DataFrame a partir de datasets da biblioteca do sklearn.datasets

        Args:
            df: Um objeto pandas DataFrame.

        Returns:
            Um objeto pandas DataFrame contendo os dados do DataFrame de entrada (df) com uma coluna adicional chamada 'target'.
        """
        dataset = pd.DataFrame(data=df.data, columns=df.feature_names)
        dataset["target"] = df.target
        return dataset

    # ------------------------------------------------------------------------
    @staticmethod
    def split_dataset(dataset: pd.DataFrame, perc_treino: float, perc_teste: float):
        """
        Divide o dataset dado nos conjunto de treino, teste e validação.

        Args:
            dataset (pd.DataFrame): A pandas DataFrame contendo o dataset a ser dividido.
            perc_treino (float): A porcentagem do dataset que será usada para treinamento
            perc_teste (float): A porcentagem do dataset que será usada para teste.

        Returns:
            tuple: A tuple contendo três numpy arrays: (X_treino, X_teste, X_valida)
        """

        dataset = dataset.copy()
        df_shuffle = dataset.sample(frac=1.0, replace=True)

        tamanho_treino = int(perc_treino * len(dataset))
        tamanho_teste = int(perc_teste * len(dataset))

        x_treino = df_shuffle.iloc[:tamanho_treino]
        x_teste = df_shuffle.iloc[tamanho_treino : tamanho_treino + tamanho_teste]
        x_valida = df_shuffle.iloc[tamanho_treino + tamanho_teste :]

        return x_treino, x_teste, x_valida

    # ------------------------------------------------------------------------
    @staticmethod
    def encode_features_categoricas(nome:str, dados:pd.DataFrame)->pd.DataFrame:
        match nome:
            case "adult":
                lista_continuas = ["education", "education-num", "occupation"]
            case "kdd":
                lista_continuas = ["industry", "occupation", "education", "wage-per-hour",
                            "major-industry", "major-occupation", "capital-gain",
                            "capital-loss", "dividends-from-stocks", "state-previous-residence",
                            "detailed-household-and-family-stat", "num-persons-worked-for-employer",
                            "country-father", "country-mother", "country-birth", "weeks-worked"]
            case "german_credit":
                lista_continuas = ["checking-account", "duration", "credit-amount", "savings-account",
                            "employment-since", "installment-rate", "residence-since", "existing-credits",
                            "number-people-provide-maintenance"]
            case "dutch":
                lista_continuas = ["household_size", "cur_eco_activity", "citizenship"]
            case "bank":
                lista_continuas = ["V2", "V5", "V7", "V8", "V11", "V6", "V10", "V12", "V13", "V14", "V15"]
            case "credit_card":
                lista_continuas = ["x1", "x5", "x6", "x7", "x8", "x9",
                            "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19",
                            "x20", "x21", "x22", "x23"]
            case "diabetes":
                lista_continuas = ["age", "time_in_hospital", "num_procedures", "num_medications", "number_outpatient",
                            "number_emergency", "number_inpatient", "discharge_disposition_id", "admission_source_id",
                            "num_lab_procedures", "diag_1", "diag_2", "diag_3", "number_diagnoses"]
            case "compass_4k":
                lista_continuas = ["age", "juv_fel_count", "decile_score", "juv_misd_count",
                            "juv_other_count", "priors_count", "c_days_from_compas", "v_decile_score", "start", "end"]
            case "compass_7k":
                lista_continuas = ["age", "juv_fel_count", "decile_score", "juv_misd_count",
                            "juv_other_count", "priors_count", "c_days_from_compas", "v_decile_score", "start", "end"]
            case "ricci":
                lista_continuas = ["Oral", "Written", "Combine"]
            case "student_port":
                lista_continuas = ["Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout",
                            "Dalc", "Walc", "health", "absences", "G1", "G2"]
            case "student_math":
                lista_continuas = ["Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout",
                            "Dalc", "Walc", "health", "absences", "G1", "G2"]
            case "law":
                lista_continuas = ["decile1b", "decile3", "lsat", "ugpa", "zfygpa", "zgpa", "tier"]

        for col in dados.columns:
            if col not in lista_continuas:
                dados[col] = [1.0 if valor >= 0.5 else 0.0 for valor in dados[col]]

        return dados
    # ------------------------------------------------------------------------
    @staticmethod
    def retorna_dict_types_datasetsFairness() -> dict:
        return {"adult": [True,  True,  False,  True,  True,  True,  False,  False,  False,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True],
                       "ricci": [True, False, False, True, False, True],
                       "german_credit": [True,  False,  False,  True,  True,  False,  True,  False,  True,
                                        False,  False,  True,  True,  True,  True,  True,  True,  True,
                                        True,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True],
                       "bank": [True,  True,  True,  True,  False,  True,  True,  False,  True,
                                False,  False,  False,  False,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True],
                       "credit_card": [False, True, True, True, False, True, True, True, True,
                                        True, True, False, False, False, False, False, False, False,
                                        False, False, False, False, False, True],
                       "student_math": [True,  True,  True,  True,  True,  True,  False,  False,  False,
                                        False,  False,  True,  True,  True,  True,  True,  True,  True,
                                        True,  False,  False,  False,  False,  False,  False,  False,  False,
                                        False,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True],
                       "student_port": [True,  True,  True,  True,  True,  True,  False,  False,  False,
                                        False,  False,  True,  True,  True,  True,  True,  True,  True,
                                        True,  False,  False,  False,  False,  False,  False,  False,  False,
                                        False,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True],
                       "compass_7k": [True,  False,  True,  False,  False,  False,  False,  False,  False,
                                    False,  True,  True,  False,  False,  False,  True,  True,  True,
                                    True,  True,  True,  True,  True],
                       "compass_4k": [True,  False,  True,  False,  False,  False,  False,  False,  False,
                                    False,  True,  True,  False,  False,  False,  True,  True,  True,
                                    True,  True,  True,  True,  True],
                       "law":[False, False, False, False, False, False, True, True, True,
                              True, True, True, True, True, True, True],
                       "diabetes": [True,  False,  True,  True,  True,  False,  False,  False,  False,
                                    False,  False,  False,  True,  True,  True,  False,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True],
                       "kdd": [ False,  True,  True,  True,  False,  True,  True,  True,  True,
                                False,  False,  False,  True,  True,  False,  True,  True,  True,
                                False,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True], 
                       "dutch": [True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True]}
    
    
    # ------------------------------------------------------------------------
    @staticmethod
    def carrega_datasets(path_datasets: str) -> dict:
        """
        Carregue conjuntos de dados de um determinado caminho de diretório e retorne-os como um dicionário.

        Argumentos:
            path_datasets (str): O caminho para o diretório que contém os conjuntos de dados.

        Retorna:
            dict: Um dicionário contendo os conjuntos de dados carregados, onde as chaves são os nomes dos arquivos e os valores são DataFrames do pandas.

        Examplo:
            datasets = carrega_datasets('/path/to/datasets')
            print(datasets)
            # Output: {'dataset1': DataFrame1, 'dataset2': DataFrame2, ...}
        """
        datasets_carregados = {}

        for diretorio, subdiretorios, arquivos in os.walk(path_datasets):
            for nome_arquivo in arquivos:
                caminho_completo = os.path.join(diretorio, nome_arquivo)
                nome, extensao = os.path.splitext(nome_arquivo)

                if nome == "._.DS_Store" or extensao == ".names":
                    continue

                if extensao == ".csv" or extensao == ".data":
                    dados = pd.read_csv(caminho_completo)
                    datasets_carregados[nome] = dados

                elif extensao == ".arff":
                    with open(caminho_completo, "r") as f:
                        data = f.read()

                    buffer_texto = StringIO(data)
                    dados, meta = arff.loadarff(buffer_texto)
                    # Convert the numpy array into a dictionary
                    dados_dict = {name: dados[name] for name in dados.dtype.names}

                    # Decode the values to remove the 'b' prefix from the values
                    dados_decodificados = {
                        k: [x.decode() if isinstance(x, bytes) else x for x in v]
                        for k, v in dados_dict.items()
                    }

                    # Convert the decoded data to a pandas DataFrame
                    df = pd.DataFrame(dados_decodificados)
                    datasets_carregados[nome] = df

                elif extensao == ".xlsx":
                    dados = pd.read_excel(caminho_completo)

                    datasets_carregados[nome] = dados

                else:
                    raise ValueError(f"Formato de arquivo não encontrado: {extensao}")

        return datasets_carregados
    
    

    