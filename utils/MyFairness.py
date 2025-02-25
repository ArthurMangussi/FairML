# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'

from utils.MeLogSingle import MeLogger
from fairlearn.datasets import (fetch_adult, 
                                fetch_bank_marketing,
                                fetch_credit_card,
                                )
from fairlearn.metrics import (demographic_parity_ratio,                              
                               equalized_odds_ratio,                                
                               equal_opportunity_ratio,
                               demographic_parity_difference,
                               true_negative_rate,
                               MetricFrame)
from sklearn.metrics import confusion_matrix, f1_score
from utils.MyPreprocessing import PreprocessingDatasets

from aif360.algorithms.inprocessing import (GerryFairClassifier,
                                            MetaFairClassifier,
                                            PrejudiceRemover,
                                            AdversarialDebiasing)
from aif360.metrics import ClassificationMetric

import pandas as pd
import numpy as np
import tensorflow as tf

class Fairness:
    def __init__(self):

        self._logger = MeLogger()
        self._prep = PreprocessingDatasets()

        self.results_metrics = {
                    "dataset": [],
                    "fold": [],
                    "eq_odds": [],
                    "av_odds":[],
                    "av_pred_diff":[],
                    "cof_variation_allgroups":[],
                    "entropy_index_allgroups":[],
                    "theil_index_allgroups":[],
                    # "diff_bias_amplification":[],
                    "eq_opportunity": [],
                    "stat_parity": [],
                    "error_rate":[],
                    "false_discovery_rate_dif":[],
                    "false_negative_rate_diff":[],
                    "false_omisson_rate_dif":[],
                    "pred_equality": [],
                    "binary_confusion_matrix":[],
                    "ppv": [],
                    "npv": [],
                    "selection_rate":[],
                    "accuracy": [],
                    "f1-score":[]
                    }

        self.sess = tf.compat.v1.Session()
    # ------------------------------------------------------------------------
    def datasets_fairness_adult(self):
        # Dataset Adult
        data = fetch_adult()
        df = data.data
        target = data.target

        # Pré-processamento considerado no artigo
        df = df.drop("fnlwgt", axis=1).dropna().reset_index(drop=True)

        df.race = [1 if ind == "White" else 0 for ind in df.race] # race as binary feature

        limites = [0,25,60,max(df.age)]
        rotulos = ['<25', '25-60', '>60']
        df["age"] = pd.cut(df.age, bins=limites, labels=rotulos) # discretize age in <25, 25-60, >60
        df["age"] = [1 if ag == "25-60" else 0 for ag in df["age"]]

        df = self._prep.ordinal_encoder(df, ["age",
                                    "education",
                                    "occupation",
                                    "sex"])

        df = self._prep.one_hot_encode(df, ["workclass",
                                    "marital-status",
                                    "relationship",
                                    "native-country"])
        df["target"] = target
        df = self._prep.label_encoder(df, ["target"])

        return df        
    
    # ------------------------------------------------------------------------
    def datasets_fairness_bank(self):
        # Dataset Adult
        data = fetch_bank_marketing()
        df = data.data
        target = data.target

        map_columns = {"V1":"age",
                       "V3":"marital"}
        
        df = df.rename(columns=map_columns)
        df["age"] = [1.0 if idade < 25 or idade > 60 else 0.0 for idade in df.age]
        df["marital"] = [1.0 if status=="married" else 0.0 for status in df.marital]
        df["target"] = target

        df = self._prep.one_hot_encode(df, ["V4",
                                            "V9",
                                            "V16"])
        df = self._prep.ordinal_encoder(df, ["V5",
                                             "V2",
                                             "V7",
                                             "V8",
                                             "V11"])
        df = self._prep.label_encoder(df, ["target"])

        return df

    # ------------------------------------------------------------------------
    def datasets_fairness_credit(self):
        # Dataset Adult
        data = fetch_credit_card()
        df = data.data
        target = data.target

        map_columns = {"x2":"sex",
                       "x3":"education",
                       "x4":"marriage"}
        
        df = df.rename(columns=map_columns)
        st = []
        for status in df["marriage"]:
            if status==0 or status==3:
                st.append(1)
            else:
                st.append(status)
        df["marriage"] = st  
        df["education"] = [1.0 if ed==2 else 0.0 for ed in df.education]   
        df["target"] = target.astype("float64") 
        
        df = self._prep.ordinal_encoder(df, ["x6","x7","x8", "x9", "x10", "x11"])
            
        return df
    
    # ------------------------------------------------------------------------
    def pre_processing_fairness(self, datasets:dict):

        # German credit dataset
        german_credit_df = datasets['german'].copy()
        limites = [0,25,max(german_credit_df.age)]
        rotulos = ["<25", ">25"]
        german_credit_df["age"] = pd.cut(german_credit_df.age, bins=limites, labels=rotulos)
        

        map_gender = {"A91":"male",
                      "A92":"female",
                      "A93":"male",
                      "A94":"male",
                      "A95":"female"}
        
        german_credit_df["personal-status-and-sex"] = german_credit_df["personal-status-and-sex"].map(map_gender)

        german_credit_df = self._prep.ordinal_encoder(german_credit_df, ["age",
                                                                         "checking-account",
                                                                         "savings-account",
                                                                         "employment-since",
                                                                         "telephone",
                                                                         "foreign-worker",
                                                                         "personal-status-and-sex"])
        german_credit_df = self._prep.label_encoder(german_credit_df, ["target"])
        
        german_credit_df = self._prep.one_hot_encode(german_credit_df, ["credit-history",
                                                                        "purpose",
                                                                        "other-debtors",
                                                                        "property",
                                                                        "other-installment",
                                                                        "housing", 
                                                                        "job"])

        # Ricci dataset
        ricci_df = datasets['ricci_processed'].copy()
        ricci_df.Race = [1.0 if ind == "W" else 0.0 for ind in ricci_df.Race] # race as binary feature
        ricci_df = self._prep.label_encoder(ricci_df, ["Position",
                                                       "target"])

        # Datasets da própria biblioteca fairlean
        fairness = Fairness()
        adult_df = fairness.datasets_fairness_adult()
        bank_df = fairness.datasets_fairness_bank()
        credit_df = fairness.datasets_fairness_credit()
        
        # COMPASS recid dataset
        compass_7k_df = datasets["compas-scores-two-years_clean"].copy()
        clean_compass_7k = compass_7k_df.drop(columns=["id", "name", "first", "last", "compas_screening_date", "dob", "days_b_screening_arrest",
                                                       "c_jail_in", "c_jail_out", "c_case_number", "c_offense_date", "c_arrest_date", "age_cat",
                                                       "vr_case_number", "vr_offense_date", "decile_score.1", "r_case_number", "r_offense_date",
                                                       "screening_date", "v_screening_date", "in_custody", "out_custody", "priors_count.1",
                                                       "r_jail_in","r_jail_out", "vr_charge_degree", "vr_charge_desc", "v_type_of_assessment",
                                                       "type_of_assessment", "violent_recid","r_charge_degree", "r_days_from_arrest", "c_charge_desc", "r_charge_desc"])
        map_races_compass = {"African-American":1, "Caucasian":0, "Hispanic":0, "Other":0, "Asian":0, "Native American":0}
        clean_compass_7k["race"] = clean_compass_7k["race"].map(map_races_compass)
        
        map_colum = {"two_year_recid":"target"}
        clean_compass_7k = clean_compass_7k.rename(columns=map_colum)
        clean_compass_7k = self._prep.label_encoder(clean_compass_7k, ["target"])
        clean_compass_7k = self._prep.ordinal_encoder(clean_compass_7k, ["sex", 
                                                                         "c_charge_degree"                                                                         
                                                                         ])
        clean_compass_7k = self._prep.one_hot_encode(clean_compass_7k, ["score_text",
                                                                        "v_score_text"                                                                       ,
                                                                        ])

        # COMPASS viol recid dataset
        compass_4k_df = datasets["compas-scores-two-years-violent_clean"].copy()
        clean_compass_4k = compass_4k_df.drop(columns=["id", "name", "first", "last", "compas_screening_date", "dob", "days_b_screening_arrest",
                                                       "c_jail_in", "c_jail_out", "c_case_number", "c_offense_date", "c_arrest_date", "age_cat",
                                                       "vr_case_number", "vr_offense_date", "decile_score.1", "r_case_number", "r_offense_date",
                                                       "screening_date", "v_screening_date", "in_custody", "out_custody", "priors_count.1",
                                                       "r_jail_in","r_jail_out", "vr_charge_degree", "vr_charge_desc", "v_type_of_assessment",
                                                       "type_of_assessment", "violent_recid", "r_charge_degree", "r_days_from_arrest", "c_charge_desc",
                                                       "r_charge_desc"]).dropna().reset_index(drop=True)
        clean_compass_4k = clean_compass_4k.rename(columns=map_colum)
        clean_compass_4k["race"] = clean_compass_4k["race"].map(map_races_compass)
        clean_compass_4k = self._prep.label_encoder(clean_compass_4k, ["target"])
        clean_compass_4k = self._prep.ordinal_encoder(clean_compass_4k, ["sex", 
                                                                         "c_charge_degree"                                                                         
                                                                         ])
        clean_compass_4k = self._prep.one_hot_encode(clean_compass_4k, ["score_text",
                                                                        "v_score_text"
                                                                        ])

        # Dutch dataset
        dutch_df = datasets["dutch"].copy()
        map_dutch = {"occupation":"target"}
        dutch_df = dutch_df.rename(columns=map_dutch)
        dutch_df = self._prep.label_encoder(dutch_df, ["target"])
        dutch_df = self._prep.ordinal_encoder(dutch_df, ["sex", "household_size", "cur_eco_activity"])
        dutch_df = self._prep.one_hot_encode(dutch_df, ["household_position",
                                                        "country_birth",
                                                        "edu_level",
                                                        "economic_status",
                                                        "marital_status"])
        
        # Diabetes dataset
        diabetes_df = datasets["diabetes-clean"].copy()
        diabetes_df = diabetes_df.drop(columns = ["encounter_id", 
                                                  "patient_nbr",
                                                  "metformin-rosiglitazone",
                                                  "metformin-pioglitazone",
                                                  "citoglipton",
                                                  "examide",
                                                  "A1Cresult",
                                                  "max_glu_serum"
                                                  ])
        diabetes_df = self._prep.label_encoder(diabetes_df,["target"])
        diabetes_df = self._prep.ordinal_encoder(diabetes_df, ["age",
                                                               "diabetesMed",
                                                               "glimepiride-pioglitazone",
                                                               "glipizide-metformin",
                                                               "troglitazone",
                                                               "tolbutamide",
                                                               "acetohexamide",
                                                               "discharge_disposition_id",
                                                               "admission_source_id",
                                                               "diag_1",
                                                               "diag_2",
                                                               "diag_3",
                                                               "acetohexamide",
                                                               "tolbutamide",
                                                               "gender",
                                                               "change"
                                                               ])
        diabetes_df = self._prep.one_hot_encode(diabetes_df, ["race", 
                                                              "metformin",
                                                              "chlorpropamide",
                                                              "glipizide",
                                                              "rosiglitazone",
                                                              "acarbose",
                                                              "miglitol",                                                                                                         
                                                              "repaglinide",
                                                              "nateglinide",
                                                              "glimepiride",
                                                              "glyburide",
                                                              "pioglitazone",
                                                              "tolazamide",
                                                              "insulin",
                                                              "glyburide-metformin"
                                                              ])

        # Law dataset
        law_df = datasets["law_school_clean"].copy()
        law_df = self._prep.label_encoder(law_df, ["target"])
        law_df.race = [1.0 if r == "White" else 0.0 for r in law_df.race]
        law_df = self._prep.one_hot_encode(law_df, ["fam_inc"])

        # Student mathematics dataset
        student_mat_df = datasets["student-mat"].copy()
        student_mat_df.target = [1.0 if nota>= 10 else 0.0 for nota in student_mat_df.target]
        student_mat_df.age = [1.0 if idade >= 18 else 0.0 for idade in student_mat_df.age]

        student_mat_df = self._prep.ordinal_encoder(student_mat_df, ["school",
                                                                     "sex",
                                                                     "address",
                                                                     "famsize",
                                                                     "Pstatus",
                                                                     "schoolsup",
                                                                     "famsup",
                                                                     "paid",
                                                                     "activities",
                                                                     "nursery",
                                                                     "higher",
                                                                     "internet",
                                                                     "romantic"])
        student_mat_df = self._prep.one_hot_encode(student_mat_df, ["Mjob",
                                                                    "Fjob",
                                                                    "reason",
                                                                    "guardian"])

        # Student portuguese dataset
        student_port_df = datasets["student-por"].copy()
        student_port_df.target = [1 if nota>= 10 else 0 for nota in student_port_df.target]
        student_port_df.age = [1 if idade >= 18 else 0 for idade in student_port_df.age]

        student_port_df = self._prep.ordinal_encoder(student_port_df, ["school",
                                                                     "sex",
                                                                     "address",
                                                                     "famsize",
                                                                     "Pstatus",
                                                                     "schoolsup",
                                                                     "famsup",
                                                                     "paid",
                                                                     "activities",
                                                                     "nursery",
                                                                     "higher",
                                                                     "internet",
                                                                     "romantic"])
        student_port_df = self._prep.one_hot_encode(student_port_df, ["Mjob",
                                                                    "Fjob",
                                                                    "reason",
                                                                    "guardian"])

        # KDD census income
        census_df = datasets["kdd-census-income-clean"].copy()
        census_df.race = [1.0 if r == "White" else 0 for r in census_df.race]
        limites = [-1,25,60,max(census_df.age)]
        rotulos = ['<25', '25-60', '>60']
        census_df["age"] = pd.cut(census_df.age, bins=limites, labels=rotulos)
        census_df["age"] = [1 if ag == "25-60" else 0 for ag in census_df["age"]]

        census_df = self._prep.label_encoder(census_df, ['target'])
        census_df = self._prep.ordinal_encoder(census_df, ['year',
                                                           'age',
                                                           'industry',
                                                           'occupation',
                                                           'education',
                                                           'detailed-household-and-family-stat',
                                                           'major-industry',
                                                           'major-occupation',
                                                           'country-father',
                                                           'country-mother',
                                                           'country-birth',
                                                           'sex',
                                                           'state-previous-residence'
                                                           ])
        census_df = self._prep.one_hot_encode(census_df, ['workclass',
                                                          'employment-status',                                                          
                                                          'detailed-household-summary-in-household',
                                                          'family-members-under-18',
                                                          'citizenship',
                                                          'own-business',
                                                          'veterans-benefits',
                                                          'enroll-in-edu-inst-last-wk',
                                                          'hispanic-origin',                                                          
                                                          'reason-unemployment',                                                          
                                                          'live-hour-1-year-ago',                                                          
                                                          'fill-questionnaire',
                                                          'marital-status',
                                                          'memner-union',
                                                          'tax-filter-stat',
                                                          'region-previous-residence'])
        
        census_df = census_df.sample(n=60000, random_state=42)

        return (adult_df,
                census_df,
                german_credit_df,
                dutch_df,
                bank_df,
                credit_df,
                clean_compass_7k,
                clean_compass_4k,
                diabetes_df,
                ricci_df,
                student_mat_df,
                student_port_df,
                law_df)
    
    
    # ------------------------------------------------------------------------
    @staticmethod
    def cria_tabela_fairness(datasets:dict):
        tabela_resultados = {}
        fairness_instance = Fairness()
        
        (adult_df,
        census_df,
        german_credit_df,
        dutch_df,
        bank_df,
        credit_df,
        clean_compass_7k,
        clean_compass_4k,
        diabetes_df,
        ricci_df,
        student_mat_df,
        student_port_df,
        law_df) = fairness_instance.pre_processing_fairness(datasets)

        tabela_resultados["datasets"] = [adult_df,
                                        census_df,
                                        german_credit_df,
                                        dutch_df,
                                        bank_df,
                                        credit_df,
                                        clean_compass_7k,
                                        clean_compass_4k,
                                        diabetes_df,
                                        ricci_df,
                                        student_mat_df,
                                        student_port_df,
                                        law_df]
        
        tabela_resultados["nome_datasets"] = ["adult",
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
        
        tabela_resultados["missing_rate"] = [10,20,40,60]

        return tabela_resultados
    
    # ------------------------------------------------------------------------
    @staticmethod
    def predictive_equality(y_true, y_pred, sensitive_features, method="between_groups", sample_weight=None):

        fpr = MetricFrame(metrics=true_negative_rate,
                        y_true=y_true,
                        y_pred=y_pred,
                        sensitive_features=sensitive_features,
                        sample_params={"sample_weight": sample_weight})
        result = fpr.ratio(method=method)

        return result
    
    # ------------------------------------------------------------------------
    @staticmethod
    def positive_predictive_value(y_true,y_pred):
        cm = confusion_matrix(y_true,y_pred)
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
        fp = cm[0, 1] if cm.shape[1] > 1 else 0
        return tp / (tp + fp) if (tp + fp) > 0 else np.nan

    # ------------------------------------------------------------------------
    @staticmethod
    def negative_predictive_value(y_true,y_pred):
        cm = confusion_matrix(y_true,y_pred)
        tn = cm[0, 0] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[1] > 1 else 0
        return tn / (tn + fn) if (tn + fn) > 0 else np.nan
    
    # ------------------------------------------------------------------------
    @staticmethod
    def equality_positive_predicted_values(y_true,y_pred,sensitive_features, method="between_groups")->float:
        ppv = MetricFrame(
                metrics=Fairness.positive_predictive_value,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
                )
        result = ppv.ratio(method=method)
        return result 

    # ------------------------------------------------------------------------
    @staticmethod
    def equality_negative_predicted_values(y_true,y_pred,sensitive_features, method="between_groups")->float:
        npv = MetricFrame(
                metrics=Fairness.negative_predictive_value,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
                )
        result = npv.ratio(method=method)
        return result 
    
    # ------------------------------------------------------------------------
    @staticmethod
    def calcula_metricas_fairness(y_original, y_predito, val_protecte):

        metrics = {}
        
        statistical_parity = demographic_parity_ratio(y_true=y_original,
                                                y_pred=y_predito,
                                                sensitive_features=val_protecte)
        
        cv_score = demographic_parity_difference(y_true=y_original,
                                                y_pred=y_predito,
                                                sensitive_features=val_protecte)
        
        eq_odds_ratio = equalized_odds_ratio(y_true=y_original,
                                                y_pred=y_predito,
                                                sensitive_features=val_protecte)
        
        eq_oppor_ratio = equal_opportunity_ratio(y_true=y_original,
                                                y_pred=y_predito,
                                                sensitive_features=val_protecte)
        
        predictive_equality = Fairness.predictive_equality(y_true=y_original,
                                                           y_pred=y_predito,
                                                           sensitive_features=val_protecte)
        
        equality_ppv = Fairness.equality_positive_predicted_values(y_true=y_original,
                                                           y_pred=y_predito,
                                                           sensitive_features=val_protecte)
        equality_npv = Fairness.equality_negative_predicted_values(y_true=y_original,
                                                           y_pred=y_predito,
                                                           sensitive_features=val_protecte)

        metric_frame = MetricFrame(metrics=confusion_matrix,
            y_true=y_original,
            y_pred=y_predito,
            sensitive_features=val_protecte)
        
        true_negative, false_positive, false_negative, true_positive = metric_frame.overall.ravel()
        
        metrics["statistical_parity"] = statistical_parity
        metrics["eq_odds_ratio"] = eq_odds_ratio
        metrics["eq_oppor_ratio"] = eq_oppor_ratio
        metrics["predictive_equality"] = predictive_equality
        metrics["equality_positive_predicted_values"] = equality_ppv
        metrics["equality_negative_predicted_values"] = equality_npv
        metrics["TP"] = true_positive # não considera a divisão por grupo das variáveis protegidas
        metrics["TN"] = true_negative # não considera a divisão por grupo das variáveis protegidas
        metrics["FP"] = false_positive # não considera a divisão por grupo das variáveis protegidas
        metrics["FN"] = false_negative # não considera a divisão por grupo das variáveis protegidas

        return metrics
    
    # ------------------------------------------------------------------------
    @staticmethod
    def retorna_featuresFairness(nome:str) -> dict:

        if nome == "adult":
            features_protect = ["age","race","sex"]
            privileged_groups = [{"age":1.0, "race":1, "sex":1.0}]
            unprivileged_groups = [{"age":0.0, "race":0, "sex":0.0}]

        elif nome == "ricci":
            features_protect = ["Race"]
            privileged_groups = [{"Race":1.0}]
            unprivileged_groups = [{"Race":0.0}]

        elif nome == "german_credit":
            features_protect = ["age", "personal-status-and-sex"]
            privileged_groups = [{"age":1.0,"personal-status-and-sex":1.0}]
            unprivileged_groups = [{"age":0.0,"personal-status-and-sex":0.0}]

        elif nome == "bank":
            features_protect = ["age", "marital"]
            privileged_groups = [{"age":0.0, "marital":1.0}]
            unprivileged_groups = [{"age":1.0, "marital":0.0}]

        elif nome == "credit_card":
            features_protect = ["sex", "education","marriage"]
            privileged_groups = [{"sex":1.0,"education":1.0,"marriage":1.0}]
            unprivileged_groups = [{"sex":0.0,"education":0.0,"marriage":0.0}]

        elif nome == "student_math" or nome == "student_port":
            features_protect= ["sex", "age"]
            privileged_groups = [{"sex":0.0, "age":0.0}]
            unprivileged_groups = [{"sex":1.0, "age":1.0}]

        elif nome == "compass_7k" or nome == "compass_4k":
            features_protect = ["race", "sex"]
            privileged_groups = [{"race":1, "sex":1.0}]
            unprivileged_groups = [{"race":0, "sex":0.0}]

        elif nome == "law":
            features_protect = ["race","gender"]
            privileged_groups = [{"race":1.0, "gender":1.0}]
            unprivileged_groups = [{"race":0.0, "gender":0.0}]

        elif nome == "diabetes":
            features_protect = ["gender"]
            privileged_groups = [{"gender":0.0}]
            unprivileged_groups = [{"gender":1.0}]

        elif nome == "kdd":
            features_protect = ["sex","race"]
            privileged_groups = [{"sex":0.0, "race":1.0}]
            unprivileged_groups = [{"sex":1.0, "race":0.0}]

        elif nome == "dutch":
            features_protect = ["sex"]
            privileged_groups = [{"sex":0.0}]
            unprivileged_groups = [{"sex":1.0}]

        return features_protect, privileged_groups, unprivileged_groups
    
    # ------------------------------------------------------------------------
    @staticmethod
    def choose_model_fair(classifier:str,
                          unprivileged_groups,
                          privileged_groups,
                          session,
                          sensitive_attr=None):
        match classifier:
            case "gerry":
                clf = GerryFairClassifier(C=1)
            
            case "meta":
                clf = MetaFairClassifier(sensitive_attr=sensitive_attr, #Só trata uma variável sensível por vez
                                         seed=123,
                                         tau=0,
                                         type="sr")
            
            case "prejudice":
                clf = PrejudiceRemover(class_attr="target",
                                       sensitive_attr=sensitive_attr,
                                       eta=0.8)
            
            
            case "adversarial":
                    clf = AdversarialDebiasing(unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups,
                                            seed=42,
                                            scope_name='debiased_classifier',
                                            sess=session
                                            )

        return clf

    # ------------------------------------------------------------------------
    def calculate_metrics(self,
                          nome:str, 
                          fold:str, 
                          classification_metric:ClassificationMetric, 
                          y_true:np.ndarray, 
                          y_pred:np.ndarray):
        # Cálculo das métricas de fairness
        self.results_metrics["dataset"].append(nome)
        self.results_metrics["fold"].append(fold)
        self.results_metrics["accuracy"].append(classification_metric.accuracy())
        self.results_metrics["eq_odds"].append(classification_metric.equalized_odds_difference())
        self.results_metrics["av_odds"].append(classification_metric.average_odds_difference())
        self.results_metrics["av_pred_diff"].append(classification_metric.average_predictive_value_difference())
        self.results_metrics["cof_variation_allgroups"].append(classification_metric.between_all_groups_coefficient_of_variation())
        self.results_metrics["entropy_index_allgroups"].append(classification_metric.between_all_groups_generalized_entropy_index())
        self.results_metrics["theil_index_allgroups"].append(classification_metric.between_all_groups_theil_index())
        # self.results_metrics["diff_bias_amplification"].append(classification_metric.differential_fairness_bias_amplification())
        self.results_metrics["eq_opportunity"].append(classification_metric.equal_opportunity_difference())
        self.results_metrics["stat_parity"].append(classification_metric.statistical_parity_difference())
        self.results_metrics["error_rate"].append(classification_metric.error_rate_difference())
        self.results_metrics["false_discovery_rate_dif"].append(classification_metric.false_discovery_rate_difference())
        self.results_metrics["false_negative_rate_diff"].append(classification_metric.false_negative_rate_difference())
        self.results_metrics["false_omisson_rate_dif"].append(classification_metric.false_omission_rate_difference())
        self.results_metrics["pred_equality"].append(classification_metric.false_positive_rate_difference())
        self.results_metrics["binary_confusion_matrix"].append(classification_metric.generalized_binary_confusion_matrix())
        self.results_metrics["ppv"].append(classification_metric.positive_predictive_value())
        self.results_metrics["npv"].append(classification_metric.negative_predictive_value())
        self.results_metrics["selection_rate"].append(classification_metric.selection_rate())
        
        # Cálculo do F1-score
        self.results_metrics["f1-score"].append(f1_score(y_true=y_true, y_pred=y_pred))

                                            