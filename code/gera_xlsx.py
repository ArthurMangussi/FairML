import os
import pandas as pd


def format_excel(mechanism:str, classifier_fair:str):

    path = f".\\ResultadosFairML\\{mechanism}\\{classifier_fair}"

    arquivos_csv = [arquivo for arquivo in os.listdir(path) if arquivo.endswith(".csv")]

    list_results_csv = {}
    for arq in arquivos_csv:
        _,_,imputation_method = arq.split("_")
        df_resultados = pd.read_csv(os.path.join(path,arq))
        list_results_csv[f"{imputation_method.split('.')[0]}"] = df_resultados

        
    output = f'.\\ResultadosFairML\\{mechanism}\\Resultados_{classifier_fair}_{mechanism}.xlsx'
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:

        for impt_method_name, dataset in list_results_csv.items():
            dataset.to_excel(writer,sheet_name=impt_method_name, index=False)    
    
    print(f"Done! - {mechanism} -- {classifier_fair}")
    
if __name__ == "__main__":

    for md in ["MAR-random", "MNAR-determisticFalse"]:
        for classifier_fair in ["adversarial", "gerry", "meta"]:
            format_excel(mechanism=md,classifier_fair=classifier_fair)