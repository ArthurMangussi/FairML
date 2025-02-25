# Impact of Missing Data Imputation in Fairness-Aware Machine Learning

This repository contains the codebase for the paper: *Impact of Missing Data Imputation in Fairness-Aware Machine Learning*

## Paper Details
- Authors: Arthur Dantas Mangussi, Ricardo Cardoso Pereira,  Miriam Seoane Santos, Ana Carolina Lorena, and Pedro Henriques Abreu
- Abtract: Missing data, defined as the absence of information in one or more features of a dataset, presents significant challenges in real-world applications. Imputation, the process of replacing missing entries with appropriate estimates, is a widely used solution. However, imputation can propagate existing biases in the data, potentially affecting fairness—an essential principle in responsible Artificial Intelligence. This study investigates the impact of data imputation strategies on fairness within fairness-aware frameworks. We conducted a comprehensive analysis using thirteen benchmark datasets, five state-of-the-art imputation strategies, and varying missing rates (10\%, 20\%, 40\%, and 60\%) under Missing At Random and Missing Not At Random assumptions in a multivariate scenario. Fairness was evaluated using nine group fairness metrics. Our findings demonstrate that combining appropriate imputation strategies with fairness-aware classifiers can result in more equitable and fair outcomes, even under high missing rates.
- Keywords: Fairness-aware Models, Fairness, Missing Data, Missing Data Imputation
- Year: 2025
- Contact: mangussiarthur@gmail.com

## Installation
```bash
git clone https://github.com/ArthurMangussi/FairML.git
cd FairML
pip install -r requirements.txt
```

## Reproducibility
Follow the steps below to reproduce the experiments:
1. **Run the baseline experiments**
Execute the classification for original datasets with fairness-aware classifiers:
```bash
python fairML_baseline.py
python fairML_baseline_more_than_one._sensitive.py
```
2. **Run the missing data imputation experiments**
```bash
python fairML.py
python fairMl_more_than_one_sensitive.py
```
Replace ```ADV``` with ```poison``` or ```evasion```, and ```MD``` with ```mar``` or ```mnar``` or ```mcar``` accordingly.

3. **Combine the final results table**
Combine the results into a unified table using the script:
```bash
python gera_xlsx.py
```

## Acknowledgements
This study was financed, in part, by the São Paulo Research Foundation (FAPESP), Brasil. Process Numbers 2021/06870-3, 2022/10553-6, and 2023/13688-2. This work was also financed through national funds by FCT - Fundação para a Ciência e a Tecnologia, I.P., in the framework of the Project UIDB/00326/2025 and UIDP/00326/2025. Additionally, it was supported by the Portuguese Recovery and Resilience Plan (PRR) through project C645008882-00000055-Center for Responsable AI.
