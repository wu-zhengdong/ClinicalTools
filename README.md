# Research on Interpretability of Clinical Medicine Based on Machine Learning Predictive Model (基于机器学习预测模型的临床医学可解释性研究)

Please see the "Clinical Research Based on Machine Learning (WorkFlow).ipynb".

ClinicalTools Moudles:
- preprocessing
  - description.py
  - imputation.py
  - standardzation.py
- models 
  - FeatureSelection: LASSO OR RFE
  - GridSearchCV (LR, SVC, GNB, GBM, ADA, MLP)
- metrics
  - evaluate.py
  - plot.py
- SHAP
  - SHAP.py
  - selenium_png.py
  - chromedriver (need to download by yourself: https://chromedriver.chromium.org/)
- utils
  - colors.py
  - load.py (load large cvs file)
