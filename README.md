# Customer Behaviour Analytics & Churn Prediction (DS-341 Data Mining)

End-to-end data-mining project on the UCI **Online Retail** dataset (~540k transactions, UK e-commerce, 2010–2011): cleaning, RFM feature engineering, leak-free churn classification, K-Means segmentation, Apriori market-basket rules, and a Flask dashboard.

## Highlights
- **Leak-free churn label**: features from purchases before a cutoff, label from the 90 days after it.
- 4 classifiers compared (LogReg, Decision Tree, Random Forest, XGBoost); best ROC-AUC ≈ 0.71.
- 4 RFM customer segments with business recommendations.
- Top association rules for cross-selling.

See [REPORT.md](REPORT.md) for methodology and results.

## Run
1. Download `Online Retail` from the UCI ML Repository (https://archive.ics.uci.edu/dataset/352/online+retail), export as CSV and save it as `data.csv` in the project root.
2. `pip install -r requirements.txt`
3. `python src/data_processor.py` → `python src/models.py` → `python app.py` (http://localhost:5000)
