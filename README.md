# Commercial Fleet Claim Cost Prediction

## 📌 Executive Summary
Developed during a competitive hackathon, this project predicts maintenance and claim costs for commercial trucking fleets. By merging disparate claim logs and truck option codes, this pipeline utilizes advanced ensemble modeling (Random Forest and XGBoost) to isolate the vehicle attributes that drive the highest financial impact.

## 🛠️ Tools & Methodology
* **Stack:** Python (Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib)
* **Methodology:** * Automated data merging and categorical encoding.
  * Tree-based feature importance extraction.
  * Gradient Boosting Regression (XGBoost).
  * Robust model evaluation via MAE, RMSE, and 5-Fold Cross-Validation.

## 📈 Key Insights
* Successfully unified siloed logistics datasets (`Claim Information` and `Option Codes`) into a single ML-ready pipeline.
* Deployed XGBoost to identify specific truck attributes and labor costs that heavily influence total scaled claim costs.
* Achieved rigorous model validation using Cross-Validated Mean Absolute Error to ensure predictive stability across different fleet samples.

## 💻 Reproduction Instructions
```bash
git clone [https://github.com/g1d33p/Fleet-Claim-Cost-Prediction.git](https://github.com/g1d33p/Fleet-Claim-Cost-Prediction.git)
cd Fleet-Claim-Cost-Prediction
pip install -r requirements.txt
python src/train_claim_model.py --claims_path data/claims.xlsx --options_path data/options.xlsx
