# Agri-Crop Recommendation Portal

## Run
```powershell
py -m pip install -r requirements.txt
py model_comparison.py
py -m streamlit run app.py
```

The training script evaluates Random Forest and XGBoost separately for each district using a stratified 80/20 test split. The model with higher test accuracy is selected for that district, then retrained on all rows of that district and saved under `models/`. `app.py` loads the selected district model.

If the two accuracies tie, Random Forest is selected as the tie-breaker.
