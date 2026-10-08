FINAL MODEL SELECTION
======================

The app now evaluates BOTH Random Forest and XGBoost for every district.
It selects the model with the higher validation accuracy and uses that
district-specific model for predictions.

The Streamlit UI shows:
- Random Forest validation accuracy
- XGBoost validation accuracy
- Selected model
- Evaluation method

Run:
  py -m pip install -r requirements.txt
  py model_comparison.py
  py -m streamlit run app.py

Do not manually change model names in the JSON. They are generated from
the actual comparison results.
