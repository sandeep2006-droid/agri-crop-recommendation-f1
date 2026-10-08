FINAL SCORE + MODEL COMPARISON FIX

1. Random Forest and XGBoost are trained with the ORIGINAL 9 project features used by the existing UI.
2. The winning model is selected per district using validation accuracy.
3. The app loads the actual winning district model.
4. Model comparison is shown in UI with readable cards matching the existing design.
5. The crop Score uses semantic suitability classes:
   Low = 0.0, Moderate = 0.5, Good = 1.0.
   This is important for districts that contain only two of the three classes.
6. Final Score remains:
   0.6 * original Suitability_Score + 0.4 * ML_Score

Run:
py -m pip install -r requirements.txt
py model_comparison.py
py -m streamlit run app.py
