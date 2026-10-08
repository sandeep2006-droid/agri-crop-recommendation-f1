import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

DATA_FILE = 'FINAL_READY_DATASET.csv'


def infer_target_column(df: pd.DataFrame):
    candidate_names = [
        'Suitability_Category',
        'Category',
        'Crop_Category',
        'Label',
        'Target',
        'Class',
        'Suitability_Class',
        'Status',
    ]

    for name in candidate_names:
        if name in df.columns:
            return name

    if 'Suitability_Score' in df.columns:
        return 'Suitability_Score'

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    for name in ['Score', 'Final_Score', 'ML_Score']:
        if name in numeric_cols:
            return name

    raise ValueError(
        'Could not find a target column. Add a label column like Suitability_Category or use Suitability_Score.'
    )


def build_target(df: pd.DataFrame, target_col: str):
    if target_col in ['Suitability_Category', 'Category', 'Crop_Category', 'Label', 'Target', 'Class', 'Suitability_Class', 'Status']:
        return df[target_col].astype(str)

    score_series = pd.to_numeric(df[target_col], errors='coerce')
    bins = [float('-inf'), 0.4, 0.7, float('inf')]
    labels = ['Low', 'Moderate', 'Good']
    return pd.cut(score_series, bins=bins, labels=labels, right=False).fillna('Low')


FEATURES = [
    'Water_Score',
    'Temp_Score',
    'Soil_Score',
    'Avg_Rainfall_mm',
    'Production_MT',
    'District_Temp_Min',
    'District_Temp_Max',
    'Water_Min',
    'Water_Max',
]


def prepare_features(df: pd.DataFrame):
    X = df[FEATURES].copy()
    for col in FEATURES:
        if col not in df.columns:
            raise ValueError(f'Missing required feature column: {col}')
        X[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return X


def train_and_score(X_train, X_test, y_train, y_test, model_name):
    if model_name == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    elif model_name == 'XGBoost':
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False,
            verbosity=0,
        )
    else:
        raise ValueError(f'Unknown model: {model_name}')

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, pred, average='weighted', zero_division=0),
    }


def evaluate_global(df: pd.DataFrame):
    target_col = infer_target_column(df)
    y = build_target(df, target_col)
    X = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    report = []
    for model_name in ['Random Forest', 'XGBoost']:
        metrics = train_and_score(X_train, X_test, y_train, y_test, model_name)
        report.append({
            'Model': model_name,
            'Scope': 'Global',
            'District': 'All',
            **metrics,
        })
    return report


def evaluate_per_district(df: pd.DataFrame):
    target_col = infer_target_column(df)
    district_best_models = {}
    results = []

    for district in sorted(df['District'].dropna().unique()):
        district_df = df[df['District'] == district].copy()
        if len(district_df) < 10:
            continue

        y = build_target(district_df, target_col)
        X = prepare_features(district_df)

        unique_classes = sorted(set(y.tolist()))
        if len(unique_classes) < 2:
            continue

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            continue

        model_scores = {}
        for model_name in ['Random Forest', 'XGBoost']:
            metrics = train_and_score(X_train, X_test, y_train, y_test, model_name)
            model_scores[model_name] = metrics['accuracy']
            results.append({
                'Model': model_name,
                'Scope': 'District',
                'District': district,
                **metrics,
            })

        winner = max(model_scores, key=model_scores.get)
        district_best_models[district] = winner

    return results, district_best_models


def main():
    print('Loading dataset...')
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    print('Dataset rows:', len(df))
    print('Dataset columns:', list(df.columns[:15]), '...')

    global_results = evaluate_global(df)

    district_results, district_best_models = evaluate_per_district(df)

    all_results = global_results + district_results
    result_df = pd.DataFrame(all_results)
    result_df.to_csv('model_results.csv', index=False)

    with open('district_best_models.json', 'w') as f:
        json.dump(district_best_models, f, indent=2)

    report_path = Path('model_comparison_report.txt')
    with report_path.open('w') as f:
        f.write('MODEL COMPARISON REPORT\n')
        f.write('======================\n\n')
        f.write('GLOBAL RESULTS\n')
        f.write('--------------\n')
        for row in global_results:
            f.write(
                f"{row['Model']:15} | Accuracy: {row['accuracy']:.4f} | "
                f"Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f} | "
                f"F1: {row['f1']:.4f}\n"
            )

        f.write('\nDISTRICT WINNERS\n')
        f.write('---------------\n')
        for district, winner in sorted(district_best_models.items()):
            f.write(f'{district}: {winner}\n')

        f.write('\nTOTAL DISTRICTS ANALYZED: ' + str(len(district_best_models)) + '\n')
        rf_wins = sum(1 for winner in district_best_models.values() if winner == 'Random Forest')
        xgb_wins = sum(1 for winner in district_best_models.values() if winner == 'XGBoost')
        f.write(f'Random Forest wins: {rf_wins}\n')
        f.write(f'XGBoost wins: {xgb_wins}\n')

    print('\n=== Global Results ===')
    print(pd.DataFrame(global_results).to_string(index=False))

    print('\n=== District Winners ===')
    for district, winner in sorted(district_best_models.items()):
        print(f'{district}: {winner}')

    print('\nSaved: model_results.csv')
    print('Saved: district_best_models.json')
    print('Saved: model_comparison_report.txt')


if __name__ == '__main__':
    main()
