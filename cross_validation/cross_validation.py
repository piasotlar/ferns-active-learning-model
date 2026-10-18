import argparse
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut


#define columns as categorical or numerical
categorical_cols = ["REALM", "KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY","GENUS", "SPECIES",
                    "MVP", "CP", "POLYPROTEIN","HMMER_MP_HIT","CONSISTENT LENGTH (tax)", "PLANT HOST (tax)"]

numeric_cols = ["LENGTH", "SPADES_COV", "COVERM RPKM", "COVERM READ COUNT","LONGEST ORF LENGTH", 
                "LONGEST ORF COVERAGE", "NUM OF ORFS", "HMMER_MP_EVALUE", "HMMER_MP_SCORE"]

#target column and key columns for merging
target = "YES/NO"
key_cols = ["SAMPLE", "CONTIG NAME"]

#metadata columns to keep track of when a virus/contig was added to the training set and its probabilities and uncertainty at that time
metadata_cols = ["ITERATION_ADDED","PROB_NO_AT_SELECTION","PROB_YES_AT_SELECTION","UNCERTAINTY_AT_SELECTION"]

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-file",
        default="train_pool.tsv"

    )

    parser.add_argument(
        "--output-file",
        default="cross-validation/cross_validation_results_final.csv"
    )

    return parser.parse_args()

#building model, preprocess + XGBClassifier
def build_model(neg_pos_ratio):
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])


    preprocess = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),

            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), [v for v in numeric_cols if v not in ["HMMER_MP_EVALUE","HMMER_MP_SCORE"]]),

            ("evalue", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=1)),
                ("scaler", StandardScaler())
            ]), ["HMMER_MP_EVALUE"]),

            ("score", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler())
            ]), ["HMMER_MP_SCORE"]),
        ]
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=neg_pos_ratio,  
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", xgb_model)
    ])

    return model

def cross_validation(train_file, output_file="cross_validation_results.csv"):
    train = pd.read_csv(train_file, sep="\t", decimal=".")
    labeled = train[train[target].notna()].copy()
    model_cols = categorical_cols + numeric_cols
    X = labeled[model_cols].copy()
    y = labeled[target].map({"No": 0, "Yes": 1})

    loo = LeaveOneOut()
    loo.get_n_splits(X)

    results = []
    n_samples = len(X)
    for i, (train_index, test_index) in enumerate(loo.split(X), start=1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        ratio = neg / pos
        model = build_model(ratio)
        model.fit(X_train, y_train)

        
        probs = model.predict_proba(X_test)[0]
        prob_no = probs[0]
        prob_yes = probs[1]
        pred_numeric = np.argmax(probs)
        predicted_label = "Yes" if pred_numeric == 1 else "No"
        true_label = "Yes" if y_test.iloc[0] == 1 else "No"
        confidence = np.max(probs)
        original_idx = test_index[0]
        row = labeled.iloc[original_idx].to_dict()

        row.update({
            "TRUE_LABEL": true_label,
            "MODEL_PREDICTION": predicted_label,
            "PROB_NO": prob_no,
            "PROB_YES": prob_yes,
            "CONFIDENCE": confidence,
            "CORRECT": true_label == predicted_label
        })

        results.append(row)
        
        print(f"{i}/{n_samples}")


    results_df = pd.DataFrame(results)

    # ustvari mapo, če še ne obstaja
    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(output_file, index=False)

    return results_df

def main():

    args = parse_args()
    results = cross_validation(
        train_file=args.train_file,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()
