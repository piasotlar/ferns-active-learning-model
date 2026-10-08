import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.impute import SimpleImputer
from visualizations import probability_distribution, uncertainty_distribution, confusion_matrix_heatmap, prediction_histogram, calibration_plot, feature_importance


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

#parse arguments 
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--train-file", default="train_pool.tsv")
    parser.add_argument("--test-file", default="test.tsv")
    parser.add_argument("--validation-batch", default=None)
    parser.add_argument("--add-to-train-batch", default=None)
    parser.add_argument("--stats-file", default="active_learning_stats.tsv")

    parser.add_argument("--batch-size", type=int, default=20)
    return parser.parse_args()

def best_threshold(y_true, probs, name):
    fpr, tpr, thresholds = roc_curve(y_true, probs[:, 1])

    youden = tpr - fpr
    best_idx = np.argmax(youden)

    best_thr = thresholds[best_idx]

    print(
        f"{name}: "
        f"Best threshold = {best_thr:.4f}, "
        f"TPR = {tpr[best_idx]:.4f}, "
        f"FPR = {fpr[best_idx]:.4f}, "
        f"Youden = {youden[best_idx]:.4f}"
    )

    return best_thr

def create_output_dirs():
    dirs = [
        "batches",
        "histograms",
        "predictions",
        "prediction_histograms",
        "uncertainty_histograms",
        "heatmaps",
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

#get predictions, probabilities and uncertainty for a given dataframe
def df_predictions(df, pred, probs, uncertainty):

    df = df.copy()
    df["prediction"] = pred
    df["prob_No"] = probs[:, 0]
    df["prob_Yes"] = probs[:, 1]
    df["uncertainty"] = uncertainty

    return df

#adding annotations from previous iteration batch, to the train_pool file
def update_train(train_df, labeled_batch, iteration):

    labeled_df = pd.read_excel(labeled_batch)
    labeled_filtered = labeled_df[key_cols + [target, "prob_No", "prob_Yes", "uncertainty"]].copy()

    labeled_filtered["ITERATION_ADDED"] = int(iteration)

    labeled_filtered = labeled_filtered.rename(columns={
        "prob_No": "PROB_NO_AT_SELECTION",
        "prob_Yes": "PROB_YES_AT_SELECTION",
        "uncertainty": "UNCERTAINTY_AT_SELECTION"
    })

    train_updated = train_df.merge(labeled_filtered, on=key_cols, how="left", suffixes=("", "_new"))
    train_updated[target] = train_updated[f"{target}_new"].combine_first(train_updated[target])

    for col in metadata_cols:
        new_col = f"{col}_new"
        if new_col in train_updated.columns:
            train_updated[col] = train_updated[new_col].combine_first(train_updated[col])
            train_updated = train_updated.drop(columns=[new_col])

    train_updated = train_updated.drop(columns=[f"{target}_new"])
    
    return train_updated

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

def load_validation_batch(validation_batch):

    model_cols = categorical_cols + numeric_cols
    val = pd.read_excel(validation_batch)

    X_val = val[model_cols].copy()
    y_val = val[target].map({"No": 0, "Yes": 1})

    return X_val, y_val

def permutation_feature_weights(model, X_val, y_val):
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=10,
        random_state=42,
        scoring="f1"
    )

    weights = np.abs(result.importances_mean)

    return weights

def select_high_certainty_batch(model, X_pool, batch_size, df_unlabeled,feature_weights, max_uncertainty = 0.1):

    probs = model.predict_proba(X_pool)
    uncertainty = np.min(probs, axis=1)
    mask = uncertainty < max_uncertainty
    candidate_indices = np.where(mask)[0]
    X_candidates = X_pool.iloc[candidate_indices].copy()

    preprocess = model.named_steps["preprocess"]
    X_candidates_transformed = preprocess.transform(X_candidates)

    if hasattr(X_candidates_transformed, "toarray"):
        X_candidates_transformed = X_candidates_transformed.toarray()

    if feature_weights is None:
        transformed_weights = np.ones(X_candidates_transformed.shape[1])
    else:
        model_cols = categorical_cols + numeric_cols

        feature_weights_series = pd.Series(
            feature_weights,
            index=model_cols
        )

        transformed_feature_names = preprocess.get_feature_names_out()
    
        transformed_weights = []

        for transformed_name in transformed_feature_names:
            name = transformed_name.split("__", 1)[1]

            matched_weight = 0

            for original_col in model_cols:
                if name == original_col or name.startswith(original_col + "_"):
                    matched_weight = feature_weights_series[original_col]
                    break

            transformed_weights.append(matched_weight)

        transformed_weights = np.array(transformed_weights)

    X_candidates_weighted = X_candidates_transformed * np.sqrt(transformed_weights)

    kmeans = KMeans(n_clusters=batch_size, random_state=42).fit(X_candidates_weighted)

    centroids, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_candidates_weighted)

    indices = candidate_indices[centroids]
    batch = df_unlabeled.iloc[indices].copy()
    batch["prob_No"] = probs[indices, 0]
    batch["prob_Yes"] = probs[indices, 1]
    batch["uncertainty"] = uncertainty[indices]

    return batch


#selecting the most uncertain viruses/contigs from the pool to label next + make predictions and plots for the pool
def select_batch(model,X_pool, batch_size, df_unlabeled, iteration, feature_weights, min_uncertainty = 0.1):
    #predict probability
    probs = model.predict_proba(X_pool)
    #calculates uncertainty for every virus/contig
    uncertainty = np.min(probs, axis=1)

    # select viruses with uncertainty >= 0.05
    mask = uncertainty >= min_uncertainty
    candidate_indices = np.where(mask)[0]
    X_candidates = X_pool.iloc[candidate_indices].copy()


    preprocess = model.named_steps["preprocess"]
    X_candidates_transformed = preprocess.transform(X_candidates)

    if hasattr(X_candidates_transformed, "toarray"):
        X_candidates_transformed = X_candidates_transformed.toarray()

    if feature_weights is None:
        transformed_weights = np.ones(X_candidates_transformed.shape[1])
    else:
        model_cols = categorical_cols + numeric_cols

        feature_weights_series = pd.Series(
            feature_weights,
            index=model_cols
        )

        transformed_feature_names = preprocess.get_feature_names_out()
    
        transformed_weights = []

        for transformed_name in transformed_feature_names:
            name = transformed_name.split("__", 1)[1]

            matched_weight = 0

            for original_col in model_cols:
                if name == original_col or name.startswith(original_col + "_"):
                    matched_weight = feature_weights_series[original_col]
                    break

            transformed_weights.append(matched_weight)

        transformed_weights = np.array(transformed_weights)

    X_candidates_weighted = X_candidates_transformed * np.sqrt(transformed_weights)

        
    clustering = AgglomerativeClustering(
        n_clusters=batch_size,
        linkage="ward"
    )

    cluster_labels = clustering.fit_predict(X_candidates_weighted)

    selected_pool_indices = []
    selected_clusters = []

    for cluster_id in range(batch_size):
        cluster_member_positions = np.where(cluster_labels == cluster_id)[0]

        X_cluster = X_candidates_weighted[cluster_member_positions]
        centroid = X_cluster.mean(axis=0)

        closest_position, _ = pairwise_distances_argmin_min(
            centroid.reshape(1, -1),
            X_cluster
        )

        selected_candidate_position = cluster_member_positions[closest_position[0]]
        selected_pool_index = candidate_indices[selected_candidate_position]

        selected_pool_indices.append(selected_pool_index)
        selected_clusters.append(cluster_id)

    selected_pool_indices = np.array(selected_pool_indices)


    to_label = df_unlabeled.iloc[selected_pool_indices].copy()

    to_label["prob_No"] = probs[selected_pool_indices, 0]
    to_label["prob_Yes"] = probs[selected_pool_indices, 1]
    to_label["uncertainty"] = uncertainty[selected_pool_indices]

    to_label = to_label.sort_values("uncertainty", ascending=False)

    #to get predictions for pool
    pred = model.predict(X_pool)
    pool_predictions = df_predictions(df_unlabeled, pred, probs, uncertainty)

    #predictions and histograms
    pool_predictions.to_csv(os.path.join("predictions", f"pool_predictions_iter_{iteration}.csv"), index=False)
    uncertainty_distribution(uncertainty, "pool", iteration, output_dir="uncertainty_histograms")
    probability_distribution(probs,name="pool",iteration=iteration,output_dir="histograms")

    return uncertainty, to_label

#testing the model on the test set, calculating accuracy and f1 score, feature importance, predictions and plots
def test_model(model, test_file, iteration):

    test = pd.read_csv(test_file, sep="\t", decimal=".")
    X_test = test.drop(columns=["YES/NO", "SAMPLE", "CONTIG NAME", "HMMER_MP_TARGET","HMMER_MP_QUERY"])
    y_test = test[target].map({"No": 0, "Yes": 1})
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    #NOVO FEATURE IMPORTANCE
    feature_importance(model, X_test, y_test, iteration, "test")

    #to get predictions for test set
    probs = model.predict_proba(X_test)
    uncertainty = np.min(probs, axis=1)
    test_predictions = df_predictions(test, y_pred, probs, uncertainty)
    roc_auc_test = roc_auc_score(y_test, probs[:, 1])
    calibration_plot(y_test, probs, "test", iteration)
    best_threshold(y_test, probs, "Test")

    #predictions and plots
    test_predictions.to_csv(os.path.join("predictions",f"test_predictions_iter_{iteration}.csv"), index=False)
    uncertainty_distribution(uncertainty,"test",iteration,output_dir="uncertainty_histograms")
    probability_distribution(probs,name="test set",iteration=iteration,output_dir="histograms")
    prediction_histogram(y_test, y_pred, iteration)
    confusion_matrix_heatmap(y_test, y_pred, iteration)
    
    return accuracy, f1, roc_auc_test

#saving stats for each iteration to a tsv file
def stats(stats_file, n_labeled, n_unlabeled, iteration, accuracy, f1, uncertainty,roc_auc_test, roc_auc_train, roc_auc_val):
    stats_row = pd.DataFrame([{
        "iteration": iteration,
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
        "accuracy": accuracy,
        "f1": f1,
        "ROC on test set": roc_auc_test,
        "ROC on train set": roc_auc_train,
        "ROC on validation set": roc_auc_val,
        "uncertainty_over_0_05": (uncertainty > 0.05).sum(),
        "uncertainty_over_0_10": (uncertainty > 0.10).sum(),
        "uncertainty_over_0_15": (uncertainty > 0.15).sum(),
        "uncertainty_over_0_20": (uncertainty > 0.20).sum(),
        "uncertainty_over_0_25": (uncertainty > 0.25).sum(),
        "uncertainty_over_0_30": (uncertainty > 0.30).sum(),
        "uncertainty_over_0_35": (uncertainty > 0.35).sum(),
        "uncertainty_over_0_40": (uncertainty > 0.40).sum(),
        "uncertainty_over_0_45": (uncertainty > 0.45).sum(),
        "uncertainty_over_0_48": (uncertainty > 0.48).sum()
    }])
    if os.path.exists(stats_file):
        old_stats = pd.read_csv(stats_file, sep="\t")
        new_stats = pd.concat([old_stats, stats_row], ignore_index=True)
    else:
        new_stats = stats_row

    new_stats.to_csv(stats_file, sep="\t", index=False)



def main():
    args = parse_args()
    drop_cols = ["YES/NO", "SAMPLE", "CONTIG NAME","HMMER_MP_TARGET", "HMMER_MP_QUERY"]
    create_output_dirs()

    train = pd.read_csv(args.train_file, sep="\t", decimal=".")
    
    #add columns for metadata in iteration 0
    if args.iteration == 0:

        train["ITERATION_ADDED"] = pd.Series(pd.NA, dtype="Int64")
        for col in ["PROB_NO_AT_SELECTION","PROB_YES_AT_SELECTION","UNCERTAINTY_AT_SELECTION"]:
            train[col] = np.nan

        train.loc[train[target].notna(), "ITERATION_ADDED"] = 0
        train.to_csv(args.train_file, sep="\t", index=False)

    if args.iteration > 0:
        if args.validation_batch is None:
            raise ValueError("For iteration > 0, you must provide --validation-batch.")
        
    if args.iteration > 1:
        if args.add_to_train_batch is None:
            raise ValueError("For iteration > 1, you must provide --add-to-train-batch.")

        train = update_train(train, args.add_to_train_batch, args.iteration)
        train.to_csv(args.train_file, sep="\t", index=False)

    df_labeled = train[train[target].notna()].copy().drop(columns=metadata_cols)
    df_unlabeled = train[train[target].isna()].copy().drop(columns=metadata_cols)

    if args.iteration > 0:
        validation_df = pd.read_excel(args.validation_batch)

        validation_keys = set(
            zip(
                validation_df["SAMPLE"],
                validation_df["CONTIG NAME"]
            )
        )

        keep_mask = [
            (sample, contig) not in validation_keys
            for sample, contig in zip(
                df_unlabeled["SAMPLE"],
                df_unlabeled["CONTIG NAME"]
            )
        ]

        df_unlabeled = df_unlabeled[keep_mask].copy()
    
    X_train = df_labeled.drop(columns=drop_cols)
    y_train = df_labeled[target].map({"No": 0, "Yes": 1})
    X_pool = df_unlabeled.drop(columns=drop_cols)

    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    ratio = neg / pos
    model = build_model(ratio)
    model.fit(X_train, y_train)

    feature_weights = None
    roc_auc_val = None
    if args.iteration > 0:

        X_val, y_val = load_validation_batch(args.validation_batch)
        feature_weights = permutation_feature_weights(model, X_val, y_val)
        val_probs = model.predict_proba(X_val)
        calibration_plot(y_val, val_probs, "validation", args.iteration)
        roc_auc_val = roc_auc_score(y_val, val_probs[:, 1])
        best_threshold(y_val, val_probs, "Validation")


    #NOVO FEATURE IMPORTANCE
    feature_importance(model, X_train, y_train, args.iteration, "train")

    #probabilities and uncertainties for train set
    train_probs = model.predict_proba(X_train)
    uncertainty_train = np.min(train_probs, axis=1)

    #predictions za train
    train_pred = model.predict(X_train)
    train_predictions = df_predictions(df_labeled, train_pred, train_probs, uncertainty_train)

    roc_auc_train = roc_auc_score(y_train, train_probs[:, 1])
    best_threshold(y_train, train_probs, "Train")
    calibration_plot(y_train, train_probs, "train", args.iteration)

    #predictions and histograms for train set
    train_predictions.to_csv(os.path.join("predictions",f"train_predictions_iter_{args.iteration}.csv"),index=False)
    probability_distribution(train_probs,name="train",iteration=args.iteration,output_dir="histograms")
    uncertainty_distribution(uncertainty_train,"train",args.iteration,output_dir="uncertainty_histograms")


    uncertainty, uncertain_batch = select_batch(model, X_pool, 20, df_unlabeled, args.iteration, feature_weights)
    certain_batch = select_high_certainty_batch(model, X_pool, 10, df_unlabeled, feature_weights)

    to_label = pd.concat([uncertain_batch, certain_batch])


    accuracy, f1, roc_auc_test = test_model(model, args.test_file, args.iteration)

    output_file = os.path.join("batches",f"to_label_iteration_{args.iteration}.xlsx")
    to_label.to_excel(output_file, index=False)

    stats(args.stats_file, len(df_labeled), len(df_unlabeled), args.iteration, accuracy, f1, uncertainty, roc_auc_test, roc_auc_train, roc_auc_val)

    print(f"Iteration: {args.iteration}")
    print(f"Labeled examples: {len(df_labeled)}")
    print(f"Unlabeled examples: {len(df_unlabeled)}")
    print(f"Accuracy on test set: {accuracy:.4f}")
    print(f"F1 on test set: {f1:.4f}")
    print(f"Saved next batch to: {output_file}")
    print(f"Saved stats to: {args.stats_file}")


if __name__ == "__main__":
    main()


#python model.py --iteration 4 --validation-batch "batches/to_label_iteration_3.xlsx" --add-to-train-batch "batches/to_label_iteration_2.xlsx"
