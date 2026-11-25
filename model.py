import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.impute import SimpleImputer
from visualizations import probability_distribution, uncertainty_distribution, confusion_matrix_heatmap, prediction_histogram, calibration_plot, feature_importance

# Variables

CATEGORICAL_COLS = ["REALM", "KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY","GENUS", "SPECIES","MVP", "CP", "POLYPROTEIN","HMMER_MP_HIT","CONSISTENT LENGTH (tax)", "PLANT HOST (tax)"]
NUMERIC_COLS = ["LENGTH", "SPADES_COV", "COVERM RPKM", "COVERM READ COUNT","LONGEST ORF LENGTH", "LONGEST ORF COVERAGE", "NUM OF ORFS", "HMMER_MP_EVALUE", "HMMER_MP_SCORE"]
TARGET = "YES/NO"
KEY_COLS = ["SAMPLE", "CONTIG NAME"]
METADATA_COLS = ["ITERATION_ADDED","PROB_NO_AT_SELECTION","PROB_YES_AT_SELECTION","UNCERTAINTY_AT_SELECTION"]
DROP_COLS = ["YES/NO", "SAMPLE", "CONTIG NAME","HMMER_MP_TARGET", "HMMER_MP_QUERY"]

NEGATIVE_THRESHOLD = 0.05
POSITIVE_THRESHOLD = 0.95 
NEGATIVE_BATCH_SIZE = 10
POSITIVE_BATCH_SIZE = 20

sampling_version = 4 # options: 1, 2, 3, 4


# creates output directories for results (batches, histograms, predictions, prediction_histograms, uncertainty_histograms, heatmaps)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--train-file", default="train_pool.tsv")
    parser.add_argument("--test-file", default="test.tsv")
    parser.add_argument("--validation-batch", default=None)
    parser.add_argument("--add-to-train-batch", default=None)
    parser.add_argument("--stats-file", default="active_learning_stats.tsv")

    return parser.parse_args()

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

# loads train data and adds labeled data

def update_train(train_df, labeled_batch, iteration):

    labeled_df = pd.read_excel(labeled_batch)
    labeled_filtered = labeled_df[KEY_COLS + [TARGET, "prob_No", "prob_Yes", "uncertainty"]].copy()

    labeled_filtered["ITERATION_ADDED"] = int(iteration)

    labeled_filtered = labeled_filtered.rename(columns={
        "prob_No": "PROB_NO_AT_SELECTION",
        "prob_Yes": "PROB_YES_AT_SELECTION",
        "uncertainty": "UNCERTAINTY_AT_SELECTION"
    })

    train_updated = train_df.merge(labeled_filtered, on=KEY_COLS, how="left", suffixes=("", "_new"))
    train_updated[TARGET] = train_updated[f"{TARGET}_new"].combine_first(train_updated[TARGET])

    for col in METADATA_COLS:
        new_col = f"{col}_new"
        if new_col in train_updated.columns:
            train_updated[col] = train_updated[new_col].combine_first(train_updated[col])
            train_updated = train_updated.drop(columns=[new_col])

    train_updated = train_updated.drop(columns=[f"{TARGET}_new"])
    
    return train_updated

def load_training_data(args):
    train = pd.read_csv(args.train_file, sep="\t", decimal=".")
     #add columns for metadata in iteration 0
    if args.iteration == 0:

        train["ITERATION_ADDED"] = pd.Series(pd.NA, dtype="Int64")
        for col in ["PROB_NO_AT_SELECTION","PROB_YES_AT_SELECTION","UNCERTAINTY_AT_SELECTION"]:
            train[col] = np.nan

        train.loc[train[TARGET].notna(), "ITERATION_ADDED"] = 0
        train.to_csv(args.train_file, sep="\t", index=False)

    if args.iteration > 1:
        train = update_train(train, args.add_to_train_batch, args.iteration)
        train.to_csv(args.train_file, sep="\t", index=False)

    return train

def load_validation_batch(args):
    model_cols = CATEGORICAL_COLS + NUMERIC_COLS
    val = pd.read_excel(args.validation_batch)

    X_val = val[model_cols].copy()
    y_val = val[TARGET].map({"No": 0, "Yes": 1})

    return X_val, y_val

def remove_validation_from_pool(df_unlabeled, args):
    val = pd.read_excel(args.validation_batch)
    val_keys = val[KEY_COLS]
    df_unlabeled = df_unlabeled.merge(val_keys, on=KEY_COLS, how="left", indicator=True)
    df_unlabeled = df_unlabeled[df_unlabeled["_merge"] == "left_only"].drop(columns=["_merge"])
    return df_unlabeled

def split_labeled_unlabeled(train, args):

    df_labeled = train[train[TARGET].notna()].copy().drop(columns=METADATA_COLS)
    df_unlabeled = train[train[TARGET].isna()].copy().drop(columns=METADATA_COLS)

    if args.iteration > 0 and args.validation_batch is not None:
        df_unlabeled = remove_validation_from_pool(df_unlabeled, args)

    return df_labeled, df_unlabeled

def load_test(args):
    test = pd.read_csv(args.test_file, sep="\t", decimal=".")
    X_test = test.drop(columns=DROP_COLS)
    y_test = test[TARGET].map({"No": 0, "Yes": 1})

    return test, X_test, y_test


# building model, preprocess + XGBClassifier
def build_model(neg_pos_ratio):
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])


    preprocess = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_COLS),

            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), [v for v in NUMERIC_COLS if v not in ["HMMER_MP_EVALUE","HMMER_MP_SCORE"]]),

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


#evaluation of the model, best threshold, accuracy, f1, roc_auc, calibration plot, feature importance
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

def evaluate_model(model, X, y, iteration, name):

    probs = model.predict_proba(X)
    pred = model.predict(X)
    uncertainty = np.min(probs, axis=1)

    accuracy = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    roc_auc = roc_auc_score(y, probs[:, 1])

    best_threshold(y, probs, name)
    calibration_plot(y, probs, name, iteration)
    feature_importance(model, X, y, iteration, name)

    return accuracy, f1, roc_auc, pred, probs, uncertainty

def df_predictions(df, pred, probs, uncertainty):

    df = df.copy()
    df["prediction"] = pred
    df["prob_No"] = probs[:, 0]
    df["prob_Yes"] = probs[:, 1]
    df["uncertainty"] = uncertainty

    return df

# ACTIVE LEARNING SAMPLING TECHNIQUES

#v1 -> select candidates closest to uncertainty = 0.5
def select_most_uncertain_samples(model, X_pool, df_unlabeled, batch_size=20):
    probs = model.predict_proba(X_pool)
    uncertainty = np.min(probs, axis=1)
    candidate_indices = np.argsort(uncertainty)[-batch_size:]
    batch = df_unlabeled.iloc[candidate_indices].copy()
    batch["prob_No"] = probs[candidate_indices, 0]
    batch["prob_Yes"] = probs[candidate_indices, 1]
    batch["uncertainty"] = uncertainty[candidate_indices]

    return batch

#v2 -> select 20 candidates with uncertainty > 0.2

def select_batch(model,X_pool, batch_size, df_unlabeled, feature_weights, min_uncertainty = 0.2):
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
        model_cols = CATEGORICAL_COLS + NUMERIC_COLS

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

    batch = to_label.sort_values("uncertainty", ascending=False)

    return batch


#v3 -> select 20 candidates with  uncertainty > 0.1 (v2) and 10 candidates with uncertainty < 0.1
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
        model_cols = CATEGORICAL_COLS + NUMERIC_COLS

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


#v4 -> select most certain candidates and cluster them to get diverse samples

def select_negative_positive_samples(model, X_pool, df_unlabeled, feature_weights, batch_size=10, prob_limit = 0.05, negative=True):

    probs = model.predict_proba(X_pool)
    uncertainty = np.min(probs, axis=1)
    if negative:
        thresholds = [prob_limit, prob_limit + 0.05, prob_limit + 0.1]
        for threshold in thresholds:
            mask = probs [:, 1] < threshold
            if mask.sum() >=batch_size:
                break
    else:
        thresholds = [prob_limit, prob_limit-0.05, prob_limit-0.1]
        for threshold in thresholds:
            mask = probs [:, 1] > threshold
            if mask.sum() >= batch_size:
                break
    candidate_indices = np.where(mask)[0]
    X_candidates = X_pool.iloc[candidate_indices].copy()

    preprocess = model.named_steps["preprocess"]
    X_candidates_transformed = preprocess.transform(X_candidates)

    if hasattr(X_candidates_transformed, "toarray"):
        X_candidates_transformed = X_candidates_transformed.toarray()
        
    if feature_weights is None:
        transformed_weights = np.ones(X_candidates_transformed.shape[1])

    else:
        model_cols = CATEGORICAL_COLS + NUMERIC_COLS

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


#outputs

def save_predictions(args,df_unlabeled, pred, probs, uncertainty, name):
    predictions = df_predictions(df_unlabeled, pred, probs, uncertainty)
    predictions.to_csv(os.path.join("predictions", f"{name}_predictions_iter_{args.iteration}.csv"), index=False)
    uncertainty_distribution(uncertainty, name, args.iteration, output_dir="uncertainty_histograms")
    probability_distribution(probs,name=name,iteration=args.iteration,output_dir="histograms")

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

def print_stats(args, df_labeled, df_unlabeled, test_accuracy, f1, output_file):

    print(f"Iteration: {args.iteration}")
    print(f"Labeled examples: {len(df_labeled)}")
    print(f"Unlabeled examples: {len(df_unlabeled)}")
    print(f"Accuracy on test set: {test_accuracy:.4f}")
    print(f"F1 on test set: {f1:.4f}")
    print(f"Saved next batch to: {output_file}")
    print(f"Saved stats to: {args.stats_file}")


#main function

def main():

    args = parse_args()
    create_output_dirs()
    
    if args.iteration > 0 and args.validation_batch is None:
        print("No validation batch provided. Skipping validation.")

    if args.iteration > 1 and args.add_to_train_batch is None:
        raise ValueError(
            "For iteration > 1, you must provide --add-to-train-batch."
        )
    
    train = load_training_data(args)
    df_labeled, df_unlabeled = split_labeled_unlabeled(train, args)

    X_train = df_labeled.drop(columns=DROP_COLS)
    y_train = df_labeled[TARGET].map({"No": 0, "Yes": 1})

    X_pool = df_unlabeled.drop(columns=DROP_COLS)

    #caluclate ratio
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    ratio = neg / pos
    model = build_model(ratio)
    model.fit(X_train, y_train)

    #validation batch
    feature_weights = None
    roc_auc_val = None
    if args.validation_batch is not None:
        X_val, y_val = load_validation_batch(args)
        feature_weights = permutation_feature_weights(model, X_val, y_val)

        _, _, roc_auc_val, _, _, _ = evaluate_model(model, X_val, y_val, args.iteration, "validation")

    #train evaluation
    _, _, roc_auc_train, train_pred , train_probs, train_uncertainty = evaluate_model(model, X_train, y_train, args.iteration, "train")
    save_predictions(args, df_labeled, train_pred, train_probs, train_uncertainty, "train")

    #pool evaluation

    pool_probs = model.predict_proba(X_pool)
    pool_uncertainty = np.min(pool_probs, axis=1)
    pool_pred = model.predict(X_pool)

    save_predictions(args, df_unlabeled, pool_pred, pool_probs, pool_uncertainty, "pool")

    #batch for annotation

    if sampling_version == 1:
        to_label = select_most_uncertain_samples(model, X_pool, df_unlabeled, batch_size=POSITIVE_BATCH_SIZE)
    elif sampling_version == 2:
        to_label = select_batch(model, X_pool, batch_size=POSITIVE_BATCH_SIZE, df_unlabeled=df_unlabeled, feature_weights=feature_weights, min_uncertainty=0.2)
    elif sampling_version == 3:
        high_certainty_batch = select_high_certainty_batch(model, X_pool, batch_size=NEGATIVE_BATCH_SIZE, df_unlabeled=df_unlabeled, feature_weights=feature_weights, max_uncertainty=0.1)
        uncertrain_batch = select_batch(model, X_pool, batch_size=POSITIVE_BATCH_SIZE, df_unlabeled=df_unlabeled, feature_weights=feature_weights, min_uncertainty=0.2)
        to_label = pd.concat([high_certainty_batch, uncertrain_batch])
    elif sampling_version == 4:
        negative_batch = select_negative_positive_samples(model, X_pool, df_unlabeled, feature_weights, batch_size= NEGATIVE_BATCH_SIZE, prob_limit= NEGATIVE_THRESHOLD, negative=True)
        positive_batch = select_negative_positive_samples(model, X_pool, df_unlabeled, feature_weights, batch_size= POSITIVE_BATCH_SIZE, prob_limit= POSITIVE_THRESHOLD, negative=False)
        to_label = pd.concat([negative_batch, positive_batch])

    else:
        raise ValueError(f"Invalid sampling_version: {sampling_version}. "
                         "Choose 1, 2, 3 or 4.")

    output_file = os.path.join("batches",f"to_label_iteration_{args.iteration}.xlsx")
    to_label.to_excel(output_file, index=False)

    #testing the model
    test, X_test, y_test = load_test(args)
    accuracy, f1, roc_auc_test, test_pred, test_probs, test_uncertainty = evaluate_model(model, X_test, y_test, args.iteration, "test")
    save_predictions(args, test, test_pred, test_probs, test_uncertainty, "test")
    prediction_histogram(y_test, test_pred, args.iteration)
    confusion_matrix_heatmap(y_test, test_pred, args.iteration)

    stats(args.stats_file, len(df_labeled), len(df_unlabeled), args.iteration, accuracy, f1, pool_uncertainty, roc_auc_test, roc_auc_train, roc_auc_val)

    print_stats(args, df_labeled, df_unlabeled, accuracy, f1, output_file)


if __name__ == "__main__":
    main()
