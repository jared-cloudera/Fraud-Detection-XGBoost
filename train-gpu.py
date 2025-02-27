import time
import cupy as cp
import onnxmltools
import pandas as pd
import xgboost as xgb
import numpy as np
from cuml.model_selection import train_test_split
from onnxconverter_common import FloatTensorType
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def main():
    data = pd.read_csv("creditcard.csv")
    X = data.drop("Class", axis=1).to_numpy()
    y = data["Class"].to_numpy()
    X = cp.array(X)
    y = cp.array(y)

    # Create 0.75/0.25 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, train_size=0.75, random_state=42
    )

    # Specify sufficient boosting iterations to reach a minimum
    num_round = 3000

    params = {
        "colsample_bytree": 0.974,
        "learning_rate": 0.017,
        "max_depth": 10,
        "min_child_weight": 7,
        "reg_alpha": 0.206,
        "reg_lambda": 0.377,
        "subsample": 0.577,
    }

    # Initialize classifier with GPU support
    clf = xgb.XGBClassifier(
        **params, device="cuda", n_estimators=num_round, early_stopping_rounds=500
    )

    # Train model with early stopping.
    # The eval_set is used to monitor performance and early stopping rounds is set to 10.
    start = time.time()
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=True,  # Set to True to see early stopping info
    )

    # You can access the best iteration via clf.best_iteration
    print("Best iteration:", clf.best_iteration)
    gpu_res = clf.evals_result()
    print("GPU Training Time: %s seconds" % (str(time.time() - start)))

    # Evaluate on the test set with default threshold of 0.5
    X_test_cp = cp.asarray(X_test)
    y_pred = clf.predict(X_test_cp)

    y_pred_np = cp.asnumpy(y_pred)
    y_test_np = cp.asnumpy(y_test)

    print("\nMetrics with Default Threshold (0.5):")
    print("Accuracy Score:", accuracy_score(y_test_np, y_pred_np))
    print("Classification Report:\n", classification_report(y_test_np, y_pred_np))
    print("Confusion Matrix:\n", confusion_matrix(y_test_np, y_pred_np))

    # Now, empirical thresholding: find optimal threshold based on F1 score
    # Get predicted probabilities for the positive class
    y_proba = clf.predict_proba(X_test_cp)[:, 1]
    y_proba_np = cp.asnumpy(y_proba)

    thresholds = np.linspace(0.0, 1.0, 101)
    best_thresh = 0.5
    best_f1 = -np.inf
    for thresh in thresholds:
        y_pred_thresh = (y_proba_np >= thresh).astype(int)
        f1 = f1_score(y_test_np, y_pred_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    print("\nOptimal Threshold based on F1 score:", best_thresh)
    print("Best F1 score at this threshold:", best_f1)

    # Recalculate metrics with the optimal threshold
    y_pred_opt = (y_proba_np >= best_thresh).astype(int)
    print("\nMetrics with Optimal Threshold:")
    print("Accuracy Score:", accuracy_score(y_test_np, y_pred_opt))
    print("Classification Report:\n", classification_report(y_test_np, y_pred_opt))
    print("Confusion Matrix:\n", confusion_matrix(y_test_np, y_pred_opt))
    clf.save_model("fraud_classifier.ubj")

    num_features = X.shape[1]
    initial_type = [("input", FloatTensorType([None, num_features]))]
    onnx_model = onnxmltools.convert_xgboost(clf, initial_types=initial_type)
    onnxmltools.utils.save_model(onnx_model, "fraud_classifier.onnx")
    print("\nFinal model saved as 'fraud_classifier.onnx'.")


if __name__ == "__main__":
    main()
