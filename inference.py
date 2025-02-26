from xgboost import XGBClassifier
import numpy as np
import pandas as pd

def thresholded_predict(X, estimator, threshold):
    return np.array([1 if (p >= threshold) else 0 for p in estimator.predict_proba(X)[:,1]])

def main():
    params = {
        "colsample_bytree": 0.7638,
        "learning_rate": 0.0518,
        "max_depth": 8,
        "min_child_weight": 3,
        "reg_alpha": 0.2349,
        "reg_lambda": 0.995,
        "subsample": 0.7053
    }
    xgboost_clf = XGBClassifier(**params, n_estimators=1500)
    xgboost_clf.load_model("fraud_detection_xgboost.model")

    # Optimal classification threshold = 0.000
    # TODO seems wrong?
    classification_cutoff = 0.0

    data = pd.read_csv('creditcard.csv')
    data = data.drop('Class', axis = 1)
    X_test = data.sample(10000)

    y_pred = thresholded_predict(X_test, xgboost_clf, threshold = classification_cutoff)
    print(y_pred)

if __name__ == "__main__":
    main()