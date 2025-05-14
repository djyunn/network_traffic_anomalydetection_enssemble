from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
import numpy as np

def train_ensemble_models(X_normal):
    models = {
        'IsolationForest': IsolationForest(contamination=0.1, random_state=42),
        'OneClassSVM': OneClassSVM(nu=0.1),
        'GMM': GaussianMixture(n_components=1, covariance_type='full')
    }
    for name, model in models.items():
        if name == 'GMM':
            model.fit(X_normal)
        else:
            model.fit(X_normal)
    return models

def predict_anomalies_ensemble(models, X):
    predictions = []
    for name, model in models.items():
        if name == 'GMM':
            score = model.score_samples(X)
            threshold = np.percentile(score, 10)
            pred = np.where(score < threshold, -1, 1)
        else:
            pred = model.predict(X)
        predictions.append(pred)
    ensemble_pred = np.mean(predictions, axis=0)
    return np.where(ensemble_pred < 0, 1, 0)