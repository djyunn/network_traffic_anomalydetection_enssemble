from data_preprocessing import load_data, preprocess_data
from ensemble_anomaly_detection import train_ensemble_models, predict_anomalies_ensemble
from xgboost_classifier import train_xgboost, classify_attacks
from evaluation import evaluate_anomaly_detection, evaluate_attack_classification
import numpy as np

df_train, df_test = load_data('KDDTrain+.txt', 'KDDTest+.txt')
X_train, y_train, X_test, y_test, label_encoder = preprocess_data(df_train, df_test)

normal_label = list(label_encoder.classes_).index('normal')

# 희소 행렬 인덱싱
normal_indices = np.where(y_train == normal_label)[0]
attack_indices = np.where(y_train != normal_label)[0]

X_normal_train = X_train[normal_indices, :]
X_attack_train = X_train[attack_indices, :]
y_attack_train = y_train[attack_indices]

ensemble_models = train_ensemble_models(X_normal_train)
y_pred_anomaly = predict_anomalies_ensemble(ensemble_models, X_test)

indices_anomalies = [i for i in range(X_test.shape[0]) if y_pred_anomaly[i] == 1]
X_test_anomalies = X_test[indices_anomalies, :]

xgb_model, label_encoder_attack = train_xgboost(X_attack_train, y_attack_train)
y_pred_attack = classify_attacks(xgb_model, X_test_anomalies, label_encoder_attack)

y_test_binary = [0 if y == normal_label else 1 for y in y_test]
f1_anomaly = evaluate_anomaly_detection(y_test_binary, y_pred_anomaly)
print(f"이상 탐지 F1-Score: {f1_anomaly}")

true_attack_indices = [i for i in indices_anomalies if y_test[i] != normal_label]
y_true_attack = y_test[true_attack_indices]
y_pred_attack_selected = [y_pred_attack[indices_anomalies.index(i)] for i in true_attack_indices]
f1_attack = evaluate_attack_classification(y_true_attack, y_pred_attack_selected, label_encoder)
print(f"공격 유형 분류 F1-Score: {f1_attack}")