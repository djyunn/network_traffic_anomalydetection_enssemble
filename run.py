from data_preprocessing import load_data, preprocess_data
from autoencoder import train_autoencoder, predict_anomalies
from xgboost_classifier import train_xgboost, classify_attacks
from evaluation import evaluate_anomaly_detection, evaluate_attack_classification
import numpy as np

# 데이터 로드 및 전처리
df_train, df_test = load_data('KDDTrain+.txt', 'KDDTest+.txt')
X_train, y_train, X_test, y_test, label_encoder = preprocess_data(df_train, df_test)

# 정상 데이터와 공격 데이터 분리
normal_label = list(label_encoder.classes_).index('normal')
X_normal_train = X_train[y_train == normal_label, :]
X_attack_train = X_train[y_train != normal_label, :]
y_attack_train = y_train[y_train != normal_label]

# 오토인코더 학습 및 이상 탐지
autoencoder = train_autoencoder(X_normal_train)
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.square(X_test - reconstructions), axis=1)
threshold = np.percentile(mse, 95)
y_pred_anomaly = np.where(mse > threshold, 1, 0)

# 이상 데이터 추출
indices_anomalies = [i for i in range(X_test.shape[0]) if y_pred_anomaly[i] == 1]
X_test_anomalies = X_test[indices_anomalies, :]

# XGBoost로 공격 유형 분류
xgb_model, label_encoder_attack = train_xgboost(X_attack_train, y_attack_train)
y_pred_attack = classify_attacks(xgb_model, X_test_anomalies, label_encoder_attack)

# 이상 탐지 평가
y_test_binary = [0 if y == normal_label else 1 for y in y_test]
f1_anomaly = evaluate_anomaly_detection(y_test_binary, y_pred_anomaly)
print(f"이상 탐지 F1-Score: {f1_anomaly}")

# 공격 유형 분류 평가
true_attack_indices = [i for i in indices_anomalies if y_test[i] != normal_label]
y_true_attack = y_test[true_attack_indices]
y_pred_attack_selected = [y_pred_attack[indices_anomalies.index(i)] for i in true_attack_indices]
f1_attack = evaluate_attack_classification(y_true_attack, y_pred_attack_selected, label_encoder)
print(f"공격 유형 분류 F1-Score: {f1_attack}")