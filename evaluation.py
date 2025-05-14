from sklearn.metrics import classification_report, f1_score
import numpy as np

def evaluate_anomaly_detection(y_true_binary, y_pred_anomaly):
    print("이상 탐지 성능:")
    print(classification_report(y_true_binary, y_pred_anomaly))
    f1 = f1_score(y_true_binary, y_pred_anomaly)
    return f1

def evaluate_attack_classification(y_true_attack, y_pred_attack, label_encoder):
    print("공격 유형 분류 성능:")
    # y_true_attack과 y_pred_attack의 고유 레이블 수집
    unique_labels = np.unique(np.concatenate([y_true_attack, y_pred_attack]))
    # label_encoder.classes_에서 해당 레이블 이름 추출
    target_names = [label_encoder.classes_[i] for i in unique_labels]
    print(classification_report(y_true_attack, y_pred_attack, target_names=target_names))
    f1 = f1_score(y_true_attack, y_pred_attack, average='weighted')
    return f1