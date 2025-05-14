from sklearn.metrics import classification_report, f1_score

def evaluate_anomaly_detection(y_true_binary, y_pred_anomaly):
    print("이상 탐지 성능:")
    print(classification_report(y_true_binary, y_pred_anomaly))
    f1 = f1_score(y_true_binary, y_pred_anomaly)
    return f1

def evaluate_attack_classification(y_true_attack, y_pred_attack, label_encoder):
    print("공격 유형 분류 성능:")
    print(classification_report(y_true_attack, y_pred_attack, target_names=label_encoder.classes_))
    f1 = f1_score(y_true_attack, y_pred_attack, average='weighted')
    return f1