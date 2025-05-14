import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

def train_xgboost(X_attack, y_attack):
    # y_attack을 0부터 시작하는 연속적인 레이블로 변환
    label_encoder = LabelEncoder()
    y_attack_encoded = label_encoder.fit_transform(y_attack)
    print("XGBoost 학습용 인코딩된 클래스:", label_encoder.classes_)
    print("인코딩된 y_attack 고유값:", sorted(set(y_attack_encoded)))

    # XGBoost 모델 학습
    model = xgb.XGBClassifier(objective='multi:softprob', random_state=42)
    model.fit(X_attack, y_attack_encoded)
    
    return model, label_encoder

def classify_attacks(model, X, label_encoder):
    # 예측
    y_pred_encoded = model.predict(X)
    # 원래 레이블로 복원
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    return y_pred