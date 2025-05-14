import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

def train_xgboost(X_attack, y_attack):
    # y_attack 레이블 재인코딩 (0부터 시작하는 연속적인 정수로)
    label_encoder_attack = LabelEncoder()
    y_attack_encoded = label_encoder_attack.fit_transform(y_attack)
    
    # XGBoost 모델 학습
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(label_encoder_attack.classes_),
        random_state=42
    )
    model.fit(X_attack, y_attack_encoded)
    
    return model, label_encoder_attack

def classify_attacks(model, X, label_encoder_attack):
    y_pred_encoded = model.predict(X)
    y_pred = label_encoder_attack.inverse_transform(y_pred_encoded)
    return y_pred