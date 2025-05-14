import xgboost as xgb

def train_xgboost(X_attack, y_attack):
    model = xgb.XGBClassifier(objective='multi:softprob', random_state=42)
    model.fit(X_attack, y_attack)
    return model

def classify_attacks(model, X):
    return model.predict(X)