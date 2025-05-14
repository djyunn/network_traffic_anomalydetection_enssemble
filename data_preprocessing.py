import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from scipy.sparse import hstack

def load_data(train_file, test_file):
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
               'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
               'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 
               'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
               'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
               'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']
    df_train = pd.read_csv(train_file, names=columns)
    df_test = pd.read_csv(test_file, names=columns)
    return df_train, df_test

def preprocess_data(df_train, df_test):
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']
    X_test = df_test.drop('label', axis=1)
    y_test = df_test['label']
    
    categorical_cols = ['protocol_type', 'service', 'flag']
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
    
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X_train[categorical_cols])
    X_train_cat = ohe.transform(X_train[categorical_cols])
    X_test_cat = ohe.transform(X_test[categorical_cols])
    
    scaler = StandardScaler()
    scaler.fit(X_train[numerical_cols])
    X_train_num = scaler.transform(X_train[numerical_cols])
    X_test_num = scaler.transform(X_test[numerical_cols])
    
    X_train_processed = hstack([X_train_num, X_train_cat])
    X_test_processed = hstack([X_test_num, X_test_cat])
    
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    return X_train_processed, y_train_encoded, X_test_processed, y_test_encoded, label_encoder