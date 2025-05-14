import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from scipy.sparse import hstack

def load_data(train_file, test_file):
    # 헤더 없이 원시 데이터 로드
    df_train_raw = pd.read_csv(train_file, delimiter=',', header=None, skipinitialspace=True)
    df_test_raw = pd.read_csv(test_file, delimiter=',', header=None, skipinitialspace=True)
    
    # 열 수와 첫 5행 출력
    print("훈련 데이터 열 수:", df_train_raw.shape[1])
    print("훈련 데이터 첫 5행:\n", df_train_raw.head())
    print("테스트 데이터 열 수:", df_test_raw.shape[1])
    print("테스트 데이터 첫 5행:\n", df_test_raw.head())
    
    # 각 행의 열 수 확인
    train_row_lengths = df_train_raw.apply(lambda x: len(x), axis=1)
    print("훈련 데이터 행별 열 수 불일치:", train_row_lengths[train_row_lengths != 42].index.tolist())
    
    # 열 이름 정의 (42개 열 기준)
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 
               'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
               'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
               'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
               'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
               'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
               'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
               'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
               'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
               'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']
    
    # 열 수가 43개일 경우 마지막 열 제거
    if df_train_raw.shape[1] == 43:
        print("경고: 열 수가 43개입니다. 마지막 열을 제거합니다.")
        df_train_raw = df_train_raw.iloc[:, :42]
        df_test_raw = df_test_raw.iloc[:, :42]
    
    # 열 이름 지정
    df_train = df_train_raw.set_axis(columns, axis=1)
    df_test = df_test_raw.set_axis(columns, axis=1)
    
    # duration, protocol_type, label 고유값 확인
    print("훈련 데이터 duration 고유값:", df_train['duration'].unique())
    print("훈련 데이터 protocol_type 고유값:", df_train['protocol_type'].unique())
    print("훈련 데이터 label 고유값:", df_train['label'].unique())
    print("테스트 데이터 label 고유값:", df_test['label'].unique())
    
    return df_train, df_test

def preprocess_data(df_train, df_test):
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']
    X_test = df_test.drop('label', axis=1)
    y_test = df_test['label']
    
    categorical_cols = ['protocol_type', 'service', 'flag']
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
    
    # 숫자 열 검증
    for col in numerical_cols:
        non_numeric_train = X_train[col][pd.to_numeric(X_train[col], errors='coerce').isna()]
        non_numeric_test = X_test[col][pd.to_numeric(X_test[col], errors='coerce').isna()]
        if not non_numeric_train.empty or not non_numeric_test.empty:
            print(f"{col} 열에 숫자가 아닌 값 존재:")
            print("훈련 데이터:", non_numeric_train.unique())
            print("테스트 데이터:", non_numeric_test.unique())
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
    
    # 범주형 열 인코딩
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X_train[categorical_cols])
    X_train_cat = ohe.transform(X_train[categorical_cols])
    X_test_cat = ohe.transform(X_test[categorical_cols])
    
    # 숫자 열 스케일링
    scaler = StandardScaler()
    scaler.fit(X_train[numerical_cols])
    X_train_num = scaler.transform(X_train[numerical_cols])
    X_test_num = scaler.transform(X_test[numerical_cols])
    
    # 숫자 + 범주형 결합 (희소 행렬)
    X_train_processed = hstack([X_train_num, X_train_cat])
    X_test_processed = hstack([X_test_num, X_test_cat])
    
    # 희소 행렬을 NumPy 배열로 변환
    X_train_processed = X_train_processed.toarray()
    X_test_processed = X_test_processed.toarray()
    
    # 레이블 인코딩
    label_encoder = LabelEncoder()
    all_labels = pd.concat([y_train, y_test]).unique()  # 모든 레이블 통합
    label_encoder.fit(all_labels)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print("인코딩된 레이블 클래스:", label_encoder.classes_)
    
    return X_train_processed, y_train_encoded, X_test_processed, y_test_encoded, label_encoder