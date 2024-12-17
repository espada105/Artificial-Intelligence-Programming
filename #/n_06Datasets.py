# Get train and val dataset instances
train_dataset = RoadsDataset(  # 학습 데이터셋 객체 생성
    x_train_dir,                             # 학습 이미지가 저장된 디렉토리 경로
    y_train_dir,                             # 학습 라벨(마스크)이 저장된 디렉토리 경로
    augmentation=get_training_augmentation(),  # 학습용 데이터 증강 함수 (랜덤 크롭, 회전 등)
    preprocessing=get_preprocessing(preprocessing_fn),  # 학습용 데이터 전처리 함수
    class_rgb_values=select_class_rgb_values,  # 클래스에 해당하는 RGB 색상값 리스트
)

valid_dataset = RoadsDataset(  # 검증 데이터셋 객체 생성
    x_valid_dir,                              # 검증 이미지가 저장된 디렉토리 경로
    y_valid_dir,                              # 검증 라벨(마스크)이 저장된 디렉토리 경로
    augmentation=get_validation_augmentation(),  # 검증용 데이터 증강 함수 (패딩 등)
    preprocessing=get_preprocessing(preprocessing_fn),  # 검증용 데이터 전처리 함수
    class_rgb_values=select_class_rgb_values,  # 클래스에 해당하는 RGB 색상값 리스트
)
