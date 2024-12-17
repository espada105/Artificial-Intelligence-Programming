# load best saved model checkpoint from the current run
if os.path.exists('./best_model.pth'):  # 현재 디렉토리에 'best_model.pth' 파일이 존재하는지 확인
    best_model = torch.load('./best_model.pth', map_location=DEVICE)  
    # 모델을 불러오고 현재 DEVICE (GPU/CPU)에 매핑
    print('Loaded UNet model from this run.')  # 모델이 성공적으로 로드되었음을 알림

# load best saved model checkpoint from previous commit (if present)
elif os.path.exists('./input/unet-resnet50-frontend-road-segmentation-pytorch/best_model.pth'):
    # 이전 커밋에 저장된 모델 체크포인트가 존재하는 경우
    best_model = torch.load('./input/unet-resnet50-frontend-road-segmentation-pytorch/best_model.pth', map_location=DEVICE)
    # 모델을 불러오고 현재 DEVICE에 매핑
    print('Loaded UNet model from a previous commit.')  # 이전 체크포인트 모델이 로드되었음을 알림



# create test dataloader to be used with UNet model (with preprocessing operation: to_tensor(...))
test_dataset = RoadsDataset(  # 테스트 데이터셋 생성
    x_test_dir,                      # 테스트 이미지 디렉토리
    y_test_dir,                      # 테스트 라벨(마스크) 디렉토리
    augmentation=get_validation_augmentation(),  # 테스트용 증강 함수 (해상도 맞춤)
    preprocessing=get_preprocessing(preprocessing_fn),  # 데이터 전처리 함수 (to_tensor 등)
    class_rgb_values=select_class_rgb_values,    # 클래스별 RGB 값
)

test_dataloader = DataLoader(test_dataset)  # 테스트 데이터 로더 생성

# test dataset for visualization (without preprocessing transformations)
test_dataset_vis = RoadsDataset(  # 전처리를 적용하지 않은 테스트 데이터셋 생성
    x_test_dir,                      # 테스트 이미지 디렉토리
    y_test_dir,                      # 테스트 라벨 디렉토리
    augmentation=get_validation_augmentation(),  # 테스트용 증강 함수
    class_rgb_values=select_class_rgb_values,    # 클래스별 RGB 값
)
