# augmented_dataset 생성: RoadsDataset 클래스를 사용하여 데이터셋을 구성
augmented_dataset = RoadsDataset(
    x_train_dir,               # 훈련 이미지가 저장된 디렉토리
    y_train_dir,               # 훈련 마스크가 저장된 디렉토리
    augmentation=get_training_augmentation(),  # 학습 데이터에 적용할 데이터 증강 함수
    class_rgb_values=select_class_rgb_values,  # 클래스별 RGB 값 리스트
)

# 랜덤 인덱스를 생성하여 augmented_dataset에서 무작위로 이미지/마스크 쌍 선택
random_idx = random.randint(0, len(augmented_dataset)-1)

# 주어진 이미지/마스크 쌍에 대해 다른 증강을 3번 적용 (256x256 crop 포함)
for i in range(3):  
    # 랜덤 인덱스에 해당하는 이미지와 마스크를 가져옴
    image, mask = augmented_dataset[random_idx]
    
    # 시각화 함수 호출: 원본 이미지, Ground Truth, One-Hot 인코딩된 마스크를 출력
    visualize(
        original_image=image,  # 원본 이미지
        ground_truth_mask=colour_code_segmentation(  # 원본 마스크에 색상을 입힘
            reverse_one_hot(mask),  # One-Hot 마스크를 단일 채널로 변환
            select_class_rgb_values  # 클래스에 해당하는 RGB 색상값
        ),
        one_hot_encoded_mask=reverse_one_hot(mask)  # 마스크를 다시 단일 채널로 변환
    )
