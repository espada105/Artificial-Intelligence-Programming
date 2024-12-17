# 학습 및 설정 관련 플래그
TRAINING = True  # 모델 학습 여부를 설정. False일 경우 기존 모델 체크포인트를 사용해 예측만 수행

# 학습 반복 수 설정
EPOCHS = 15  # 학습을 반복할 에폭 수를 15로 설정

# 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
# GPU(CUDA)가 사용 가능한 경우 'cuda'를 사용하고, 아니면 'cpu'를 사용

# 손실 함수 설정
loss = smp.losses.DiceLoss('multilabel')  # Dice 손실 함수를 사용, 'multilabel' 옵션은 다중 클래스 세그멘테이션에 사용됨
loss.__name__ = 'Dice_loss'  # 손실 함수 이름을 'Dice_loss'로 설정

import segmentation_models_pytorch.utils.metrics  # 메트릭을 가져오기 위한 라이브러리 임포트

# 메트릭 정의
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),  # IoU (Intersection over Union) 메트릭을 사용하며 임계값은 0.5로 설정
]

# 최적화 함수 설정
optimizer = torch.optim.Adam([  # Adam 최적화 알고리즘 사용
    dict(params=model.parameters(), lr=0.00008),  # 모델의 파라미터와 학습률을 설정 (0.00008)
])

# 학습률 스케줄러 설정
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(  
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,  
    # Cosine Annealing Warm Restarts 스케줄러 사용:
    # - T_0=1: 첫 주기 길이
    # - T_mult=2: 주기 길이를 두 배로 증가
    # - eta_min=5e-5: 최소 학습률
)

# 이전 학습 모델 체크포인트 불러오기
if os.path.exists('./input/unet-resnet50-frontend-road-segmentation-pytorch/best_model.pth'):
    model = torch.load('./input/unet-resnet50-frontend-road-segmentation-pytorch/best_model.pth', map_location=DEVICE)
    # 체크포인트 경로에 저장된 모델을 불러오고, 현재 장치(GPU 또는 CPU)에 로드
