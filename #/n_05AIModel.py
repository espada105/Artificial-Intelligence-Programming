# 사용할 인코더(Encoder) 모델을 지정합니다.
ENCODER = 'resnet50'  # ResNet-50을 백본(encoder)으로 사용하는 모델

# 인코더의 가중치(weights)를 지정합니다.
ENCODER_WEIGHTS = 'imagenet'  # ImageNet으로 사전 학습된 가중치를 사용

# 세그멘테이션 모델의 클래스 수를 정의합니다.
CLASSES = select_classes  # 사용할 클래스 리스트 (예: 도로, 차선 등)

# 모델의 활성화 함수(activation function)를 정의합니다.
ACTIVATION = 'sigmoid'  # 이진 세그멘테이션의 경우 'sigmoid', 다중 클래스는 'softmax2d' 사용
# NOTE: None으로 설정하면 로짓(logits)을 반환합니다.

# 사전 학습된 인코더를 사용하여 세그멘테이션 모델 생성
model = smp.Unet(  
    encoder_name=ENCODER,                # 사용할 인코더 모델 이름 ('resnet50')
    encoder_weights=ENCODER_WEIGHTS,     # 사전 학습된 가중치 (ImageNet 가중치 사용)
    classes=len(CLASSES),                # 세그멘테이션 모델의 출력 클래스 수
    activation=ACTIVATION,               # 출력층 활성화 함수 (sigmoid or softmax2d)
)
