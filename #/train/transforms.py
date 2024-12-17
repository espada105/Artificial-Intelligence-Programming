import torch  # PyTorch 라이브러리
import albumentations as A  # 이미지 데이터 증강을 위한 Albumentations 라이브러리
import numpy as np  # 수치 연산을 위한 NumPy 라이브러리
from albumentations.pytorch import ToTensorV2  # Albumentations에서 Tensor 변환 기능 제공

# 이미지 데이터 증강을 위한 클래스 정의
class ImageAug:
    def __init__(self, train):  # 클래스 초기화 시 train 여부를 인자로 받음
        if train:  
            # 학습용 데이터 증강 설정
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),  # 50% 확률로 수평 뒤집기
                A.VerticalFlip(p=0.5),  # 50% 확률로 수직 뒤집기
                # A.ShiftScaleRotate(p=0.5),  # 주석 처리된 코드: 이동, 확대/축소, 회전
                A.RandomCrop(512, 512),  # 512x512 크기로 랜덤 크롭
                ToTensorV2()  # Tensor로 변환
            ])
        else:
            # 테스트용 데이터 증강 설정 (학습용과 동일하지만 코드가 주석 처리됨)
            self.aug = A.Compose([
                # A.HorizontalFlip(p=0.5),  
                # A.VerticalFlip(p=0.5),
                # A.ShiftScaleRotate(p=0.5),
                A.RandomCrop(512, 512),  # 512x512 크기로 랜덤 크롭
                ToTensorV2()  # Tensor로 변환
            ])

    def __call__(self, img, mask_img):  # 객체를 함수처럼 호출할 때 실행되는 메서드
        transformed = self.aug(image=img, mask=mask_img)  # 이미지와 마스크에 증강 적용
        return transformed['image'], transformed['mask']  # 증강된 이미지와 마스크 반환

    def remap_idxes(self, mask):  # 마스크  인덱스를 재매핑하는 함수
        # 특정 조건을 만족하는 마스크 값을 수정
        mask = torch.where(mask >= 1000, mask.div(1000, rounding_mode='floor'), mask)
        for void_idx in self.void_idxes:  # 무효 인덱스 처리
            mask[mask == void_idx] = self.ignore_idx
        for valid_idx in self.valid_idxes:  # 유효 인덱스를 클래스 맵에 매핑
            mask[mask == valid_idx] = self.class_map[valid_idx]
        return mask  # 수정된 마스크 반환

# 데이터 증강 파이프라인을 생성하는 함수
def get_transforms(train):
    transforms = ImageAug(train)  # ImageAug 클래스 인스턴스를 생성
    return transforms  # 데이터 증강 파이프라인 반환
