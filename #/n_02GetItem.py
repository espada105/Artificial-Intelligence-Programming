# __getitem__: Dataset 클래스에서 인덱스를 통해 이미지와 마스크를 읽고 처리하는 함수
def __getitem__(self, i):
    # read images and masks
    # 이미지 파일을 읽어서 BGR 형식에서 RGB 형식으로 변환 (OpenCV는 BGR 형식으로 이미지를 읽음)
    image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
    
    # 마스크 파일을 읽어서 BGR 형식에서 RGB 형식으로 변환
    mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
    
    # one-hot-encode the mask
    # 원-핫 인코딩을 통해 마스크를 클래스 수에 맞는 다채널 형태로 변환
    mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
    
    # apply augmentations
    # 데이터 증강(augmentation)을 적용하는 부분: 설정된 augmentation이 있을 경우 수행
    if self.augmentation:
        # augmentation 함수를 적용하고, 반환된 결과에서 'image'와 'mask'를 추출
        sample = self.augmentation(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']  # 증강된 이미지와 마스크를 업데이트
    
    # apply preprocessing
    # 사전처리(preprocessing)를 적용하는 부분: 설정된 preprocessing이 있을 경우 수행
    if self.preprocessing:
        # preprocessing 함수를 적용하고, 반환된 결과에서 'image'와 'mask'를 추출
        sample = self.preprocessing(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']  # 사전처리된 이미지와 마스크를 업데이트
    
    # 최종적으로 처리된 이미지와 마스크를 반환
    return image, mask

# __len__: Dataset의 길이(데이터셋에 있는 이미지 수)를 반환하는 함수
def __len__(self):
    # self.image_paths 리스트의 길이(전체 이미지 파일의 수)를 반환
    return len(self.image_paths)
