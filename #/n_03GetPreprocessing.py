def get_preprocessing(preprocessing_fn=None):
    _transform = []  # 변환(transform)을 저장할 리스트

    # 만약 데이터 정규화 함수(preprocessing_fn)가 주어진 경우
    if preprocessing_fn:
        # 1. 이미지에 정규화 함수를 적용하는 Lambda 변환 추가
        _transform.append(album.Lambda(image=preprocessing_fn))

    # 2. 이미지와 마스크를 텐서 형태로 변환하는 Lambda 변환 추가
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    # 정의된 변환을 albumentations.Compose로 결합하여 반환
    return album.Compose(_transform)
