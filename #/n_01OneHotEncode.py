# 원-핫 인코딩 함수: 다채널 형태로 변환
def one_hot_encode(label, label_values):
    semantic_map = []  # 각 클래스에 대한 이진 맵을 저장할 리스트
    for colour in label_values:  # 각 클래스별 색상(또는 값)에 대해 반복
        equality = np.equal(label, colour)  # 현재 클래스 색상과 라벨이 같은지 비교 (True/False)
        class_map = np.all(equality, axis=-1)  # RGB 채널 모두가 일치하는지 확인하여 2D 이진 맵 생성
        semantic_map.append(class_map)  # 이진 맵을 리스트에 추가
    semantic_map = np.stack(semantic_map, axis=-1)  # 클래스별 이진 맵을 쌓아 다채널 배열 생성 (원-핫 인코딩)

    return semantic_map  # 원-핫 인코딩된 다채널 배열 반환


# 리버스 원-핫 인코딩 함수: 다채널에서 한 채널로 변환
def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)  # 각 픽셀에서 가장 큰 값(클래스 인덱스)을 가진 채널의 인덱스를 반환
    return x  # 클래스 인덱스로 이루어진 2D 배열 반환


# 클래스 인덱스를 컬러 이미지로 변환하는 함수
def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)  # 클래스에 해당하는 색상값을 배열로 변환
    x = colour_codes[image.astype(int)]  # 클래스 인덱스를 사용해 색상값을 할당
    return x  # 컬러로 표현된 세그멘테이션 결과 반환
