# utils.general 모듈에서 다양한 유틸리티 함수 및 클래스 불러오기
from utils.general import (yaml_load, yaml_save, print_args, LOGGER, colorstr, 
                            one_cycle, increment_path, check_img_size, check_yaml,
                            methods, check_suffix, init_seeds, intersect_dicts, check_dataset, 
                            strip_optimizer, get_latest_run, AverageMeter, attempt_download, 
                            labels_to_class_weights, TQDM_BAR_FORMAT)

# YOLO 모델 정의를 불러오기
from models.yolo import Model  

# 모델 로딩을 위한 실험적 모듈 불러오기
from models.experimental import attempt_load  

# 성능 평가를 위한 메트릭 함수 불러오기
from utils.metrics import fitness  

# 분산 학습과 관련된 유틸리티 함수들 불러오기
from utils.torch_utils import (select_device, de_parallel, is_main_process, EarlyStopping, 
                                torch_distributed_zero_first, smart_optimizer, ModelEMA, 
                                smart_resume, smart_DDP)  

# Anchor 관련 체크 함수 불러오기
from utils.autoanchor import check_anchors  

# 학습 손실 계산을 위한 함수
from utils.loss import ComputeLoss  

# 콜백 함수 관리 모듈
from utils.callbacks import Callbacks  

# 객체 깊은 복사를 위해 copy 모듈 사용
from copy import deepcopy  

# 날짜 및 시간을 다루기 위한 모듈
from datetime import datetime  

# Validation 함수 호출 시 별칭 설정 (mAP 계산)
import val as validate  # for end-of-epoch mAP
from pathlib import Path  # 파일 경로를 처리하기 위한 Path 클래스




# 현재 실행 중인 파일의 절대 경로를 가져옴
FILE = Path(__file__).resolve()

# 상위 경로 설정 (ROOT 디렉토리)
ROOT = FILE.parents[0]  

# ROOT 경로가 sys.path에 없으면 추가
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  

# 현재 경로를 기준으로 상대 경로를 설정
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative  

# 분산 학습 관련 환경 변수 설정
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # 로컬 랭크 (기본값 -1)
RANK = int(os.getenv('RANK', -1))  # 전체 프로세


##############################################
def train(hyp, opt, device, callbacks):
    """
    학습을 수행하는 함수.
    Args:
        hyp (dict): 하이퍼파라미터 설정.
        opt (object): 옵션 객체, 실행 설정을 포함.
        device (str): 학습 실행 장치 ('cpu' 또는 'cuda').
        callbacks (object): 콜백 함수 모음.
    """

# 학습 결과를 저장할 디렉토리 설정
w = save_dir / 'weights'  # 'weights' 하위 디렉토리 경로
w.mkdir(parents=True, exist_ok=True)  # 부모 디렉토리까지 포함하여 생성 (이미 존재해도 에러 없음)

# 체크포인트 파일 경로 설정
last, best = w / 'last.pt', w / 'best.pt'  # 'last.pt'와 'best.pt' 파일 경로

# 하이퍼파라미터가 문자열인 경우 YAML 파일을 로드
if isinstance(hyp, str):
    hyp = yaml_load(hyp)

# 주 프로세스에서만 하이퍼파라미터를 로깅
if is_main_process():
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

# 하이퍼파라미터를 옵션 객체에 저장 (체크포인트 용도)
opt.hyp = hyp.copy()

# 실행 설정과 하이퍼파라미터를 YAML 파일로 저장
yaml_save(save_dir / 'hyp.yaml', hyp)  # 하이퍼파라미터 저장
yaml_save(save_dir / 'opt.yaml', vars(opt))  # 실행 옵션 저장

if is_main_process():
    # 로거 초기화
    loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
    
    # Resume 상태인 경우 학습 설정 복구
    if loggers.wandb:
        if resume:
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # 콜백 함수에 로깅 메서드 등록
    for k in methods(loggers):
        callbacks.register_action(k, callback=getattr(loggers, k))

# CUDA 장치 사용 여부 설정
cuda = device.type != 'cpu'

# 랜덤 시드 초기화
init_seeds(opt.seed + 1 + RANK)

# 분산 학습에서 순차적 로딩 보장
with torch_distributed_zero_first(LOCAL_RANK):
    data_dict = check_dataset(data)  # 데이터셋 설정 검증
    train_path, val_path = data_dict['train'], data_dict['val']  # 학습 및 검증 데이터 경로
    nc = int(data_dict['nc'])  # 클래스 수
    names = data_dict['names']  # 클래스 이름
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO 데이터셋 여부

# 모델 체크포인트 파일 확장자 확인
check_suffix(weights, '.pt')
pretrained = weights.endswith('.pt')  # 사전 학습 모델 여부 확인

# 사전 학습 모델 로드
if pretrained:
    with torch_distributed_zero_first(LOCAL_RANK):
        weights = attempt_download(weights)  # 모델 다운로드
        ckpt = torch.load(weights, map_location='cpu')  # 체크포인트 로드
        model = Model(ckpt['model'].yaml, ch=3, nc=80, anchors=hyp.get('anchors')).to(device)  # 모델 생성
        exclude = ['anchor'] if hyp.get('anchors') and not resume else []  # 제외할 키 설정
        csd = ckpt['model'].float().state_dict()  # 체크포인트의 state_dict
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # 모델과 체크포인트 state_dict 병합
        model.load_state_dict(csd, strict=False)  # 모델에 로드
        LOGGER.info(f'Transfered {len(csd)}/{len(model.state_dict())} items from {weights}')  
        del csd  # 메모리 해제
else:
    # 새 모델 생성
    model = Model('', ch=3, nc=80, anchors=hyp.get('anchors')).to(device)

# 특정 레이어를 Freeze 설정
freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]

# 모델의 파라미터 이름과 값을 순회하며 Freeze 설정 적용
for k, v in model.named_parameters():
    v.requires_grad = True  # 기본적으로 모든 파라미터 학습 가능
    if any(x in k for x in freeze):  # 특정 조건에 맞으면 Freeze
        LOGGER.info(f'freezing {k}')  # Freeze된 파라미터 로그 출력
        v.requires_grad = False

# Image size
gs = max(int(model.stride.max()), 32)  
# 모델의 최대 stride 값을 가져와 grid size(gs)로 설정, 최소값은 32로 보장

imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  
# 입력 이미지 크기(imgsz)가 grid size(gs)의 배수인지 확인 및 조정
# floor=gs * 2: 이미지 크기의 최소값 설정 (grid size의 두 배)

# Optimizer
nbs = 64  # nominal batch size (기준 배치 크기)

# accumulate: 손실을 최적화하기 전까지 누적시킬 횟수 설정
accumulate = max(round(nbs / batch_size), 1)  

# weight_decay를 배치 크기와 누적 횟수를 기준으로 스케일링
hyp['weight_decay'] *= batch_size * accumulate / nbs  

# 주 프로세스에서만 스케일링된 weight decay 값을 로그에 출력
if is_main_process():
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

# 스마트 옵티마이저 설정 (최적의 옵티마이저 반환)
optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
