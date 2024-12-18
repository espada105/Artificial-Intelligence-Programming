import os  # 운영 체제 관련 기능을 제공하는 모듈
import argparse  # 명령줄 인자 파싱을 위한 모듈
import yaml  # YAML 파일을 읽고 쓰기 위한 모듈
import math  # 수학적 계산을 위한 모듈
import torch  # PyTorch 라이브러리, 딥러닝 프레임워크
import sys  # 시스템 관련 기능을 제공하는 모듈
import time  # 시간 관련 기능을 다루는 모듈

from tqdm import tqdm  # 진행 상황(progress bar)을 시각적으로 표시하는 라이브러리
import numpy as np  # 수치 연산 및 배열 처리를 위한 라이브러리

# 데이터 전처리를 위한 사용자 정의 함수 get_transforms를 불러옴
from transforms import get_transforms  

# torchvision: PyTorch에서 제공하는 컴퓨터 비전 관련 라이브러리
from torchvision import models, datasets, transforms  

import torch.nn as nn  # 신경망 모듈 관련 기능 제공
import torch.distributed as dist  # 다중 GPU를 사용한 분산 학습 관련 기능 제공

# 데이터 로딩과 배치 처리를 위한 클래스 및 함수
from torch.utils.data import DataLoader, Dataset, dataloader, distributed  

from torch.optim import lr_scheduler  # 학습률 스케줄링 기능 제공
from torch.cuda import amp  # PyTorch 자동 혼합 정밀도(Amp) 기능 지원
from pathlib import Path  # 파일 및 디렉토리 경로를 객체로 관리하는 모듈

import torch.backends.cudnn as cudnn  # NVIDIA GPU 최적화를 위해 cudnn 백엔드 사용

# 데이터 전처리 기능을 정의하는 사용자 지정 함수 get_transforms 호출
from transforms import get_transforms  

# 데이터 로더를 생성하는 사용자 정의 함수 create_dataloader 호출
from utils.dataloaders import create_dataloader  

# 학습 기록 및 로깅 기능을 위한 사용자 정의 클래스 Loggers 호출
from utils.loggers import Loggers  
