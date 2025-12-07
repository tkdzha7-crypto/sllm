# 1부: 시스템 환경 구축

**목차** | [다음: 2부 모델 학습 →](02-model-training.md)

---

## 개요

이 섹션에서는 CESCO sLLM 추론 서버 실행에 필요한 시스템 환경을 구축합니다.

---

## Step 1.1: 시스템 요구사항 확인

### 목표
시스템이 sLLM 추론 서버 실행에 필요한 최소 요구사항을 충족하는지 확인합니다.

### 하드웨어 요구사항

**GPU (필수)**
```bash
# GPU 정보 확인
lspci | grep -i nvidia
```

**요구사항:**
- NVIDIA GPU (RTX 3060 이상 권장)
- VRAM 8GB 이상 (추론용)
- VRAM 12GB 이상 (학습용)

**메모리 확인**
```bash
# 시스템 메모리 확인
free -h
```

**요구사항:**
- 추론: 16GB 이상
- 학습: 32GB 이상

**디스크 공간 확인**
```bash
# 디스크 공간 확인
df -h
```

**요구사항:**
- 학습 및 모델 저장: 100GB 이상
- 추론만: 20GB 이상

**운영체제 확인**
```bash
# Ubuntu 버전 확인
lsb_release -a
```

**요구사항:**
- Ubuntu 20.04 이상 (22.04 권장)

### 확인 방법
위 명령어들을 실행하여 각 요구사항을 충족하는지 확인합니다.

---

## Step 1.2: NVIDIA 드라이버 설치

### 목표
GPU를 사용하기 위한 NVIDIA 드라이버를 설치합니다.

### 현재 드라이버 확인
```bash
nvidia-smi
```

**드라이버가 없는 경우:**
`nvidia-smi: command not found` 또는 에러 메시지 출력

### 권장 드라이버 확인
```bash
# 권장 드라이버 목록 확인
ubuntu-drivers devices
```

### 드라이버 설치
```bash
# 권장 드라이버 자동 설치
sudo ubuntu-drivers autoinstall

# 또는 특정 버전 설치 (예: 535)
sudo apt update
sudo apt install -y nvidia-driver-535

# 시스템 재부팅
sudo reboot
```

### 설치 확인
```bash
nvidia-smi
```

**예상 출력:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   35C    P0    25W / 250W |      0MiB /  8192MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

---

## Step 1.3: CUDA Toolkit 설치

### 목표
GPU 프로그래밍을 위한 CUDA Toolkit을 설치합니다.

### CUDA 12.x 다운로드 및 설치

**Ubuntu 22.04 기준:**
```bash
# CUDA 키링 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 패키지 목록 업데이트
sudo apt update

# CUDA Toolkit 12.8 설치
sudo apt install -y cuda-toolkit-12-8
```

**Ubuntu 20.04 기준:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

### 환경 변수 설정
```bash
# .bashrc 파일 편집
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# 변경사항 적용
source ~/.bashrc
```

### 설치 확인
```bash
nvcc --version
```

**예상 출력:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Oct_30_...
Cuda compilation tools, release 12.8, V12.8.xxx
```

---

## Step 1.4: Docker 설치

### 목표
컨테이너 기반 배포를 위한 Docker를 설치합니다.

### 기존 Docker 제거 (있는 경우)
```bash
sudo apt remove docker docker-engine docker.io containerd runc
```

### Docker 공식 리포지토리 추가
```bash
# 필수 패키지 설치
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release

# Docker GPG 키 추가
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Docker 리포지토리 추가
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### Docker 설치
```bash
# 패키지 목록 업데이트
sudo apt update

# Docker 설치
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Docker 서비스 시작
sudo systemctl start docker
sudo systemctl enable docker
```

### 사용자를 docker 그룹에 추가
```bash
# 현재 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER

# 그룹 변경사항 적용 (로그아웃 후 재로그인하거나)
newgrp docker
```

### 설치 확인
```bash
docker --version
docker run hello-world
```

**예상 출력:**
```
Docker version 24.x.x, build ...

Hello from Docker!
This message shows that your installation appears to be working correctly.
...
```

---

## Step 1.5: NVIDIA Container Toolkit 설치

### 목표
Docker 컨테이너에서 GPU를 사용할 수 있도록 NVIDIA Container Toolkit을 설치합니다.

### 리포지토리 설정
```bash
# GPG 키 추가
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 리포지토리 추가
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### NVIDIA Container Toolkit 설치
```bash
# 패키지 목록 업데이트
sudo apt update

# 설치
sudo apt install -y nvidia-container-toolkit

# Docker 데몬 재시작
sudo systemctl restart docker
```

### GPU 컨테이너 테스트
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

**예상 출력:**
nvidia-smi 출력이 정상적으로 표시되어야 합니다.

---

## Step 1.6: Python 환경 설정

### 목표
Python 3.10 환경을 설정하고 필요한 패키지 관리자를 설치합니다.

### Python 3.10 설치
```bash
# Ubuntu 22.04는 기본적으로 Python 3.10 포함
python3 --version

# Ubuntu 20.04인 경우
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

### pip 설치 및 업그레이드
```bash
sudo apt install -y python3-pip
python3.10 -m pip install --upgrade pip
```

### uv 패키지 매니저 설치 (선택사항, 권장)
```bash
# uv는 pip보다 10-100배 빠른 패키지 매니저입니다
curl -LsSf https://astral.sh/uv/install.sh | sh

# 환경 변수 적용
source $HOME/.cargo/env
```

### 설치 확인
```bash
python3.10 --version
pip --version
uv --version  # uv 설치한 경우
```

**예상 출력:**
```
Python 3.10.x
pip 24.x from ...
uv 0.x.x
```

---

**[목차로 돌아가기](README.md)** | **[다음: 2부 모델 학습 →](02-model-training.md)**
