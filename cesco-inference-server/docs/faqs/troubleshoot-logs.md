# 6부: 문제 해결

**[← 이전: 5부 프로덕션 강화](05-production-hardening.md)** | **[목차로 돌아가기](README.md)**

---

## 개요

이 섹션에서는 자주 발생하는 문제들과 해결 방법을 다룹니다.

---

## Step 6.1: 모델 로딩 실패

### 증상
서버 시작 시 모델을 로드하지 못하고 에러가 발생합니다.

### 체크리스트

**1. MODEL_PATH 환경 변수 확인:**
```bash
echo $MODEL_PATH
# 또는
grep MODEL_PATH .env
```

**2. 모델 파일 존재 확인:**
```bash
ls -la $MODEL_PATH
# 또는
ls -la ./outputs_final/best_model
```

**필수 파일 목록:**
- adapter_config.json (LoRA 사용 시)
- adapter_model.safetensors 또는 adapter_model.bin (LoRA 사용 시)
- config.json
- tokenizer.json
- tokenizer_config.json

**3. 파일 권한 확인:**
```bash
ls -l $MODEL_PATH/*
```

모든 파일이 읽기 가능해야 합니다.

**4. 디렉토리 경로 확인:**
```bash
# 절대 경로 확인
readlink -f $MODEL_PATH
```

### 일반적인 오류 및 해결

**오류 1: "Model path does not exist"**
```bash
# 해결: 올바른 경로로 수정
export MODEL_PATH=/home/user/cesco-inference-server/outputs_final/best_model
# 또는 .env 파일 수정
```

**오류 2: "adapter_config.json not found"**
```bash
# 해결: merged 모델 사용
# final_model 대신 final_model_merged 디렉토리 사용
cp -r ./training_output/[timestamp]/final_model_merged/* ./outputs_final/best_model/
```

**오류 3: "CUDA out of memory during model loading"**
```bash
# 해결: 다른 GPU 프로세스 종료
nvidia-smi
# PID 확인 후
kill [PID]

# 또는 4-bit quantization 확인 (app.py에 이미 적용됨)
```

### 로그 확인

```bash
# 상세 로그 확인
tail -50 ~/cesco-inference-server/logs/service.log | grep -i error
```

---

## Step 6.2: CUDA Out of Memory

### 증상
추론 중 "CUDA out of memory" 에러가 발생합니다.

### GPU 메모리 확인

```bash
nvidia-smi
```

**메모리 사용량 확인:**
```
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   45C    P2   120W / 250W |   7856MiB /  8192MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

### 해결 방법

**1. 다른 GPU 프로세스 종료:**
```bash
# GPU 사용 중인 프로세스 확인
nvidia-smi

# 프로세스 종료
kill -9 [PID]
```

**2. max_new_tokens 줄이기:**
```bash
# 요청 시 max_new_tokens를 512에서 256으로 줄임
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input_text": "테스트", "max_new_tokens": 256}'
```

**3. 배치 크기 줄이기:**
```bash
# 한 번에 많은 요청을 보내지 않음
# 배치 요청 시 항목 수를 줄임 (예: 10개 -> 5개)
```

**4. GPU 캐시 정리:**
```python
# Python 인터프리터에서
import torch
torch.cuda.empty_cache()
```

**5. 다른 GPU 사용 (멀티 GPU 환경):**
```bash
# .env 파일에서
CUDA_VISIBLE_DEVICES=1  # 두 번째 GPU 사용
```

**6. 메모리 최적화 설정 확인:**
```bash
# .env 파일에서
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### 예방 조치

- 모델 로딩 후 불필요한 프로세스 종료
- 추론 전 GPU 메모리 여유 확인
- 동시 요청 수 제한 (API Gateway 레벨)

---

## Step 6.3: 포트 충돌

### 증상
서버 시작 시 "Address already in use" 에러가 발생합니다.

### 포트 사용 확인

```bash
# 포트 8000 사용 중인 프로세스 확인
sudo lsof -i :8000

# 또는
sudo netstat -tulpn | grep 8000

# 또는
sudo ss -tulpn | grep 8000
```

**예상 출력:**
```
COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
python3  1234 user   3u  IPv4  12345      0t0  TCP *:8000 (LISTEN)
```

### 해결 방법

**1. 프로세스 종료:**
```bash
# PID로 종료
kill 1234

# 강제 종료
kill -9 1234
```

**2. Docker 컨테이너 확인 및 중지:**
```bash
docker ps
docker stop cesco-api
```

**3. 대체 포트 사용:**
```bash
# .env 파일에서
PORT=8001

# 또는 start_server.sh 실행 시
./start_server.sh 8001
```

**4. systemd 서비스 확인:**
```bash
sudo systemctl status cesco-api
sudo systemctl stop cesco-api
```

### Nginx 포트 설정 변경 (포트 변경 시)

```bash
sudo nano /etc/nginx/sites-available/cesco-api
```

**proxy_pass 수정:**
```nginx
location / {
    proxy_pass http://127.0.0.1:8001;  # 변경된 포트
    ...
}
```

**Nginx 재시작:**
```bash
sudo nginx -t
sudo systemctl restart nginx
```

---
  
## Step 6.4: 느린 추론 속도

### 증상
API 응답이 느리거나 시간이 오래 걸립니다.

### 진단

**1. GPU 사용 확인:**
```bash
nvidia-smi
```

GPU Utilization이 0%이면 CPU로 실행 중입니다.

**2. 추론 시간 측정:**
```bash
time curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input_text": "테스트"}'
```

**3. 로그 확인:**
```bash
tail -f ~/cesco-inference-server/logs/service.log
```

### 해결 방법

**1. GPU 사용 확인:**
```python
# Python에서 확인
python3 << EOF
import torch
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
EOF
```

**CUDA가 False인 경우:**
- NVIDIA 드라이버 재확인
- CUDA Toolkit 재확인
- PyTorch CUDA 버전 확인

**2. temperature 파라미터 조정:**
```bash
# temperature를 0.1로 낮춤 (더 빠르고 결정적)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input_text": "테스트", "temperature": 0.05}'
```

**3. max_new_tokens 최적화:**
```bash
# 필요한 만큼만 생성 (기본 512에서 256으로)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input_text": "테스트", "max_new_tokens": 256}'
```

**4. 배치 처리 활용:**
```bash
# 여러 요청을 묶어서 처리
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '[{"input_text": "요청1"}, {"input_text": "요청2"}]'
```

**5. 모델 최적화 (재학습 시):**
- 더 작은 base model 사용 (예: llama-3-8b 대신 llama-2-7b)
- Quantization 확인 (4-bit 사용 중)

### 성능 벤치마크

**정상 범위:**
- GPU 사용 시: 1-3초 (512 tokens)
- CPU 사용 시: 30-60초 (권장하지 않음)

---

## Step 6.5: Docker 빌드 실패

### 증상
Docker 이미지 빌드 중 에러가 발생합니다.

### 일반적인 오류

**오류 1: CUDA 버전 불일치**

**증상:**
```
ERROR: Could not find a version that satisfies the requirement torch==2.8.0+cu128
```

**해결:**
```bash
# Dockerfile에서 CUDA 버전 변경
# FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
# 를 호스트 CUDA 버전과 일치시킴

# 호스트 CUDA 버전 확인
nvidia-smi | grep "CUDA Version"

# CUDA 12.2인 경우
nano Dockerfile
# FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 로 변경
```

**오류 2: 메모리 부족**

**증상:**
```
ERROR: failed to solve: executor failed running ...
```

**해결:**
```bash
# Docker 빌드 메모리 증가
sudo nano /etc/docker/daemon.json
```

**추가:**
```json
{
  "storage-opts": ["dm.basesize=30G"]
}
```

```bash
sudo systemctl restart docker
```

**오류 3: 네트워크 타임아웃**

**해결:**
```bash
# 빌드 시 타임아웃 증가
docker build --network=host -t cesco-inference-server .

# 또는 DNS 변경
sudo nano /etc/docker/daemon.json
```

**추가:**
```json
{
  "dns": ["8.8.8.8", "8.8.4.4"]
}
```

**오류 4: 권한 문제**

**해결:**
```bash
# Docker 그룹에 사용자 추가
sudo usermod -aG docker $USER
newgrp docker

# 또는 sudo 사용
sudo docker build -t cesco-inference-server .
```

### 빌드 캐시 삭제 후 재시도

```bash
# 캐시 없이 빌드
docker build --no-cache -t cesco-inference-server .

# 오래된 이미지 정리
docker system prune -a
```

### CPU 전용 빌드 (GPU 없는 환경)

```bash
# docker-build.sh 스크립트로
./docker-build.sh 8000 ./outputs_final true

# 또는 수동으로
docker run -d \
  --name cesco-api \
  -p 8000:8000 \
  -v $(pwd)/outputs_final:/app/model:ro \
  cesco-inference-server
```

---

## Step 6.6: 연결 거부 오류

### 증상
외부에서 API에 접근할 수 없습니다.

### 진단 단계

**1. 로컬 접속 확인:**
```bash
curl http://localhost:8000/health
```

성공하면 서버는 정상입니다.

**2. 호스트 바인딩 확인:**
```bash
# .env 파일 확인
grep HOST .env
```

**반드시:**
```
HOST=0.0.0.0
```

`127.0.0.1`이면 로컬에서만 접근 가능합니다.

**3. 방화벽 확인:**
```bash
sudo ufw status
```

포트 80, 443 (Nginx) 또는 8000 (직접 접근)이 열려있어야 합니다.

**4. 포트 리스닝 확인:**
```bash
sudo netstat -tulpn | grep 8000
```

**예상 출력:**
```
tcp        0      0 0.0.0.0:8000            0.0.0.0:*               LISTEN      1234/python3
```

`0.0.0.0:8000`이어야 하며, `127.0.0.1:8000`이면 HOST 설정이 잘못되었습니다.

### 해결 방법

**1. HOST 설정 수정:**
```bash
nano .env
# HOST=0.0.0.0로 변경

# 서비스 재시작
sudo systemctl restart cesco-api
# 또는
docker restart cesco-api
```

**2. 방화벽 규칙 추가:**
```bash
# 직접 접근 허용 (권장하지 않음)
sudo ufw allow 8000/tcp

# 또는 Nginx 포트 확인
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

**3. 클라우드 보안 그룹 확인 (클라우드 환경):**

**AWS:**
- EC2 인스턴스의 Security Group 확인
- Inbound 규칙에 포트 80, 443 추가

**GCP:**
- Firewall rules 확인
- tcp:80, tcp:443 허용 규칙 추가

**Azure:**
- Network Security Group 확인
- Inbound security rules 추가

**4. Nginx 설정 확인:**
```bash
sudo nginx -t
sudo systemctl status nginx

# Nginx 로그 확인
sudo tail -f /var/log/nginx/error.log
```

**5. SELinux 확인 (CentOS/RHEL):**
```bash
# SELinux 상태 확인
getenforce

# 일시적 비활성화 (테스트용)
sudo setenforce 0

# 영구 비활성화 (권장하지 않음)
sudo nano /etc/selinux/config
# SELINUX=disabled
```

### 외부 접속 테스트

**다른 머신에서:**
```bash
curl http://your-server-ip/health
curl http://your-domain.com/health
```

**브라우저에서:**
```
http://your-server-ip/docs
https://your-domain.com/docs
```

---

**[← 이전: 5부 프로덕션 강화](05-production-hardening.md)** | **[목차로 돌아가기](README.md)**
