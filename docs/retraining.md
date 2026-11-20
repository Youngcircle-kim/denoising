# MSTMN 논문 기반 재학습

이 코드는 **논문을 바탕으로 작성한 비공식 구현**입니다. 저자 홈페이지의 실제 링크와 공개 저장소를
다시 조사했지만 MSTMN 공식 코드·가중치를 확보하지 못했습니다.
[공식 아티팩트 재확인 기록](../research/MSTMN_official_artifact_recheck.md)을 참조하세요.
다른 논문의 모델·가중치를 MSTMN으로 대체하지 않았습니다.

현재 본 학습은 원격 서버에서 진행 중입니다. 실제 실행 위치·최근 checkpoint·검증 범위는
[실행 기록](../research/MSTMN_retraining_status.md)을 참조하세요.

## 구현과 논문 조건

[모델 구현 명세](../research/MSTMN_implementation_spec.md)는 논문 명시 사항과 로컬 선택을 구분합니다.
기본 MSTMN 한 단계 모델을 구현했고, MSTMN-s/+는 구현하지 않았습니다.
Gaussian 4단계/Laplacian 3단계, RAW 패킹과 pixel-unshuffle, 과거 noisy 3개의 메모리,
이전 복원 정렬, 잡음·정렬 가중치, 식(16c)의 융합과 pyramid 합성을 포함합니다.

학습 조건은 [configs/mstmn_retrain.json](../configs/mstmn_retrain.json)에 있습니다.

| 항목 | 설정 | 근거 |
| --- | --- | --- |
| 학습 데이터 | CRVD scene 1–6, ISO 5조건 | 논문 IV-A |
| 패치 | 패킹 후 128×128, 원 Bayer 256×256 | 논문 IV-A |
| 시퀀스 / 배치 | 25 / 유효 batch 16 | 논문 IV-A |
| 메모리 | 과거 noisy 3개 | 논문 IV-B |
| 손실 / 최적화 | L1 / Adam, beta .9/.999, epsilon 1e-8 | 논문 IV-A·식(18) |
| 학습률 | .001, update 15000부터 .0005, 40000부터 .0001 | 논문 값 + 경계 시점은 로컬 선택 |
| 목표 | 120000 optimizer updates | 논문 IV-A |
| 정밀도 | FP32, AMP·TF32 비활성화 | 로컬 선택, 논문 정밀도 미확인 |

공식 학습 코드가 없어 다음은 **확정된 논문 설정이 아닙니다**.

- CRVD GT에 저자 제공 계수의 Poisson–Gaussian 잡음을 생성하는 synthetic-first 학습.
  DN clipping과 uint16 절삭은 [CRVD 제공 코드](https://github.com/cao-cong/RViDeNet/blob/master/synthesize_noise.py)를 따릅니다.
  실측 noisy만으로 학습하는 `dataset.mode=real`도 구현했지만 기본 실행에는 사용하지 않습니다.
- 7개 원본 프레임을 길이 25로 만드는 random phase/direction ping-pong. 첫 프레임·과거 메모리 초기값은 첫 noisy입니다.
- 공간 증강 없음, packed crop 시작점은 짝수. 전체 25프레임 BPTT, 프레임 평균 L1, weight decay 0, gradient clipping 없음.
- Binomial 5×5 Gaussian, reflect padding, even-phase downsample, bilinear upsample/warp, 스케일별 독립 파라미터.
- 메모리 패치 4×4/stride 4. 식(14)의 모호한 행렬 차원은 원문의 frame+channel 합치기를 따라
  feature attention으로 해석하고 외곽 곱 순서를 조정했습니다. sigmoid를 유지하며 softmax·추가 정규화는 적용하지 않습니다.
- SPyNet은 공개 Sintel 체크포인트를 고정해서 사용하며, RAW에서 R/G평균/B를 구성하는 bridge를 둡니다.
  **이것은 MSTMN 체크포인트가 아닙니다.** 저자가 쓴 flow 파일·입력 변환·동결 정책은 미확인입니다.

이 차이 때문에 학습 완료만으로 논문 Table III를 재현했다고 판정하지 않습니다.
seed를 고정하고 표본·RNG·소스를 기록하지만 CUDA `grid_sample` 역전파의 비결정성으로 GPU bitwise 일치를 보장하지 않습니다.

## 서버 환경과 데이터

- 프로젝트: `C:\Users\HP\yhkim\denoising`.
- 학습 환경: `.venv-train`, Python 3.13.17 / PyTorch 2.10.0+cu130.
- 장치: NVIDIA RTX PRO 6000 Blackwell Max-Q, CUDA compute capability 12.0.
- 데이터 루트: `E:\datasets\crvd`, GT/noisy collection 안에 scene 1–6.
- 원본 ZIP: `E:\datasets\archives\crvd`, 출처·해시 receipt: `E:\datasets\receipts\crvd`.
- SPyNet: `checkpoints\upstream\spynet_sintel_final-3d2a1287.pth`.

서버에는 환경을 준비했습니다. 다른 Windows 장치에서는 Python 3.11–3.13과 호환 GPU/driver를 확인한 뒤
공식 [PyTorch 배포](https://download.pytorch.org/whl/cu130/torch/)를 사용합니다.

```powershell
python -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv-train\Scripts\python.exe -m pip install -r requirements-train.txt
```

학습 ZIP 확보·해제는 별도 명령이며 이미 완료한 목적지를 덮어쓰지 않습니다.

```powershell
.\.venv-download\Scripts\python.exe scripts\download_benchmarks.py download --split train --data-root E:\datasets
.\.venv-download\Scripts\python.exe scripts\extract_crvd_train.py --data-root E:\datasets
```

SPyNet은 [VRT 프로젝트의 공개 파일](https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth)을
사용하고 SHA-256 접두어 `3d2a1287`를 확인했습니다. 코드 호환 근거는
[BasicSR SPyNet](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/spynet_arch.py)이며
[Apache 라이선스 사본](../third_party/BasicSR-LICENSE.txt)을 함께 보관했습니다.
학습 시 파일의 전체 SHA-256도 checkpoint에 저장합니다.

## 작은 실행 후 본 학습·재개

아래 `mstmn-crvd-run01`은 예시 새 폴더 이름입니다. 실제 실행 위치와 상태는 별도 실행 기록을 확인하세요.

```powershell
# 논문과 같은 입력 크기·배치·시퀀스로 처음 3회 확인
.\.venv-train\Scripts\python.exe -m mstmn_train.train --data-root E:\datasets\crvd --flow-checkpoint checkpoints\upstream\spynet_sintel_final-3d2a1287.pth --output E:\denoising_runs\mstmn-crvd-run01 --max-steps 3

# 동일 표본 순서·모델·optimizer에서 목표 120000까지 재개, SSH와 독립 실행
.\.venv-train\Scripts\python.exe scripts\launch_training.py --data-root E:\datasets\crvd --flow-checkpoint checkpoints\upstream\spynet_sintel_final-3d2a1287.pth --output E:\denoising_runs\mstmn-crvd-run01 --resume E:\denoising_runs\mstmn-crvd-run01\last.pth

# 진행 확인
Get-Content E:\denoising_runs\mstmn-crvd-run01\status.json
Get-Content E:\denoising_runs\mstmn-crvd-run01\train.jsonl -Tail 5

# 현재 optimizer update가 끝난 경계에서 저장하고 중지
New-Item -ItemType File E:\denoising_runs\mstmn-crvd-run01\STOP
```

재개 전에는 완료 상태와 PID를 확인하고 `STOP` 파일을 제거합니다. 기존 `.train.lock`이 있으면
프로세스가 살아 있는지 확인해야 합니다. 잠금 파일을 무조건 지우거나 같은 출력에 중복 실행하지 않습니다.
설정·flow SHA·구현 소스 SHA가 checkpoint와 다르면 재개를 거부합니다.
OOM이 확인된 경우에만 `--microbatch 8`처럼 줄일 수 있고, 유효 배치 16은 gradient accumulation으로 유지합니다.
이 경우의 미세한 부동소수점 차이와 사용한 microbatch 값도 기록합니다.

| 산출물 | 내용 |
| --- | --- |
| `recipe.json` | 원본 120000-step 조건과 모든 가정 |
| `status.json` | PID, 완료 update 수, 최근 loss·실측 step 시간·peak VRAM, checkpoint 위치, 실패 원인 |
| `train.jsonl` | 첫 update·10회 간격·마지막 update의 학습 기록 |
| `last.pth` | 첫 update·100회 간격·정상 종료/STOP 때 저장한 모델·optimizer·RNG·소스 hash·소비 표본 위치 |
| `../launches/*` | SSH 독립 프로세스의 명령·PID·stdout·stderr |

checkpoint는 완성된 optimizer update 경계에서 임시 파일로 쓴 후 교체합니다.
에러가 난 반쪽 update는 새 checkpoint로 저장하지 않고, 마지막 정상 checkpoint 위치를 상태에 남깁니다.
데이터·모델 가중치·실행 출력은 Git에 넣지 않습니다.

## 학습 가중치 평가

학습 결과는 기존 평가 도구에 연결합니다. 아래 모델명과 결과는 **비공식 재학습**으로 표시합니다.
학습 엔진은 시험 장면을 읽거나 시험 지표로 모델을 선택하지 않습니다.

```powershell
.\.venv-train\Scripts\python.exe -m denoising_eval evaluate --data-root E:\datasets\crvd --output results\mstmn-retrained-01 --adapter mstmn_train.adapter:create_predictor --checkpoint E:\denoising_runs\mstmn-crvd-run01\last.pth --model-name MSTMN-unofficial-retrain --device cuda:0
```

다른 컴퓨터에서는 같은 SPyNet 파일 경로를 환경 변수 `MSTMN_FLOW_CHECKPOINT`로 지정할 수 있습니다.
평가의 clip·RAW 도메인·SSIM·집계 조건은 [평가 사용법](evaluation.md)의 미확인 조건을 그대로 유지합니다.
학습 중 성능이나 총 소요 시간을 추정해서 보고하지 않습니다. 완료 후 고정한 시험 manifest로 평가하고
논문 수치와 차이 및 재현 가정을 함께 보고해야 합니다.
