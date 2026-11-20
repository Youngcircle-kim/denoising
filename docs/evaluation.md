# CRVD RAW 평가 코드

공식 MSTMN 모델을 연결할 수 있는 독립 평가 도구입니다. 공식 코드·가중치는 아직 확보하지 못했습니다.
공식 MSTMN 구현이나 저자가 공개한 가중치는 포함하지 않습니다.
후속 요청으로 추가한 [논문 기반 비공식 재학습 모듈](retraining.md)은 별도로 구분합니다.
데이터·지표·반복 실행을 준비한 상태이며, Table III 재현 성공을 주장하지 않습니다.

## 원격 서버에서 바로 실행

작업 위치는 `C:\Users\HP\yhkim\denoising`, 데이터는 `E:\datasets\crvd`입니다.
다운로드용 `.venv-download`와 별도로 평가용 `.venv-eval`을 준비했습니다.
평가 전용 `.venv-eval`에는 PyTorch·CUDA 및 공식 모델 의존성을 설치하지 않았습니다.
후속 재학습용 `.venv-train`에는 별도로 GPU PyTorch를 준비했습니다.
Windows CPython 3.13.17에서 실제 설치·검증한 전체 패키지 버전은
[requirements-eval.windows.lock.txt](../requirements-eval.windows.lock.txt)에 기록했습니다.

다른 장치에서 처음 준비할 때는 Python 3.11–3.13으로 다음을 실행합니다.
고정한 패키지는 [NumPy 2.2.6](https://pypi.org/project/numpy/2.2.6/),
[tifffile 2025.10.4](https://pypi.org/project/tifffile/2025.10.4/),
[scikit-image 0.25.2](https://pypi.org/project/scikit-image/0.25.2/)입니다.

```powershell
cd C:\Users\HP\yhkim\denoising
python -m venv .venv-eval
.\.venv-eval\Scripts\python.exe -m pip install -r requirements-eval.txt
```

서버에는 이미 준비되어 있으므로 다음 검사부터 사용할 수 있습니다. 출력 폴더는 매번 새 이름을 사용합니다.

```powershell
# 모든 선택 파일의 TIFF 헤더 검사 + 첫 GT/noisy 쌍의 픽셀 로딩
.\.venv-eval\Scripts\python.exe -m denoising_eval inspect --data-root E:\datasets\crvd --output results\inspect-01

# 첫 시퀀스의 입력 복사 진단: 모델 추론 결과가 아닙니다.
.\.venv-eval\Scripts\python.exe -m denoising_eval evaluate --data-root E:\datasets\crvd --output results\identity-01 --baseline identity --max-sequences 1

# 합성 TIFF·가짜 adapter를 사용하는 오프라인 테스트
.\.venv-eval\Scripts\python.exe -m unittest discover -s tests -v
```

`inspect`는 모든 TIFF의 전체 픽셀 무결성 검사를 뜻하지 않습니다. 선택된 파일의 헤더와 한 쌍의 픽셀을
검사합니다. 압축 해제 때 수행한 전체 파일 해시 검증은 [별도 기록](../research/CRVD_test_download_status.md)에 있습니다.
`evaluate`는 사용한 각 GT/noisy 파일을 실제로 디코딩하고 유한한 값인지 확인합니다.

## 공식 모델 연결

연결 계약은 [backends.py](../denoising_eval/backends.py)의 `Predictor`입니다.
공식 아티팩트를 확보한 뒤, 공식 전처리·모델 API에 맞춘 Python 모듈의 factory를 연결합니다.
다음 명령의 `my_mstmn_bridge:create_predictor`와 가중치 경로는 실제 제공 모듈·파일로 바꿔야 합니다.

```powershell
.\.venv-eval\Scripts\python.exe -m denoising_eval evaluate --data-root E:\datasets\crvd --output results\mstmn-01 --adapter my_mstmn_bridge:create_predictor --checkpoint checkpoints\mstmn.pth --model-name MSTMN --device cuda:0
```

Factory는 `checkpoint: pathlib.Path`, `device: str`, `protocol: dict`를 키워드 인자로 받고,
다음 두 메서드를 갖는 객체를 반환해야 합니다.

| 메서드 | 계약 |
| --- | --- |
| `reset(sequence)` | `scene`, `iso`, `noisy_index`, `sequence_id`만 전달됩니다. 장면·ISO·noisy 반복마다 이전 복원과 noisy 메모리를 초기화합니다. |
| `predict(noisy, *, frame_index)` | `noisy`는 정규화된 `float32` HWC 배열 `(H/2, W/2, 4)`, RGBG 순서입니다. `frame_index`는 실제 파일명 기준 1–7입니다. 같은 모양의 유한한 실수 ndarray를 반환합니다. |

GT는 adapter에 전달하지 않습니다. 입력은 `(DN - 240) / (4095 - 240)`이며, GBRG에서 채널을
`R=(1,0), G=(0,0), B=(0,1), G=(1,1)` 위치로 뽑습니다. 공식 모델의 채널 순서가 다르면 bridge에서
재배열하고 반환할 때 되돌립니다. Torch 모델이라면 HWC↔NCHW와 device 이동도 bridge에서 수행합니다.
Pixel-unshuffle, optical flow, 이전 복원, 과거 noisy 3개, noise 계수, padding, 타일링은 실제 공식 코드에
맞춰 bridge가 처리해야 합니다. 현재 프레임 외 미래 프레임이나 GT로 상태를 초기화하지 않습니다.
첫 프레임·메모리 초기화 규칙은 이 평가 코드가 추정하지 않습니다.

외부 adapter와 가중치는 신뢰할 수 있는 자료만 사용합니다. PyTorch state_dict를 읽는 bridge라면
`torch.load(checkpoint, map_location='cpu', weights_only=True)`와 `load_state_dict(..., strict=True)`를
사용하고, 로드 실패를 random 가중치나 다른 모델로 자동 대체하지 않습니다.
근거: [torch.load 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.load.html),
[Module.load_state_dict](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict).
모델 이름을 `MSTMN`으로 지정하는 것만으로 공식 모델의 출처나 프로토콜을 검증하지 않습니다.

## 현재 평가 조건과 미확인 사항

조건은 [configs/crvd_raw_test.json](../configs/crvd_raw_test.json)에 고정했습니다.
이는 공식 MSTMN 설정을 추출한 파일이 아니라 확인 가능한 데이터와 명시적인 선택을 기록한 설정입니다.

| 항목 | 현재 동작 | 근거·상태 |
| --- | --- | --- |
| 장면·ISO | scene 7–11, ISO 1600/3200/6400/12800/25600 | 논문 분할 + 데이터 저자 값 |
| 입력/GT | `frameN_noisyK.tiff` / `frameN_clean_and_slightly_denoised.tiff` | 실제 파일 목록. noisy 폴더의 `frameN_clean.tiff`는 입력·GT로 사용하지 않음 |
| 반복·시간축 | noisy 0–9 각각 독립적인 7프레임 시퀀스 | 배포본 전체 선택. 논문과 같은 noisy realization 선택인지 미확인 |
| 기본 수량 | 250시퀀스, 1,750쌍, 고유 noisy 1,750개 + GT 175개 | GT를 noisy 반복별로 재사용. 해제 파일 2,100개와 평가 쌍 수는 다름 |
| DN/패킹 | black 240, white 4095, RGBG 4채널 | 데이터 저자 값 + 로컬 연결 계약. 공식 MSTMN 인덱스 대응은 미확인 |
| clip | 입력 false / GT false / 예측 true | 로컬 선택. 입력 복사 진단에도 예측 clip이 적용됨 |
| 평가 도메인 | 2D Bayer mosaic, `data_range=1` | 로컬 선택. 공식 packed/채널별 처리 방식 미확인 |
| crop·warm-up | border 0, 제외 프레임 0 | 로컬 선택. warm-up을 설정하면 추론은 하고 지표만 제외 |
| PSNR | float64 MSE에서 프레임별 PSNR 계산 | 평균 MSE 기반 PSNR과 구분 |
| SSIM | Gaussian 11×11, sigma 1.5, K1=.01, K2=.03, sample covariance false | scikit-image 구현 선택. 공식 라이브러리·창 크기 미확인 |
| 집계 | frame mean과 sequence mean 모두 기록, ISO·장면별 내역 포함 | 공식 평균 순서 미확인. 표 수치와 자동 성공/실패 판정 없음 |

SSIM은 [scikit-image 공식 안내](https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html#skimage.metrics.structural_similarity)의
명시적 옵션을 사용합니다. 경계의 SSIM 창 처리는 라이브러리에 따르며, `crop_border`는 그 전에 적용하는 별도 옵션입니다.
기본값을 변경한 실험은 설정 파일을 복사해 수정하고 결과와 함께 보관하세요.
서브셋 설정이나 `--max-sequences` 실행은 `partial_run=true`로 표시합니다.

## 결과와 실패 처리

| 파일 | 내용 |
| --- | --- |
| `config.json`, `manifest.json` | 실제 설정과 시간순 입력·GT 상대 경로 |
| `inspection.json` | TIFF 헤더·첫 쌍 범위와 검사 환경 |
| `run.json` | 모델·checkpoint SHA-256·평가 소스 SHA-256·패키지 버전, 실행 상태 |
| `frames.csv` | 프레임별 MSE/PSNR/SSIM |
| `summary.json` | 완료 실행의 전체·시퀀스·장면·ISO 집계와 미검증 조건 |
| `failure.json` | 실패 원인. 종료 코드는 0이 아니며 partial 출력은 진단용으로 보존 |
| `predictions/.../frameN.npy` | `--save-predictions` 선택 시 정규화된 float32 Bayer mosaic |

완벽한 복원의 PSNR은 CSV에 `inf`, JSON에 표준 JSON 문자열 `"Infinity"`로 기록합니다.
기존 출력 폴더와 데이터 폴더 내부 출력은 거부합니다. 누락 파일·잘못된 shape·NaN/Inf·가중치 로드 실패가
발생하면 입력 복사 모델로 자동 전환하지 않습니다. `summary.json`의 논문값은 **비교 참고값**이며,
`official_reproduction=false`, `comparison_validated=false`를 유지합니다.
공식 코드·가중치와 미확인 평가 조건을 해소한 뒤 별도 근거 검토가 필요합니다.

학습, 모델 구현, ISP/sRGB 평가, Table V 속도·FLOPs 측정은 이 코드의 범위에 포함하지 않습니다.

## 수행한 검증

2026-11-20 로컬과 원격 Windows에서 오프라인 테스트 **46개**가 모두 통과했습니다.
실제 CRVD 선택 전체의 **250시퀀스·1,750쌍·고유 TIFF 1,925개** 헤더 검사를 통과했고,
noisy는 1080×1920 uint16, GT는 같은 크기의 float32임을 확인했습니다.
첫 시퀀스의 **7프레임 입력 복사 진단**은 종료 코드 0으로 완료했습니다.
이는 공식 모델 추론이 아닙니다. 실행한 코드 해시·설정·환경·검사 및 진단 결과는
[검증 기록](../research/MSTMN_eval_validation.json)에 보관했습니다.
