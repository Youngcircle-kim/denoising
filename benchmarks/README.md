# Benchmark 다운로드 준비

2026-11-20 후속 실행으로 원격 `.venv-download`(Python 3.13.17 / gdown 6.5.2)와
CRVD 시험 ZIP 10개를 `E:\datasets`에 준비하고 검증했습니다.
추가 요청으로 `E:\datasets\crvd`에 압축 해제한 2,100개 파일(8.79 GiB)도 검증했습니다.
[완료 기록](../research/CRVD_test_download_status.md)과
[파일별 inventory](../research/CRVD_test_download_inventory.json)를 확인할 수 있습니다.

## 기본 대상과 공식 출처

| 대상 | 선택 범위 | 공식 배포 |
| --- | --- | --- |
| CRVD `test` (기본) | 실내 장면 7–11, GT + noisy ZIP 10개 | [RViDeNet Dataset](https://github.com/cao-cong/RViDeNet#dataset), [Google Drive](https://drive.google.com/drive/folders/1TZhNrsJ4tmKaU7b9b_9hkJIhwMi9Tfk3) |
| CRVD `train` | 실내 장면 1–6, ZIP 12개 | 같은 공식 배포 |
| CRVD `indoor` | 실내 장면 1–11, ZIP 22개 | 같은 공식 배포 |
| CRVD `outdoor` | 공식 outdoor 폴더의 ZIP 10개, 정성 평가용 | 같은 공식 배포 |
| Real-RawVSR | MSTMN Table IV용 ×2 시험 데이터, 수동 접근 | [공식 Dataset](https://github.com/zmzhang1998/Real-RawVSR#dataset)의 [Baidu](https://pan.baidu.com/s/1G5_zCt_L_POzwb_mWgpuDA), 추출키 `hxyl` |

스크립트 작성 시 공식 공개 폴더의 메타데이터로 CRVD ZIP 파일명·Drive ID를
[sources.json](sources.json)에 기록했습니다(2026-11-20). 후속 실제 다운로드에서 시험 ZIP
10개의 ISO별 파일 목록도 확인했습니다. ISO를 선택적으로 받는 기능은 없고 장면 ZIP 단위로 받습니다.
`sources.json`의 크기·해시는 배포자 정보가 미확인인 상태를 유지합니다. 다운로드한 시험 데이터의
실측 크기·로컬 해시는 별도 inventory에 있으며 전체 데이터셋의 용량·시간으로 일반화하지 않습니다.

두 데이터셋의 공식 README는 학술·비상업 용도 및 CC BY-NC-SA 4.0을 명시합니다.
출처와 이용 조건을 함께 확인하세요. SID·MOT·RViDeNet 가중치는 MSTMN의 필수 자료로
확인되지 않아 기본 다운로드에 포함하지 않았습니다. Real-RawVSR README의 Google Drive는
**모델 가중치 링크**이므로 데이터셋 링크로 사용하지 않습니다.

## 1. 계획 확인: 다운로드 없음

Windows PowerShell 5.1의 기본 `Plan`은 Python·gdown 없이 동작합니다.
네트워크 요청이나 데이터 폴더 생성도 하지 않습니다.

```powershell
cd C:\Users\HP\yhkim\denoising
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_benchmarks.ps1 -Action Plan -DataRoot E:\datasets

# 한 장면의 GT/noisy ZIP 2개만 확인
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_benchmarks.ps1 -Action Plan -Scenes 7 -DataRoot E:\datasets

# JSON 출력 또는 학습 대상 확인
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_benchmarks.ps1 -Action Plan -Json
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_benchmarks.ps1 -Action Plan -Split train -DataRoot E:\datasets
```

`-ExecutionPolicy Bypass`는 이 PowerShell 프로세스에만 적용하며 영구 설정을 바꾸지 않습니다.
Python이 있는 장치에서는 다음도 오프라인 계획입니다.

```bash
python scripts/download_benchmarks.py
python scripts/download_benchmarks.py plan --split test --scenes 7 8 --data-root /path/to/datasets --json
```

`test`에서 장면 1을 지정하거나 같은 장면을 중복 지정하면 실패합니다.
Table III의 전체 평균과 비교할 때는 시험 장면 7–11 전체와 필요한 ISO·프레임 조건을
확정해야 합니다. 한 장면 다운로드는 기능 점검용입니다.

## 2. 실제 다운로드: 추후 명시적으로 실행

Python **3.10 이상**의 실제 인터프리터와 `gdown==6.5.2`가 필요합니다.
다운로드 전용 환경이며 모델 학습 환경이 아닙니다. 이 프로젝트의 원격 가상환경에는 이미 준비했습니다.
스크립트 자체는 Python·pip·패키지를 자동 설치하지 않습니다. 다른 장치에서는 WindowsApps 별칭만으로
실제 Python이 설치됐다고 판단하지 않습니다.

실제 Python 설치 후 아래처럼 별도 환경을 준비해 실행할 수 있습니다.

```powershell
python -m venv .venv-download
.\.venv-download\Scripts\python.exe -m pip install -r .\requirements-download.txt

# CRVD 시험 ZIP 10개. 이 명령부터 실제 대용량 데이터 전송이 발생합니다.
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_benchmarks.ps1 -Action Download -Python .\.venv-download\Scripts\python.exe -DataRoot E:\datasets
```

동일한 Python CLI는 Windows·macOS·Linux에서 사용할 수 있습니다.

```bash
python scripts/download_benchmarks.py download --dataset crvd --split test --data-root /path/to/datasets
python scripts/download_benchmarks.py download --split train --data-root /path/to/datasets
```

같은 데이터 폴더에서는 다운로드를 동시에 여러 번 실행하지 마세요.
Google Drive의 quota·접근 정책·링크 변경으로 실패할 수 있습니다. 오류를 성공으로 처리하거나
다른 미확인 미러로 자동 전환하지 않습니다. 파일 ID는 공식 폴더 메타데이터와 다시 대조해 갱신합니다.

## 3. 중단·재실행·검증

저장 구조:

```text
<data-root>/
  archives/crvd/indoor_raw_gt/indoor_raw_gt_scene7.zip
  archives/crvd/indoor_raw_noisy/indoor_raw_noisy_scene7.zip
  receipts/crvd/<collection>/<archive>.json
  .staging/crvd/<Drive-ID>/...
```

- gdown의 부분 파일을 staging 폴더에 유지하고 같은 `Download` 명령으로 이어받습니다.
- 완성된 staging ZIP의 CRC를 검사하고 SHA-256을 계산한 뒤 최종 archive 위치로 이동합니다.
- 기존 archive는 출처 ID·크기·SHA-256·ZIP CRC를 확인한 후에만 건너뜁니다.
- quota 안내 HTML, 손상 ZIP, 출처 기록 없는 기존 파일을 완료로 처리하지 않습니다.
  이런 파일은 자동 삭제하지 않으며 오류의 경로를 보고합니다. 파일을 별도로 보관하거나
  의도적으로 제거한 뒤 다시 실행하세요. `.part` 파일은 이어받기용이므로 임의로 지우지 마세요.
- 공식 SHA-256이 없는 현재 상태에서 로컬 SHA-256은 **이후 변경 탐지용 기록**입니다.
  배포자 체크섬으로 검증했다고 주장하지 않습니다.
- ZIP 10개 검증 성공은 다운로드 패키지 검증입니다. 내부 장면·ISO·프레임의 완전성이나
  논문 평가 프로토콜까지 검증한 것은 아닙니다.

다운로드 이후 재검증은 네트워크나 gdown 없이 전체 파일을 읽습니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_benchmarks.ps1 -Action Verify -Python .\.venv-download\Scripts\python.exe -DataRoot E:\datasets
```

```bash
python scripts/download_benchmarks.py verify --split test --data-root /path/to/datasets
```

다운로드 스크립트는 자동 압축 해제를 수행하지 않습니다. 기본 `data/`, `datasets/`,
다운로드 전용 가상환경은 Git에서 제외했습니다. 다른 위치를 지정해도 데이터 파일을 Git에 추가하지 마세요.

## 4. CRVD 시험 데이터 압축 해제

원격 서버에서는 아래 별도 작업까지 완료했습니다. 원본 ZIP을 보존하고 GT와 noisy를 분리합니다.

```text
E:\datasets\crvd\indoor_raw_gt\scene7\ISO1600\...
E:\datasets\crvd\indoor_raw_noisy\scene7\ISO1600\...
# 각 collection에 scene7–11 및 ISO 1600/3200/6400/12800/25600
```

다른 데이터 루트에서 처음 해제할 때는 다음 명령을 사용합니다.
**이미 해제가 끝난 원격 폴더에서는 재실행할 필요가 없습니다.** 기존 목적지는 덮어쓰지 않고 실패합니다.

```powershell
.\.venv-download\Scripts\python.exe .\scripts\extract_crvd_test.py --data-root E:\datasets
```

이 스크립트는 완료된 `receipts/crvd-test-inventory.json`이 필요합니다. 원본 ZIP의 로컬 SHA-256,
경로·파일 수·ISO별 수량·디스크 공간을 확인한 뒤 새 staging에서 해제합니다. ZIP CRC를 확인하고,
저장한 파일을 다시 읽어 크기·CRC·SHA-256을 대조한 다음 최종 `crvd` 폴더로 이동합니다.
파일별 결과는 `receipts/crvd-test-extraction.json`에 기록하며 프로젝트에도
[검증 목록 사본](../research/CRVD_test_extraction_inventory.json)을 보관했습니다.
압축 해제는 TIFF 디코딩·RAW 전처리·평가 프로토콜 검증을 포함하지 않습니다.

## Real-RawVSR

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_benchmarks.ps1 -Action Plan -Dataset real-rawvsr -DataRoot E:\datasets
```

이 명령은 공식 Baidu 링크·추출키·보관 경로만 출력합니다. 로그인·클라이언트가 필요한 경우
공식 다운로드 흐름으로 받아 `<data-root>/manual/real-rawvsr`에 원본을 보관하는 경로입니다.
`Download`·`Verify`를 지정하면 수동 접근이 필요하다는 오류로 종료합니다.
임의의 공개 URL을 만들거나 인증을 우회하지 않습니다.

MSTMN이 선택한 ×2 시험 영상 20개의 정확한 ID와 고품질 RAW 파일 선택·합성 잡음 규칙이
아직 확정되지 않아 전체 Real-RawVSR를 받았다고 바로 Table IV 입력이 완성되지는 않습니다.

## 구현 근거와 검증 범위

- 공개 Drive 접근은 [gdown v6.5.2](https://github.com/wkentaro/gdown/tree/v6.5.2)의
  `download(id=..., resume=True, retries=3, timeout=(15,60), use_cookies=False)`를 사용합니다.
  폴더 개수 제한을 우회하며 불완전한 목록을 성공 처리하는 대신 고정된 장면 ZIP ID를 사용합니다.
- 합성 소형 ZIP·가짜 downloader를 사용한 오프라인 테스트로 선택·재개·손상·변경 탐지를 검증합니다.
  후속 실행에서는 CRVD 시험 ZIP 10개의 실제 Google Drive 전송과 무결성 검사도 완료했습니다.
  다른 분할·미래 quota·Baidu 로그인을 검증한 것은 아닙니다.
- 테스트: `python -m unittest discover -s tests -v`.
