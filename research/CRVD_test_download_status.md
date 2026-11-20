# CRVD 메인 시험 데이터 다운로드·압축 해제 완료 기록

확인일: 2026-11-20 (Asia/Seoul). 사용자 후속 요청에 따라 실제 원격 설치·다운로드를 수행했다.

**완료:** Python 다운로드 전용 가상환경과 CRVD 실내 시험 장면 7–11의 GT·noisy ZIP 10개를 준비했고, 추가 요청에 따라 압축 해제까지 완료했다. 학습 장면 1–6, outdoor, Real-RawVSR, 모델·가중치는 다운로드하지 않았다. 전처리·학습·평가도 실행하지 않았다.

## 환경

| 항목 | 실제 결과 |
| --- | --- |
| 프로젝트 | `C:\Users\HP\yhkim\denoising` |
| Python | 3.13.17, 사용자 범위의 64비트 CPython |
| 가상환경 | `C:\Users\HP\yhkim\denoising\.venv-download` |
| 인터프리터 | `C:\Users\HP\yhkim\denoising\.venv-download\Scripts\python.exe` |
| 다운로드 도구 | gdown 6.5.2 |
| 패키지 검사 | `pip check` 통과 |
| Windows 코드 검증 | 오프라인 테스트 17개 통과 |
| 설치 패키지 기록 | 가상환경 내부 `packages.freeze.txt` 및 [완료 inventory](CRVD_test_download_inventory.json)의 `packages` |

이 환경은 다운로드 도구용이다. PyTorch·CUDA toolkit·MSTMN 실행 환경을 설치하거나 호환성을 검증한 것으로 해석하지 않는다.

## 자료와 실제 저장 위치

- 데이터 루트: `E:\datasets`.
- GT: `E:\datasets\archives\crvd\indoor_raw_gt`.
- noisy: `E:\datasets\archives\crvd\indoor_raw_noisy`.
- 장면: 7, 8, 9, 10, 11. 각 장면의 GT·noisy ZIP 한 쌍.
- 모든 ZIP에서 ISO 1600, 3200, 6400, 12800, 25600 폴더 확인.
- 압축 파일 합계: **6,719,303,964 bytes / 6.26 GiB**.
- ZIP의 디렉터리를 제외한 파일 목록 합계: **2,100개**.

| 장면 | GT ZIP MiB | noisy ZIP MiB | GT ZIP 내부 파일 수 | noisy ZIP 내부 파일 수 |
| --- | --- | --- | --- | --- |
| 7 | 120.36 | 1151.60 | 35 | 385 |
| 8 | 119.01 | 1171.19 | 35 | 385 |
| 9 | 115.17 | 1163.88 | 35 | 385 |
| 10 | 124.37 | 1183.47 | 35 | 385 |
| 11 | 118.35 | 1140.62 | 35 | 385 |

각 GT ZIP에는 ISO별 7개 파일이 있고, noisy ZIP에는 ISO별 77개 파일이 있다. noisy ZIP의 구성은 원본 clean 7개와 noisy 70개다. 전체 패키지에는 noisy 1,750개와 clean 계열 350개가 있다. 이는 **배포 패키지에서 관측한 파일 수**이며, 논문의 실제 평가에 모두 사용됐다는 판정은 아니다. GT ZIP의 `clean_and_slightly_denoised`와 noisy ZIP의 원본 `clean`을 혼동하지 않아야 한다.

## 검증과 기록

1. [공식 배포 안내](https://github.com/cao-cong/RViDeNet#dataset)에서 연결된 Google Drive의 [장면별 파일 ID](../benchmarks/sources.json)로 다운로드했다.
2. 각 ZIP의 CRC를 검사하고 크기·SHA-256·출처 ID를 receipt에 저장했다.
3. 다운로드 후 별도 `verify --dataset crvd --split test`로 ZIP 10개 전체를 다시 읽어 CRC·SHA-256·크기·출처 ID를 대조했다. **종료 코드 0**을 확인했다.
4. ZIP을 풀지 않고 내부 파일 목록을 읽어 장면·5개 ISO의 존재를 확인하고, 선택한 파일의 staging에 `.part` 잔여 파일이 없음을 확인했다.

공식 배포자 SHA-256은 확보하지 못했다. 기록한 SHA-256은 다운로드 당시의 로컬 지문과 이후 변경 탐지 근거이며, 독립적인 배포자 체크섬 검증이라고 부르지 않는다.

원격 기록:

- 파일별 receipt: `E:\datasets\receipts\crvd\<collection>\<ZIP 이름>.json`.
- 통합 inventory: `E:\datasets\receipts\crvd-test-inventory.json`.
- 실행 로그·상태: `E:\datasets\logs\crvd-test-1795156377702.*`.
- 프로젝트에도 [통합 inventory 사본](CRVD_test_download_inventory.json)을 보관했다. 데이터 본문은 원격 E 드라이브에만 저장했다.

## 후속 압축 해제 완료

- 최종 루트: `E:\datasets\crvd`.
- GT: `E:\datasets\crvd\indoor_raw_gt\scene7`부터 `scene11`까지, 총 175개 파일.
- noisy: `E:\datasets\crvd\indoor_raw_noisy\scene7`부터 `scene11`까지, 총 1,925개 파일(noisy 1,750개 + 원본 clean 175개).
- 각 장면에서 ISO 1600, 3200, 6400, 12800, 25600을 모두 확인했다.
- 해제 파일 합계: **2,100개 / 9,435,138,300 bytes / 8.79 GiB**.
- 원본 ZIP 10개는 `E:\datasets\archives\crvd`에 보존했다.

[해제 스크립트](../scripts/extract_crvd_test.py)의 종료 코드 0을 확인했다. ZIP별 로컬 SHA-256을 다운로드 기록과 비교한 뒤, 별도 staging에 저장하면서 ZIP CRC를 검사했다. 저장한 각 파일을 다시 읽어 크기·CRC·SHA-256을 대조하고 전체 파일 목록을 확인한 후 최종 폴더로 이동했다. 이동 후에도 장면·ISO별 파일 수와 전체 크기·파일 목록을 다시 확인했다.

파일별 해시·크기·출처 ZIP은 원격 `E:\datasets\receipts\crvd-test-extraction.json` 및
[프로젝트 검증 목록 사본](CRVD_test_extraction_inventory.json)에 기록했다. 실행 로그 경로도 이 목록에 포함했다.
파일의 TIFF 디코딩이나 RAW 값 범위·프레임 대응·논문 평가 프로토콜까지 검증한 것은 아니다.

소요 시간이나 이후 학습 시간을 추정하지 않았다. 데이터 로더·전처리·평가 프로토콜 확정은 후속 범위다.
