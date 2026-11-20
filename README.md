# denoising

MSTMN 비디오 잡음 제거 논문의 재현 조사와 실험 준비 저장소입니다.

- [재현 조사·계획 보고서](research/MSTMN_stage1_reproducibility.md)
- [Benchmark 다운로드 안내](benchmarks/README.md)
- [공식 배포 파일 목록](benchmarks/sources.json)
- [CRVD 다운로드·압축 해제 완료 기록](research/CRVD_test_download_status.md)
- [공식 모델 연결용 RAW 평가 코드 사용법](docs/evaluation.md)
- [평가 코드의 실제 데이터 검증 기록](research/MSTMN_eval_validation.json)
- [논문 기반 비공식 MSTMN 재학습](docs/retraining.md)
- [CRVD 학습 scene 1–6 확보 기록](research/CRVD_train_download_status.md)
- [재학습 실행 상태·checkpoint·남은 작업](research/MSTMN_retraining_status.md)

원격 작업 경로: `C:\Users\HP\yhkim\denoising`.

다운로드 전용 가상환경과 **CRVD 시험 장면 7–11의 GT·noisy ZIP 10개**를
원격 서버에 준비하고 무결성 검사를 완료했습니다. 압축 해제한 2,100개 파일(8.79 GiB)은
`E:\datasets\crvd`에 있으며, 원본 ZIP은 보존했습니다.
다운로드 스크립트의 기본 실행은 계속 **계획만 출력**합니다.
후속 요청으로 추가한 **논문 기반 비공식 모델·재학습 코드**는 [재학습 안내](docs/retraining.md)를 참조하세요.
공식 모델·가중치를 확보한 것으로 해석하지 않습니다.

Windows에서 Python 설치 없이 다운로드 대상을 확인할 수 있습니다.

```powershell
cd C:\Users\HP\yhkim\denoising
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_benchmarks.ps1 -Action Plan -DataRoot E:\datasets
```

CRVD 평가 도구는 `.venv-eval`에서 실행합니다. 공식 MSTMN 가중치는 아직 확보하지 못했습니다.
아래 명령은 실제 데이터 검사와 **입력 복사 진단**이며 MSTMN 성능 평가가 아닙니다.

```powershell
.\.venv-eval\Scripts\python.exe -m denoising_eval inspect --data-root E:\datasets\crvd --output results\inspect-01
.\.venv-eval\Scripts\python.exe -m denoising_eval evaluate --data-root E:\datasets\crvd --output results\identity-01 --baseline identity --max-sequences 1
```

검증:

```bash
python -m unittest discover -s tests -v
```
