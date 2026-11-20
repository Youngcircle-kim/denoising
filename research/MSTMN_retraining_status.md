# MSTMN 비공식 재학습 실행 기록

확인 시점: 2026-11-20 18:46:38 KST. 이 문서는 진행 상태의 스냅샷이며 실시간 상태는 서버의 status.json을 확인한다.

**본 학습 진행 중:** 4,758 / 120,000 optimizer updates.
최근 저장 checkpoint는 4,700회다. 학습 완료·논문 성능 재현 완료가 아니다.

- 공식 코드·가중치는 [재조사](MSTMN_official_artifact_recheck.md)에서도 확보하지 못해 논문 기반 비공식 구현을 사용한다.
- [CRVD 학습 scene 1–6](CRVD_train_download_status.md) ZIP12개와 해제 파일2520개를 검증했다.
- Windows GPU 환경의 전체 테스트78개와 최종 checkpoint 메타데이터 보강 후 엔진 테스트6개가 통과했다.
- 논문 조건(batch16, sequence25, packed128, FP32)의 [GPU 수용량 검사](MSTMN_gpu_capacity.json)가 통과했다.
- 실제 데이터 초기3회 학습 후 같은 optimizer·RNG·표본 위치에서 목표120000회까지 재개했다.
- SSH를 종료하고 다시 연결한 뒤에도 update 증가와 정상 상태를 확인했다.
- [100회 checkpoint 검증](MSTMN_checkpoint_validation.json)에서 안전 로드·optimizer·소스 hash 일치를 확인하고 별도 사본을 보관했다.
- [초기 checkpoint 추론 진단](MSTMN_initial_checkpoint_diagnostic.json)은 scene7/ISO1600/noisy0의7프레임만 사용했다.
  이 값은 초기 가중치의 연결 점검이며 전체 시험 평균이나 최종 모델 성능이 아니다. 진단 값으로 설정을 변경하지 않았다.

## 실제 실행 위치

- 프로젝트: `C:\Users\HP\yhkim\denoising`.
- 학습 환경: `.venv-train`, Python3.13.17/PyTorch2.10.0+cu130.
- 실행 폴더: `E:\denoising_runs\mstmn-crvd-20261120`.
- 실시간 진행: 실행 폴더의 `status.json`과 `train.jsonl`.
- 최신 가중치: 실행 폴더의 `last.pth`, 100회 간격과 정상 종료/STOP 때 원자적으로 교체.
- 초기 추론용 보관 사본: `checkpoint_diagnostic.pth` (100회 가중치).
- 독립 프로세스 로그: `E:\denoising_runs\launches\mstmn-crvd-20261120-b599edf1626a.*`.

[진행 상태 JSON](MSTMN_retraining_status_snapshot.json)은 실제 PID, update 수, 최근 loss, 실측 step시간·peakVRAM을 포함한다.
최근 학습 loss는 서로 다른 무작위 표본의 L1이며 시험 PSNR/SSIM과 구분한다.
초기 일부 step의 측정만으로 총 학습 시간을 단정하지 않는다.

## 남은 작업과 조건

1. 120000 updates까지 학습을 마치고 최종 checkpoint의 상태·손실·유한값을 확인한다.
2. 고정된 CRVD 시험 manifest 전체1750쌍을 평가하고 장면·ISO·반복별 지표와 집계 조건을 보관한다.
3. Table III RAW42.92dB/0.9838과 차이를 보고하되, 공식 코드·학습/평가 세부가 미확인인 점을 함께 명시한다.

모델의 메모리 수식 차원 해석, patch 크기, 25프레임 구성, 잡음 sampling, 초기 상태, flow 파일·변환 등은
[구현 명세](MSTMN_implementation_spec.md)와 [재학습 안내](../docs/retraining.md)의 비공식 선택이다.
실제 TIFF 내용이나 패키지를 매번 hash 재검증하는 resume은 아니며, source·설정·flow 일치 검증과
별도 데이터 검증 기록을 구분한다. 중지·재개 명령도 재학습 안내에 있다.
