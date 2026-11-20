# CRVD 학습 장면 확보 완료

확인일: 2026-11-20 (Asia/Seoul). 재학습 요청에 따라 scene 1–6을 실제로 확보했다.

- 공식 CRVD 배포의 GT·noisy **ZIP 12개**, 총 **8,059,359,142 bytes / 7.51 GiB**.
- 압축 해제 파일 **2,520개**, 총 **11,322,165,960 bytes / 10.54 GiB**.
- GT 210개, noisy collection 2,310개(noisy 2,100개 + 원본 clean 210개).
- scene 1–6 각각 ISO 1600, 3200, 6400, 12800, 25600 전부 포함.
- 최종 경로: `E:\datasets\crvd\indoor_raw_gt\scene1`–`scene6`, `indoor_raw_noisy\scene1`–`scene6`.
- 기존 시험 scene 7–11과 원본 ZIP은 보존했다.

다운로드와 압축 해제의 종료 코드 0을 각각 확인했다. 각 ZIP의 출처 ID·크기·로컬 SHA-256을
download receipt와 대조했고, 해제 중 ZIP CRC와 저장 파일의 재독출 크기·CRC·SHA-256을 검증했다.
장면·ISO·프레임의 정확한 파일 목록을 확인한 뒤 학습 폴더를 공개했다.
파일별 출처·크기·해시는 [검증 목록](CRVD_train_extraction_inventory.json)에 있다.
이는 공식 배포자 체크섬 검증과 구분하며, 로컬 해시를 이후 변경 탐지 근거로 보존한다.

실행 기록은 원격 `E:\datasets\logs\crvd-train-1795161114295.*` 및
`E:\datasets\logs\crvd-train-extraction.*`에 있다. 다운로드 상태 파일에는 실제 subprocess 종료 코드를 기록했다.

다운로드 첫 시도에서는 PowerShell이 gdown의 stderr 진행 메시지를 오류로 처리해 중단했다.
진행 메시지를 별도 stderr 로그로 보내도록 실행 래퍼를 수정해 부분 파일에서 재개했고,
이후 전체 12개 ZIP과 추출 파일을 검증했다. 이 실패를 데이터 전송 실패나 최종 성공으로 혼동하지 않는다.

학습 입력은 `scene1–6`의 GT 기반 합성 잡음으로 시작한다. 실제 noisy 자료도 확보했으며 별도 실측 학습 모드에서
사용할 수 있다. 시험 장면은 학습에 포함하지 않는다. 학습 조건·가정은 [재학습 안내](../docs/retraining.md)에 있다.
