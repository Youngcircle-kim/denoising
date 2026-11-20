# MSTMN 공식 코드·가중치 재확인

확인일: 2026-11-20 UTC. 대상: *Multi-Scale Spatio-Temporal Memory Network for Lightweight Video Denoising*, TIP 2024, DOI [10.1109/TIP.2024.3444315](https://doi.org/10.1109/TIP.2024.3444315).

현재 확보 상태는 **공식 MSTMN 소스 코드와 사전학습 가중치를 확인·확보하지 못함**이다. 전 세계에 공개본이 없거나 저자가 배포하지 않는다는 판정은 아니다. 확인되지 않은 모델을 공식 구현으로 대체하지 않는다.

## 확인된 근거

| 출처 | 실제 확인 결과 | 해석 |
| --- | --- | --- |
| [Dong 저자 새 공식 홈페이지](https://faculty.xidian.edu.cn/DWS/en/index.htm) | 공개 HTML의 논문 목록 [13]에 MSTMN이 있다. 해당 문단에서 `href`는 Paper 하나뿐이고, `project & code`는 링크가 없는 일반 텍스트이다. Paper의 대상도 저자 PC의 `file:///D:/Research/My_homepage/wsdong_Website/Papers/Journal/TIP2024-MSTMN.pdf`이다. | 검색 결과에 나타나는 “project & code”라는 문구만으로 공개 프로젝트·코드 URL 존재를 입증할 수 없다. 로컬 `file:` 대상을 접근하거나 내려받지 않았다. |
| [Dong 저자 기존 공식 홈페이지](https://web.xidian.edu.cn/wsdong/) | Exa의 페이지 추출에서는 논문과 소개 문구를 읽을 수 있다. 직접 HTTPS 및 HTTP HTML 요청은 연결 시간 초과. | 기존 홈페이지의 실제 MSTMN 링크 대상은 이번에도 검증하지 못했다. |
| [기존 faculty 경로](https://see.xidian.edu.cn/faculty/wsdong/) | Exa 직접 가져오기에서 `CRAWL_NOT_FOUND`. | 해당 루트 경로 접근 실패를 개별 PDF·프로젝트 전체의 비공개 판정으로 확대하지 않는다. |
| [Wu 저자 공식 논문 항목](https://faculty.xidian.edu.cn/WUFANGFANG/en/lwcg/437309/content/274213.htm) | 제목·학술지·권·쪽 정보가 있는 논문 항목. 읽은 내용에서 코드·체크포인트 배포 경로를 찾지 못했다. | 논문 존재 근거이며, 가중치 확보 근거가 아니다. |
| [sunlustar 공개 저장소 API](https://api.github.com/users/sunlustar/repos?per_page=100) 및 [프로필](https://github.com/sunlustar?tab=repositories) | 읽은 공개 목록은 `sunlustar/MAP-VDNet` 1개. | 해당 계정에서 MSTMN 저장소를 찾지 못했다. 다른 계정·비공개 저장소의 존재 여부는 알 수 없다. |
| [MAP-VDNet 공식 README](https://raw.githubusercontent.com/sunlustar/MAP-VDNet/master/README.md) | IJCV 2021의 *Deep Maximum a Posterior Estimator for Video Denoising* 구현이라고 명시. Vimeo-90K 학습, 별도 모델 Drive 폴더 등이 연결되어 있다. | MSTMN TIP 2024와 다른 논문이므로 코드와 가중치 링크를 MSTMN에 사용하지 않는다. |
| [GitHub 저장소 이름 검색](https://api.github.com/search/repositories?q=MSTMN&per_page=100) | `total_count=2`: `ts-klassen/mstmnte`, `perseus-99/mstmnit`. 이름·설명·README 검색을 포함하면 프로필 설정 저장소 `boizsitink/boizsitink`까지 3개. | 전자는 Erlang OTP 라이브러리, 두 번째는 대학 스포츠 행사 웹사이트, 세 번째는 프로필 설정이다. 공식 MSTMN 구현 후보로 채택하지 않았다. |

GitHub의 두 후속 제목 검색 요청은 공용 비인증 API의 rate limit 응답을 받았다. 실패한 요청을 검색 결과 0건으로 취급하지 않았다. 인증이나 접근 제한을 우회하지 않았다.

## 검색 범위와 계수

Exa Search 스킬의 지침을 읽고, 논문 제목, 저자·저장소, 프로젝트·코드·체크포인트, 저자 홈페이지, 별도 가중치 배포의 관점에서 8개의 성공 검색을 수행했다. 검색 결과는 발견용이며, 위 판정은 저자 홈페이지 HTML·공식 기관 페이지·GitHub 공개 메타데이터·README의 직접 확인을 우선한다. 다른 논문과 동명 약어는 제외했다.

- `sources_reviewed: 78`: 스킬이 정의한 모든 Exa 검색 호출의 `numResults` 합계. 초기 스키마 오류 3회 × 8개 요청도 재시도로 포함한 요청량 계수다.
- 성공 검색의 요청량 합계: 54.
- 실제 반환된 결과 항목: 49, URL 기준 중복 제거 후 28개.
- 위 수치는 78개 또는 28개의 독립 전문을 모두 읽었다는 뜻이 아니다. 직접 페이지 가져오기와 공개 API 요청은 이 검색 요청량 계수에 더하지 않았다.

| 검색 관점 | numResults |
| --- | ---: |
| 정확한 논문명으로 공식 코드·가중치 탐색 | 8 |
| Lu Sun·Fangfang Wu·Weisheng Dong와 GitHub 연결 | 8 |
| MSTMN 프로젝트·체크포인트·Drive·Baidu | 8 |
| 저자 공식 홈페이지의 Paper/project/code 항목 | 6 |
| GitHub MSTMN 코드·가중치 후보 | 6 |
| 기존 Xidian Projects 경로 | 6 |
| 사전학습 가중치·체크포인트 직접 탐색 | 6 |
| 저자·MSTMN·Drive·Baidu 직접 배포 탐색 | 6 |

검색 중 타 모델의 Google Drive·Baidu 링크는 나왔지만 MSTMN과 출처 대응이 성립하지 않아 채택하지 않았다. 확인된 공식 MSTMN checkpoint URL, 코드 commit, 해당 모델의 사용 조건은 모두 미확인 상태다. 대형 모델 파일은 다운로드하지 않았고 저자에게 메시지를 보내지 않았다.

## 후속 작업에 대한 판단

공식 코드·가중치를 현재 확보하지 못했으므로, 현 시점에서 진행하는 구현·학습은 **논문을 근거로 한 자체 재구현·재학습**으로 기록한다. 공식 MSTMN 가중치를 평가한 재현 결과라고 표시하지 않는다. 논문에 명시된 CRVD 학습 scene 1–6 및 주요 학습 설정을 우선 따르되, 잡음 합성의 정확한 적용 범위, 25프레임 구성, 초기 상태·패딩·광학 흐름 등 미확인 항목은 가정을 명시해 분리한다. 뒤에 공식 아티팩트를 확보하면 출처·변형·checkpoint와 구현 차이를 확인한 뒤 별도로 비교한다.
