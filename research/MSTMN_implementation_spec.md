# MSTMN 비공식 구현 명세와 논문에서 확정되지 않은 선택

검토일: 2026-11-20. 원문 PDF의 p.3-8(인쇄면 5812-5817) 전체와 Fig.2-5를 렌더링해 시각적으로 확인했다. 광류 인용 [63]은 p.13(5822) 참고문헌까지 확인했다. 이 문서는 원문에서 옮긴 사실 `[P]`, 원문만으로 확정할 수 없는 사항 `[U]`, 실행 가능한 구현을 위해 택한 정책 `[R]`를 구분한다. `[R]`은 저자 구현을 확인한 값이 아니며, 이 코드로 재학습한 결과를 공식 MSTMN 체크포인트 결과라고 표기하지 않는다.

## 입력, 피라미드, 처리 방향

- `[P]` 입력은 현재 noisy `y_t`, 직전 복원 `xhat_(t-1)`, 이전 noisy `y_(t-1)..y_(t-T)`. Memory의 이전 프레임 수 `T=3`은 IV-B.2, p.8/5817에 명시돼 있다.
- `[P]` RAW를 RGBG 순서의 4채널로 pack한다(III-D, p.6/5815). 원본 RAW가 `[B,1,H,W]`이면 packed 입력은 `[B,4,H/2,W/2]`.
- `[P]` Gaussian은 `G_1..G_4`, Laplacian은 `L_1..L_3`이다. **숫자가 큰 scale일수록 해상도가 작다.** 따라서 신경망의 계산 순서는 `l=3 -> 2 -> 1`이다(III-B, p.4/5813, Fig.2).
- `[P]` Eq.(11): `G_l = Sample(G_(l-1) * kappa_5x5)`. 5x5 Gaussian 필터 후 공간 크기를 절반으로 줄인다.
- `[P]` Eq.(12): `L_l = G_l - (Upsample(G_(l+1)) * kappa_5x5)`. Gaussian의 인접 scale 차이가 Laplacian이다.
- `[P]` III-D(p.6/5815)는 pack, pyramid decomposition, 추가 pixel-unshuffle 순서로 설명한다. 각 처리 scale에서 pixel-unshuffle(2)로 4채널을 16채널로 바꾼다. 복원 후 pixel-shuffle(2)와 pyramid collapse를 거쳐 packed RAW로 돌아온다(Fig.4, p.6; p.7 첫 문단).
- `[P]` Gaussian 저주파 성분은 원본 것을 사용하고, 복원한 Laplacian 성분과 합성한다(III-D, p.7/5816). `G_4`를 별도의 CNN으로 denoise한다고 쓰여 있지 않다.
- `[U]` 정확한 Gaussian 계수/sigma, 필터 padding, downsample 시작 위치, upsample 종류, 홀수 해상도 처리, warp와 pyramid 연산의 정확한 순서가 없다. Fig.4의 일부 Gaussian 라벨은 `G_1..G_3`만 보여 주지만 Fig.2와 Eq.(12)에는 `G_4`가 필요하다.
- `[R]` 5-tap binomial `[1,4,6,4,1]/16`의 outer product, reflect padding, 필터 후 `[...,::2,::2]`, bilinear `align_corners=False` upsample 후 같은 필터를 사용한다. Decomposition과 collapse에서 같은 expand 연산을 써 algebraic round trip을 보존한다.
- `[R]` 입력 packed 크기를 16배수로 replicate-pad하고 출력에서 제거한다. 4->16 pixel-unshuffle 전의 pyramid를 사용한다. 현재 입력이 4채널이라고 다시 Bayer pack하지 않는다.

packed 입력 크기를 `Hp x Wp`라 할 때(나눗셈은 pad 후 기준), 처리 shape는 다음과 같다.

| 성분 | pyramid의 4채널 해상도 | pixel-unshuffle 후 16채널 해상도 |
|---|---|---|
| G1, L1 | Hp x Wp | Hp/2 x Wp/2 |
| G2, L2 | Hp/2 x Wp/2 | Hp/4 x Wp/4 |
| G3, L3 | Hp/4 x Wp/4 | Hp/8 x Wp/8 |
| G4 | Hp/8 x Wp/8 | CNN 처리 없음; 합성의 base |

## Memory enhancement: Eq.(14)-(15), Fig.3

- `[P]` Gaussian 성분이 key, Laplacian 성분이 value이다. 현재 프레임이 query, 이전 T개의 **noisy** 프레임이 memory이다. 이전 복원 프레임을 memory value로 바꾸지 않는다(III-C, p.5/5814).
- `[P]` 원문의 Eq.(14)는 `vR = phi^-1((tau phi vM)^T @ sigmoid((phi kQ)^T @ (tau phi kM)))`로 인쇄돼 있다. `phi`는 patch로 나누는 fold, `phi^-1`은 unfold, `tau`는 memory의 프레임 수와 채널을 한 축으로 모으는 reshape라고 설명한다.
- `[P]` 정규화 함수는 **sigmoid**라고 명시한다. softmax, `1/sqrt(D)`, cosine/L2 key normalization, top-k 선택, attention row-sum normalization은 원문에 없다.
- `[P]` Eq.(15): `enhancedL = Conv(concat(queryL, retrievedL))`. Query와 retrieved value를 channel concat해 convolution에 보낸다.
- `[P]` 전체 retrieval은 작은 해상도의 `l=2,3`에서만 수행한다. 가장 큰 해상도 `l=1`에서는 `retrievedL_1 = U(retrievedL_2)`를 받아 Eq.(15)만 수행한다(III-C 뒤 문단, p.6/5815).
- `[U]` patch 크기/stride/overlap, `phi`의 정확한 tensor layout, `tau`의 축 순서, convolution kernel/activation, channel projection, 원문 Eq.(14)의 transpose와 곱 순서에 맞는 구체적 shape가 없다. 인쇄된 식을 보편적인 row/column 규칙에 대입하면 외곽 곱 차원이 바로 맞지 않는다. 따라서 단순히 transpose를 수정하고도 저자와 같다고 단정할 수 없다.
- `[R]` 실행 모델은 `Q:[B,N,D]`, `K,V:[B,N,T*D]`, `D=16*patch_size^2`로 둔다. 원문의 `tau` 설명대로 memory의 프레임 T와 patch-feature channel D를 합친다. `A=sigmoid(Q.transpose(-1,-2) @ K):[B,D,T*D]`, `R=(A @ V.transpose(-1,-2)).transpose(-1,-2):[B,N,D]` 후 inverse fold한다. 이는 원문 Eq.(14)의 **바깥 matrix product 순서를 교환한 비공식 차원 수정**이며, 저자 구현과 일치한다고 검증되지 않았다. T와 공간 patch 수 N을 합치는 일반 spatial-token attention은 사용하지 않는다.
- `[R]` 기본 patch/stride는 `4/4`. `stride <= patch_size`로 제한하고 마지막 patch가 경계를 덮도록 오른쪽/아래에 replicate-pad한다. Overlap이 있는 설정에서는 inverse-fold 결과를 coverage divisor로 나누어 중복 픽셀을 평균한다. 이 divisor는 attention 정규화가 아니다.
- `[R]` 전체 memory feature를 모두 쓰고 query-feature 축 D만 128개씩 나누어 계산한다. query chunk는 근사/top-k가 아니며 같은 attention을 계산한다. 추론 시 affinity의 일시 할당을 줄이지만 학습에서 저장되는 모든 backward activation까지 상수 메모리로 만드는 것은 아니다. Affinity 크기는 D x (T*D)로 공간 patch 수 N과 독립적이다.
- `[U]` Gaussian key는 양수이므로 sigmoid dot product가 포화될 수 있다. 현재 선택에서도 dot product의 축 N이 해상도에 따라 증가한다. 다만 retrieved value의 합산 축 T*D는 고정되어, 공간-token 해석의 T*N 합산 길이 증가를 추가하지 않는다. 원문은 이 포화/크기 의존성을 처리하는 scaling을 명시하지 않았으므로 임의로 L2 normalization이나 averaging을 넣지 않는다. 대규모 학습 전 patch/full-frame 출력 분포와 gradient를 실측해야 한다.
- `[R]` concat convolution은 3x3, 32->16, bias 사용, 별도 활성화 없음. Memory의 미확인 선택은 checkpoint 설정에도 저장한다.

## Noise variance / denoiser

- `[P]` Eq.(17), p.7/5816: `n ~ N(0, sigma_s*G + sigma_r^2)`. Denoiser에 넣는 noise map은 `sigma_s*G + sigma_r^2`다. 원문은 이를 `sigma_t,l`로 부르지만 식은 **분산**이다. sqrt를 취한다고 쓰지 않았다.
- `[P]` Denoiser 입력은 `concat(enhancedL, noise_map)`이다. 3개의 3x3 convolution, ReLU, 각 layer 출력 16채널이라고 서술한다(III-D.1, p.7/5816).
- `[U]` Gaussian filtering/downsampling에 따른 noise variance 전파 방법, 마지막 convolution의 활성화를 예외로 하는지 여부가 없다. Laplacian은 음수가 가능하므로 마지막 ReLU 여부는 결과에 큰 영향을 줄 수 있다.
- `[R]` 모델 API의 `shot`은 정규화 RAW의 `a`, `read`는 이미 분산인 `b`다. 즉 raw noise table의 `(a_raw,b_raw)`와 RAW range `R`에 대해 `shot=a_raw/R`, `read=b_raw/R^2`. API에서 `read`를 다시 제곱하지 않는다.
- `[R]` 각 Gaussian level에서 `clamp_min(shot*G + read,0)`을 만든 뒤 pixel-unshuffle한다. 계수의 추가 scale 보정은 하지 않는다. 이는 원문 식을 실행 가능하게 옮긴 선택이며 실제 filtering 후 분산과 동일하다고 주장하지 않는다.
- `[R]` 3개 convolution 모두 뒤에 ReLU를 둔다(문자 그대로의 해석). Negative Laplacian을 허용하는 final-linear 대안은 별도 실험으로 구분해야 한다.

## W/V predictor와 fusion

- `[P]` Noise weight `W_l` 입력: denoiser 출력 `uL_l`, enhanced noisy `enhancedL_l`, `l<3`일 때 상위 scale에서 upsample한 `W_(l+1)`(p.7/5816, III-D.1, Fig.5).
- `[P]` Alignment weight `V_l` 입력: 현재 Gaussian `G_l`, aligned previous Gaussian `Gprev_l`, `l<3`일 때 upsample한 `V_(l+1)`(III-D.2).
- `[P]` W와 V predictor는 같은 구조: `1x1 -> depthwise-separable 3x3 -> depthwise-separable 3x3 -> 1x1`. 처음 3개 stage는 16채널/ReLU, 마지막은 1채널/sigmoid. 큰 대각행렬을 만들지 않고 1채널 spatial map을 16채널에 broadcast한다.
- `[U]` 두 depthwise-separable stage 내부의 depthwise/pointwise 사이 activation, bias, scale 사이 weight sharing은 명시하지 않았다.
- `[R]` 각 scale에 별도 denoiser/W/V/memory-concat 파라미터를 둔다. Depthwise 3x3(groups=16) 후 pointwise 1x1, stage 끝 ReLU를 쓰고 모든 convolution에 bias를 둔다. 상위 W/V는 bilinear `align_corners=False`로 키운다.
- `[P]` Eq.(16c)의 fusion은 정확히 다음이다: `xhatL = W*enhancedL + V*warpedPrevL + (1-W-V)*uL`. W/V의 역할을 뒤집거나 `(1-W)*(1-V)`로 바꾸지 않는다.
- `[P]` W/V를 각각 sigmoid로 추정한다고 설명하므로 합이 1 이하라는 제약은 없다. 원문의 식을 그대로 따르면 `1-W-V`는 음수가 될 수 있다.
- `[R]` W/V의 재정규화, convex-weight 강제, fusion 후 clipping을 하지 않는다. 음수 denoiser 계수를 숨기지 않는다.

## Flow와 recurrent state

- `[P]` Flow [63]은 Ranjan & Black, *Optical flow estimation using a spatial pyramid network*, CVPR 2017, DOI 10.1109/CVPR.2017.291이다(p.13/5822). 즉 SPyNet을 지칭한다.
- `[P]` 현재 Gaussian과 이전 복원 Gaussian 사이의 flow를 가장 작은 처리 scale에서 계산하고, 두 큰 scale로 resize하여 이전 복원을 warp한다(III-D.2, p.7/5816).
- `[U]` SPyNet 구현/checkpoint/pretraining/freeze 여부, packed/16채널 입력의 3채널 변환, flow 방향, coordinate convention, warp interpolation/padding이 없다.
- `[R]` Model은 외부 `flow_model`을 반드시 주입받는다. 기본값으로 silent identity/zero flow를 사용하지 않는다. Coarsest 16채널 G3 두 개를 넣으면 **current 좌표에서 previous를 읽는 backward-sampling flow** `[B,2,h,w]`를 반환해야 한다.
- `[R]` Flow resize는 bilinear `align_corners=False`, x/y 성분을 해당 width/height 배율로 함께 조정한다. Warp는 bilinear `grid_sample`, `padding_mode='border'`, `align_corners=True`로 pixel-center 좌표를 명시한다. Previous G/L의 16채널 grid에서 같은 flow를 적용한다.
- `[R]` 첫 previous와 모자란 history는 첫 noisy 프레임 복제로 초기화한다. 학습에서는 sequence 경계마다 reset, sequence 내 이전 복원에 gradient를 유지하고 학습 구간 경계에서 detach한다. 이는 원문에서 확정되지 않은 정책이다.

## 학습과 평가: 보존할 사실 / 아직 필요한 선택

- `[P]` 학습 scene1-6, 시험 scene7-11(IV-A, p.8/5817). 시험 scene을 optimizer update나 checkpoint 선택용 loss에 섞지 않는다.
- `[P]` Bayer pack 후 128x128 crop, sequence length25, batch16. L1 복원 loss Eq.(18), Adam beta=(0.9,0.999), eps=1e-8, 초기 lr=1e-3, iteration15000부터5e-4,40000부터1e-4, 총120000 iteration. 단일 Titan XP에서 학습했다고 서술한다.
- `[U]` CRVD 각 장면은 저장된 동적 clean frame7개와 noisy realization10개를 포함하므로 sequence25를 실제로 구성하는 frame 순서, wrap/reverse/random-start 정책이 필요하다. 원문에 실측 noisy와 합성 noise의 정확한 학습 혼합 비율도 없다.
- `[U]` L1의 reduction/시간 축 합산, BPTT detach 정책, augmentation, random seed, gradient clipping, weight decay, checkpoint 주기/선택, AMP는 없다.
- `[R]` 모델 구조 코드와 실제 run 설정을 함께 checkpoint에 저장하고, 선택한 25프레임 구성 및 noise 생성 정책은 별도 학습 설정에 기입한다. 논문에 없는 정책을 논문값으로 설명하지 않는다.
- `[P]` 목표 Table III RAW의 MSTMN 행은 PSNR42.92/SSIM0.9838. 이는 비교 참조값이며 공식 프로토콜과 checkpoint가 없는 상태의 성공 판정 기준이 아니다.
- `[U]` 데이터 정규화, clipping/quantization, SSIM 세부, border, noisy realization 반복, frame/sequence/scene/ISO 집계 방식, warm-up 제외 여부가 확인되지 않았다.

## 완료 조건

1. Pyramid decomposition/collapse의 수치 round-trip, pixel-shuffle shape, 각 weight map과 fusion 수식에 대한 합성 검증이 통과한다.
2. Query chunk 크기를 바꾸어도 memory 출력이 오차 범위 내에서 동일하고, overlap inverse fold에 coverage가 0인 픽셀이 없다.
3. 실제 GPU에서 packed128 patch, T3 history의 forward/backward와 유한한 gradient를 확인한다. 학습 시간/메모리는 이 실측을 바탕으로만 기록한다.
4. 학습 scene manifest가1-6만 포함하며, 시험 scene7-11은 별도 평가 단계에서만 읽는다.
5. 모델 설정, 선택 정책, 원문과 다른 점, source/noise coefficients, environment, git commit을 checkpoint 및 결과 metadata에 보존한다.
6. 최종 표에는 비공식 재구현 여부와 실제 평가 조건을 표시하고, 공식 MSTMN과 정확히 같은 구현/평가라고 단정하지 않는다.
