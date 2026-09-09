# MonoDETR — Pseudo-Depth Supervision Variants (E0 / E1 / E2)

이 저장소는 원본 [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR)을 기반으로,
foreground pseudo-depth map에 대한 supervision 방식을 바꿔가며 비교할 수 있도록
확장한 것이다. 원본의 detection/transformer/matcher/evaluator 핵심 로직은 전혀
건드리지 않았고, depth map supervision 경로에만 새 옵션을 추가했다.

환경 이식을 위한 순수 호환성 패치(numba/CUDA 관련)는 이 문서에서 다루지 않는다.

## 실험 변형 3종

`configs/monodetr.yaml`의 `model.pseudo_depth_mode`로 선택한다.

| mode | 설명 |
|---|---|
| `center` (기본값) | 원본 MonoDETR과 동일한 hard center-depth target. 변경 없음(E0 baseline). |
| `soft_ncf` | Near/Center/Far 3개 pseudo-depth를 가중합한 soft target (E1). |
| `center_extent` | center target + depth-extent(near~far 폭) 보조 회귀 (E2). |

`soft_ncf`의 near/center/far 가중치는 `model.soft_depth_weights`(기본 0.2 / 0.6 / 0.2,
합은 1이어야 함)로 설정한다. `center_extent`의 extent loss 가중치는
`model.lambda_extent`(기본 1.0)로 설정한다.

## 파일별 변경 내용

각 항목에 어느 실험 변형 때문에 생긴 변경인지 표시한다 — **[공통]**은 E0
포함 세 모드 모두에 영향을 주는 배선/공용 코드, **[E1]**은 `soft_ncf`
전용, **[E2]**는 `center_extent` 전용이다. E0(`center`)은 새 코드를 추가로
타지 않고 원본 경로를 그대로 쓴다.

### `configs/monodetr.yaml`
- **[공통]** `pseudo_depth_mode` 추가 — 세 모드를 고르는 스위치.
- **[E1]** `soft_depth_weights`(near/center/far, 기본 0.2/0.6/0.2) 추가 —
  `soft_ncf`에서만 읽는다.
- **[E2]** `lambda_extent`(기본 1.0) 추가 — `center_extent`의 extent loss
  가중치. `center_extent`가 아니면 읽히지 않는다.

### `lib/datasets/kitti/kitti_dataset.py`
- **[E1]** 각 GT 3D box의 8개 corner(camera 좌표계)를
  `objects[i].generate_corners3d()`로 구해 z축(카메라 전방 거리) 최솟값/
  최댓값을 `depth_near`/`depth_far`로 계산해 타겟 dict에 추가했다. 이
  두 값은 `soft_ncf`의 near/far target으로 쓰인다. 기존 `depth`(3D box
  center depth)는 그대로 둔다.
- **[E2]** 위 `depth_near`/`depth_far`로부터
  `depth_extent = max(depth_far - depth_near, 0)`을 만들어 타겟 dict에
  추가했다 — `center_extent`의 extent 회귀 target으로만 쓰인다.
- **[공통]** `depth_near ≤ depth ≤ depth_far`가 깨지는 비정상 라벨은
  경고를 남기고(최대 10회) 카운트만 한다(값을 고치거나 건너뛰지 않음) —
  E1/E2 어느 쪽을 켜든 동일하게 적용되는 데이터 품질 체크다.
- **[공통, E1/E2와 무관]** `train_screening_20`처럼 `train_`으로 시작하는
  이름의 subset split도 `train`과 동일하게(같은 augmentation, 같은
  디렉터리 구조로) 다룰 수 있도록 split 검증 조건을 넓혔다 — 이건
  screening/stage2 subset 실험 인프라를 위한 것으로 E1/E2 자체와는
  무관하다.

### `lib/models/monodetr/depth_predictor/depth_predictor.py`
- **[공통]** `pseudo_depth_mode`를 저장해두고, forward가 반환하는 튜플에
  `depth_extent`(E0/E1에서는 `None`)를 하나 더 추가했다 — 호출부(`monodetr.py`)를
  세 모드 공통 시그니처로 맞추기 위한 배선.
- **[E2]** `center_extent`일 때만 depth-extent 회귀용 head
  (`extent_predictor`: conv → ReLU → conv → Softplus)를 새로 만든다. 그 외
  모드에서는 `extent_predictor = None`이고 아무 파라미터도 추가되지 않는다.
- **[E1에는 이 파일에 변경 없음]** — `soft_ncf`는 이 파일의 depth classifier/
  `weighted_depth` 계산식을 그대로 쓴다. **`weighted_depth`를 만드는 계산식과
  depth classifier 자체는 세 모드 모두 완전히 동일**하다 — E1/E2가 바꾸는
  것은 오직 학습 시 loss target이지, 추론 시 만들어지는 depth 표현이
  아니다.

### `lib/models/monodetr/depth_predictor/ddn_loss/ddn_loss.py`
- **[공통, 무변경]** 기존 `build_target_depth_from_3dcenter`(hard center
  target 생성)는 그대로 둔다 — E0와 E2의 center 항은 원본과 byte 단위로
  동일한 이 경로를 탄다.
- **[E1]** `build_soft_ncf_target`을 새로 추가했다: near/center/far
  각각을 기존과 동일한 LID(k+1 bin) 방식으로 bin index화한 뒤, 픽셀별로
  `near*0.2 + center*0.6 + far*0.2` 형태의 확률분포를 만든다. 세 값이
  같은 bin에 몰리면 `scatter_add_`가 자동으로 가중치를 합산한다(같은 bin
  충돌 시 가중치를 더하는 방식). 가중치 합이 1이 아니거나 음수면 즉시
  에러를 낸다. Background는 기존과 동일하게 마지막(“없음”) 클래스로
  채운다.
- **[E2]** `build_extent_target`/`extent_loss`를 새로 추가했다 — 픽셀
  영역에 scalar depth-extent를 칠하고 foreground 영역에서만 smooth-L1
  loss를 계산한다. E1은 이 두 함수를 전혀 호출하지 않는다.
- **[공통]** `forward()`에 `mode` 인자를 추가해 분기한다: `center`/
  `center_extent`는 기존 hard-target 경로 그대로(E2의 center 항도 여기서
  나온다), `soft_ncf`만 위 새 soft target 함수를 사용한다.

### `lib/models/monodetr/depth_predictor/ddn_loss/focalloss.py`
- **[E1 전용]** `soft_focal_loss`를 새로 추가했다 — 확률분포(soft) target에
  대한 multiclass focal loss로, 정확히 one-hot인 target을 넣으면 기존 hard
  focal loss와 수치적으로 동일한 값이 나오도록 만들었다(legacy one-hot
  구현의 epsilon 차이 정도만 남는다). `FocalLoss.forward`는 넘어온
  target의 dtype으로 자동 분기한다: `int64`(E0/E2가 쓰는 기존 hard label)
  → 기존 `focal_loss` 그대로, float(E1의 확률분포) → 새 `soft_focal_loss`.
  E2는 이 파일에서 새 코드를 전혀 타지 않는다(E2의 depth 분류 loss도
  hard label이라 기존 `focal_loss` 경로를 그대로 쓴다).

### `lib/models/monodetr/monodetr.py`
- **[공통]** `depth_predictor`가 5개 값(마지막에 `depth_extent`)을
  반환하도록 호출부를 맞췄다.
- **[공통]** `SetCriterion.__init__`에 `pseudo_depth_mode`를 새 파라미터로
  받는다. **[E1]** `soft_depth_weights`도 함께 받는다(E1이 아니면 계산에
  쓰이지 않음). **[E2]** `lambda_extent`도 함께 받는다(E2가 아니면 쓰이지
  않음).
- **[E1]** `loss_depth_map`에서 `depth_near`/`depth_far` target을 모아
  `ddn_loss`에 `mode`와 함께 넘긴다 — 이 두 값은 `soft_ncf`일 때만 실제로
  소비된다(`center`/`center_extent`는 무시).
- **[E2 전용]** `pseudo_depth_mode == 'center_extent'`일 때만
  `loss_depth_map_center` + `loss_depth_extent`(+
  `weighted_loss_depth_extent`) 세 loss 항을 추가로 계산한다. 그 외
  모드(E0, E1)는 기존과 동일하게 `loss_depth_map` 한 항만 만든다.
- **[E2 전용]** `build()`에서 `pseudo_depth_mode`에 따라 `weight_dict`의
  키를 분기한다(`center_extent`만 `loss_depth_map_center`+
  `weighted_loss_depth_extent`, 그 외엔 기존 `loss_depth_map`). criterion
  생성 시 `soft_depth_weights`([E1]용)와 `lambda_extent`([E2]용)를 config
  에서 읽어 전달한다.
- **[공통, 결과에 영향 없음]** 최종 object-level depth fusion(`pred_depth`
  계산식)은 **세 모드 모두 전혀 바뀌지 않았다.** 새로 추가된
  `pred_depth_map_sampled`(공통 배선)와 `pred_depth_extent_sampled`
  ([E2 전용]) 는 분석용으로만 붙인 출력이며 실제 detection 결과 계산에는
  관여하지 않는다.

### `lib/helpers/trainer_helper.py`
배치를 샘플 단위 dict로 변환할 때 넘기는 key 목록에 `depth_near`/
`depth_far`(**[E1]**용 near/far target 전달)와 `depth_extent`(**[E2]**용
extent target 전달)를 추가했다 — 셋 다 이 한 줄에서 함께 추가됐지만
실제로 쓰는 실험 변형은 서로 다르다.

### `lib/helpers/tester_helper.py`
**[공통, E1/E2와 무관]** 추론 직후 예측값을 들여다볼 수 있는
`analysis_callback` 훅(기본값 `None`, 지정하지 않으면 아무 동작도 하지
않음)을 추가했다. 특정 실험 변형에 종속되지 않는 범용 훅이며, 공식 평가
로직 자체는 그대로다.

## 새로 추가된 실험 인프라 (원본에는 없던 파일)

원본 MonoDETR 학습/추론 코드는 건드리지 않고, 위 supervision 변형들을
실제로 돌리고 비교하기 위한 코드를 별도로 추가했다.

- `tools/run_pseudo_depth_experiment.py` — E0/E1/E2 실험 launcher.
  `pseudo_depth_mode`만 다르고 나머지 설정(데이터, seed, optimizer, LR
  schedule 등)은 동일하게 맞춘 상태로 학습/평가를 실행한다.
- `tools/generate_screening_subset.py`, `tools/sanity_check_screening.py`,
  `tools/sanity_check_pseudo_depth.py` — 데이터 subset 생성 및 pseudo-depth
  타겟/모델 forward-backward sanity check.
- `tools/compare_pseudo_depth_experiments.py`,
  `tools/compare_screening_experiments.py`,
  `tools/compare_stage2_experiments.py`,
  `tools/audit_stage2_runtime.py` — 실험 간 결과/런타임 비교 도구.
- `tools/debug_rotate_iou_numba.py` — KITTI evaluator의 rotate-IoU CUDA
  커널 smoke test.
- `tests/test_pseudo_depth.py`, `tests/test_experiment_runner.py` — 위
  변경사항(soft target 생성, same-bin collision, soft/hard focal loss
  일치성, AP parser 등)에 대한 unit test.
