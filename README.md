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

### `configs/monodetr.yaml`
`model:` 아래에 `pseudo_depth_mode`, `soft_depth_weights`(near/center/far),
`lambda_extent`를 추가했다.

### `lib/datasets/kitti/kitti_dataset.py`
각 GT 3D box의 8개 corner(camera 좌표계)를 `objects[i].generate_corners3d()`로
구해 z축(카메라 전방 거리) 최솟값/최댓값을 `depth_near`/`depth_far`로 계산하고,
`depth_extent = max(depth_far - depth_near, 0)`을 추가로 만든다. 기존 `depth`
(3D box center depth)는 그대로 둔다. `depth_near ≤ depth ≤ depth_far`가 깨지는
비정상 라벨은 경고를 남기고(최대 10회) 카운트만 한다 — 값 자체를 고치거나
건너뛰지는 않는다.

또한 `train_screening_20`처럼 `train_`으로 시작하는 이름의 subset split도
`train`과 동일하게(같은 augmentation, 같은 디렉터리 구조로) 다룰 수 있도록
split 검증 조건을 넓혔다.

### `lib/models/monodetr/depth_predictor/depth_predictor.py`
`pseudo_depth_mode`를 저장해두고, `center_extent`일 때만 depth-extent 회귀용
head(`extent_predictor`: conv → ReLU → conv → Softplus)를 추가로 만든다. 그 외
모드에서는 `extent_predictor = None`. forward가 반환하는 튜플에 `depth_extent`
(사용 안 하면 `None`)를 하나 더 추가했다. **`weighted_depth`를 만드는 계산식과
depth classifier 자체는 세 모드 모두 완전히 동일**하다 — soft/extent 모드가
바꾸는 것은 오직 학습 시 loss target이지, 추론 시 만들어지는 depth 표현이
아니다.

### `lib/models/monodetr/depth_predictor/ddn_loss/ddn_loss.py`
- 기존 `build_target_depth_from_3dcenter`(hard center target 생성)는 그대로
  둔다 — E0/E2의 center 항은 원본과 byte 단위로 동일한 경로를 탄다.
- `build_soft_ncf_target`을 새로 추가했다: near/center/far 각각을 기존과 동일한
  LID(k+1 bin) 방식으로 bin index화한 뒤, 픽셀별로 `near*0.2 + center*0.6 +
  far*0.2` 형태의 확률분포를 만든다. 세 값이 같은 bin에 몰리면
  `scatter_add_`가 자동으로 가중치를 합산한다(같은 bin 충돌 시 가중치를
  더하는 방식). 가중치 합이 1이 아니거나 음수면 즉시 에러를 낸다. Background는
  기존과 동일하게 마지막(“없음”) 클래스로 채운다.
- `build_extent_target`/`extent_loss`를 새로 추가했다(E2 전용) —
  픽셀 영역에 scalar depth-extent를 칠하고 foreground 영역에서만 smooth-L1
  loss를 계산한다.
- `forward()`에 `mode` 인자를 추가해 분기한다: `center`/`center_extent`는
  기존 hard-target 경로 그대로, `soft_ncf`만 새 soft target을 사용한다.

### `lib/models/monodetr/depth_predictor/ddn_loss/focalloss.py`
`soft_focal_loss`를 새로 추가했다 — 확률분포(soft) target에 대한 multiclass
focal loss로, 정확히 one-hot인 target을 넣으면 기존 hard focal loss와
수치적으로 동일한 값이 나오도록 만들었다(legacy one-hot 구현의 epsilon
차이 정도만 남는다). `FocalLoss.forward`는 넘어온 target의 dtype으로
자동 분기한다: `int64`(기존 hard label) → 기존 `focal_loss` 그대로,
float(확률분포) → 새 `soft_focal_loss`.

### `lib/models/monodetr/monodetr.py`
- `depth_predictor`가 5개 값(마지막에 `depth_extent`)을 반환하도록 호출부를
  맞췄다.
- `SetCriterion.__init__`에 `pseudo_depth_mode`/`soft_depth_weights`/
  `lambda_extent`를 새 파라미터로 받는다.
- `loss_depth_map`: `depth_near`/`depth_far` target을 모아 `ddn_loss`에
  `mode`와 함께 넘긴다. `center_extent`일 때만 `loss_depth_map_center` +
  `loss_depth_extent`(+ `weighted_loss_depth_extent`) 두 항을 추가로 계산하고,
  그 외 모드는 기존과 동일하게 `loss_depth_map` 한 항만 만든다.
- `build()`: `pseudo_depth_mode`에 따라 `weight_dict`의 키를 분기하고
  (`loss_depth_map` vs `loss_depth_map_center`+`weighted_loss_depth_extent`),
  criterion 생성 시 위 세 파라미터를 config에서 읽어 전달한다.
- **최종 object-level depth fusion(`pred_depth` 계산식)은 세 모드 모두
  전혀 바뀌지 않았다.** 새로 추가된 `pred_depth_map_sampled`/
  `pred_depth_extent_sampled`는 분석용으로만 붙인 출력이며 실제 detection
  결과 계산에는 관여하지 않는다.

### `lib/helpers/trainer_helper.py`
배치를 샘플 단위 dict로 변환할 때 넘기는 key 목록에 `depth_near`/
`depth_far`/`depth_extent`를 추가했다 — 위에서 만든 새 타겟들을 학습 루프까지
전달하기 위한 배선.

### `lib/helpers/tester_helper.py`
추론 직후 예측값을 들여다볼 수 있는 `analysis_callback` 훅(기본값 `None`,
지정하지 않으면 아무 동작도 하지 않음)을 추가했다. 공식 평가 로직 자체는
그대로다.

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
