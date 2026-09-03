"""Staged E0/E1/E2 pseudo-depth experiments for MonoDETR.

This launcher intentionally runs exactly one requested experiment. It reuses
the official model, optimizer, dataloaders, decoder, and KITTI evaluator.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.decode_helper import decode_detections, extract_dets_from_outputs
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.save_helper import get_checkpoint_state, load_checkpoint, save_checkpoint
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger, set_random_seed
from utils import misc
from utils import box_ops


EXPERIMENTS = {
    'budget_search': ('center', 'phase0_budget'),
    'baseline': ('center', 'E0_baseline'),
    'soft_ncf': ('soft_ncf', 'E1_soft_ncf'),
    'center_extent': ('center_extent', 'E2_center_extent'),
    'screening_baseline': ('center', 'screening/E0_baseline'),
    'screening_soft_ncf': ('soft_ncf', 'screening/E1_soft_ncf'),
    'screening_center_extent': ('center_extent', 'screening/E2_center_extent'),
    'stage2_baseline': ('center', 'stage2_confirmation/E0_baseline'),
    'stage2_soft_ncf': ('soft_ncf', 'stage2_confirmation/E1_soft_ncf'),
}
KITTI_AP_UNIT = 'AP points (0-100)'
SCREENING_EPOCHS = 25
SCREENING_TRAIN_SPLIT = 'train_screening_20'
SCREENING_VALIDATION_INTERVAL = 5
STAGE2_EPOCHS = 40
STAGE2_TRAIN_SPLIT = 'train_confirmation_50'
STAGE2_VALIDATION_INTERVAL = 5
STAGE_LABELS = {'screening_': 'Stage 1 Screening', 'stage2_': 'Stage 2 Confirmation'}
SPLIT_STATS_PATHS = {
    SCREENING_TRAIN_SPLIT: ROOT / 'experiment_results' / 'screening' / 'subset' / 'subset_statistics.json',
    STAGE2_TRAIN_SPLIT: ROOT / 'experiment_results' / 'stage2_confirmation' / 'subset' / 'subset_statistics.json',
}


def load_split_seed(train_split):
    stats_path = SPLIT_STATS_PATHS.get(train_split)
    if stats_path is None or not stats_path.exists():
        return None
    with stats_path.open() as stream:
        return json.load(stream).get('selected_seed')


def stage_prefix_of(experiment):
    return next((prefix for prefix in STAGE_LABELS if experiment.startswith(prefix)), None)


class Phase0Controller:
    """Validation-driven LR reduction and early stopping for budget search."""
    def __init__(self, min_delta=0.5, lr_patience=2, early_patience=6,
                 lr_factor=0.1, min_lr=1e-6):
        self.min_delta = float(min_delta)
        self.lr_patience = int(lr_patience)
        self.early_patience = int(early_patience)
        self.lr_factor = float(lr_factor)
        self.min_lr = float(min_lr)
        self.significant_best = -float('inf')
        self.absolute_best = -float('inf')
        self.absolute_best_epoch = 0
        self.scheduler_bad_count = 0
        self.early_bad_count = 0
        self.lr_reduction_epochs = []

    def step(self, current_ap, epoch, optimizer=None):
        if current_ap > self.absolute_best:
            self.absolute_best = float(current_ap)
            self.absolute_best_epoch = int(epoch)
        significant = current_ap >= self.significant_best + self.min_delta
        if significant:
            self.significant_best = float(current_ap)
            self.scheduler_bad_count = 0
            self.early_bad_count = 0
        else:
            self.scheduler_bad_count += 1
            self.early_bad_count += 1

        reduced = False
        if self.scheduler_bad_count >= self.lr_patience:
            can_reduce = optimizer is None or any(
                group['lr'] > self.min_lr
                for group in optimizer.param_groups)
            if optimizer is not None and can_reduce:
                for group in optimizer.param_groups:
                    group['lr'] = max(group['lr'] * self.lr_factor,
                                      self.min_lr)
            if can_reduce:
                self.lr_reduction_epochs.append(int(epoch))
                self.scheduler_bad_count = 0
                self.early_bad_count = 0
                reduced = True
        should_stop = self.early_bad_count >= self.early_patience
        return {'significant': significant, 'lr_reduced': reduced,
                'should_stop': should_stop}

    def state_dict(self):
        return {
            key: getattr(self, key) for key in (
                'min_delta', 'lr_patience', 'early_patience', 'lr_factor',
                'min_lr', 'significant_best', 'absolute_best',
                'absolute_best_epoch', 'scheduler_bad_count',
                'early_bad_count', 'lr_reduction_epochs')}

    def load_state_dict(self, state):
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)
BASE_COLUMNS = [
    'epoch', 'learning_rate', 'AP_unit', 'train_loss_total', 'train_loss_depth_map',
    'loss_total', 'loss_depth_map', 'loss_object_depth',
    'Car_3D_AP40_Easy', 'Car_3D_AP40_Moderate',
    'Car_3D_AP40_Hard', 'Car_BEV_AP40_Easy', 'Car_BEV_AP40_Moderate',
    'Car_BEV_AP40_Hard',
    'map_depth_MAE', 'map_depth_RMSE', 'map_depth_MAE_0_20',
    'map_depth_MAE_20_40', 'map_depth_MAE_40_plus',
    'final_depth_MAE', 'final_depth_RMSE', 'final_depth_MAE_0_20',
    'final_depth_MAE_20_40', 'final_depth_MAE_40_plus',
    'absolute_best_AP', 'absolute_best_epoch', 'significant_best_AP',
    'scheduler_bad_count', 'early_bad_count', 'lr_reduced',
]
E2_COLUMNS = ['loss_depth_map_center', 'loss_depth_extent',
              'weighted_loss_depth_extent', 'extent_MAE', 'extent_RMSE',
              'extent_GT_mean', 'extent_pred_mean']


def prepare_targets(targets, batch_size):
    mask = targets['mask_2d']
    keys = ['labels', 'boxes', 'calibs', 'depth', 'depth_near', 'depth_far',
            'depth_extent', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
    return [{key: targets[key][b][mask[b]] for key in keys}
            for b in range(batch_size)]


def train_one_epoch(model, criterion, optimizer, loader, device):
    model.train()
    criterion.train()
    totals = {}
    batches = 0
    epoch_start = time.perf_counter()
    for inputs, calibs, targets, _ in loader:
        inputs, calibs = inputs.to(device), calibs.to(device)
        targets = {key: value.to(device) for key, value in targets.items()}
        image_sizes = targets['img_size']
        target_list = prepare_targets(targets, inputs.shape[0])
        optimizer.zero_grad()
        outputs = model(inputs, calibs, target_list, image_sizes, dn_args=None)
        loss_dict = criterion(outputs, target_list, None)
        weighted = [loss_dict[key] * criterion.weight_dict[key]
                    for key in loss_dict if key in criterion.weight_dict]
        total_loss = sum(weighted)
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f'non-finite total loss: {total_loss.item()}')
        total_loss.backward()
        optimizer.step()

        reduced = misc.reduce_dict(loss_dict)
        for key, value in reduced.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        totals['train_loss_total'] = totals.get('train_loss_total', 0.0) + float(total_loss.detach())
        batches += 1
    epoch_time = time.perf_counter() - epoch_start
    result = {key: value / max(batches, 1) for key, value in totals.items()}
    result['epoch_time_sec'] = epoch_time
    result['avg_iter_time_sec'] = epoch_time / max(batches, 1)
    result['num_batches'] = batches
    return result


def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    area_b = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def error_summary(prefix, predictions, targets, depths):
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    names = [f'{prefix}_MAE', f'{prefix}_RMSE', f'{prefix}_MAE_0_20',
             f'{prefix}_MAE_20_40', f'{prefix}_MAE_40_plus']
    if len(predictions) == 0:
        return {key: float('nan') for key in names}
    errors = predictions - targets
    absolute = np.abs(errors)
    result = {f'{prefix}_MAE': float(absolute.mean()),
              f'{prefix}_RMSE': float(np.sqrt(np.mean(errors ** 2)))}
    for key, mask in (
        (f'{prefix}_MAE_0_20', (depths >= 0) & (depths < 20)),
        (f'{prefix}_MAE_20_40', (depths >= 20) & (depths < 40)),
        (f'{prefix}_MAE_40_plus', depths >= 40),
    ):
        result[key] = float(absolute[mask].mean()) if mask.any() else float('nan')
    return result


class DepthMetricAccumulator:
    """Auxiliary diagnostics, separate from official KITTI AP metrics.

    These use top-k Car predictions and greedy 2D-IoU matching at IoU >= 0.5;
    they do not participate in KITTI matching or AP computation.
    """
    def __init__(self, topk=50, threshold=0.2, iou_threshold=0.5):
        self.topk = topk
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.final_predictions, self.map_predictions = [], []
        self.depth_targets, self.target_depths = [], []
        self.extent_predictions, self.extent_targets = [], []

    @torch.no_grad()
    def __call__(self, outputs, info, dataset):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        probabilities = logits.sigmoid()
        scores, flat_indices = torch.topk(
            probabilities.flatten(1), self.topk, dim=1)
        query_indices = flat_indices // logits.shape[2]
        labels = flat_indices % logits.shape[2]
        gather_boxes = query_indices.unsqueeze(-1).expand(-1, -1, 6)
        selected_boxes = torch.gather(boxes, 1, gather_boxes)
        selected_boxes = box_ops.box_cxcylrtb_to_xyxy(selected_boxes).cpu().numpy()

        def gather_scalar(name):
            value = outputs[name]
            if value.ndim == 2:
                value = value.unsqueeze(-1)
            return torch.gather(value, 1, query_indices.unsqueeze(-1)).squeeze(-1).cpu().numpy()

        final_depth = gather_scalar('pred_depth') if outputs['pred_depth'].shape[-1] == 1 else torch.gather(
            outputs['pred_depth'][:, :, 0], 1, query_indices).cpu().numpy()
        map_depth = gather_scalar('pred_depth_map_sampled')
        extent = gather_scalar('pred_depth_extent_sampled') if 'pred_depth_extent_sampled' in outputs else None
        scores, labels = scores.cpu().numpy(), labels.cpu().numpy()
        image_sizes = info['img_size'].cpu().numpy()
        image_ids = info['img_id'].cpu().numpy()
        for b, image_id in enumerate(image_ids):
            width, height = image_sizes[b]
            scale = np.array([width, height, width, height], dtype=np.float32)
            pred_boxes = selected_boxes[b] * scale
            keep = (labels[b] == dataset.cls2id['Car']) & (scores[b] >= self.threshold)
            pred_indices = np.flatnonzero(keep)
            objects = [obj for obj in dataset.get_label(int(image_id)) if obj.cls_type == 'Car']
            gt_boxes = np.asarray([obj.box2d for obj in objects], dtype=np.float32)
            used = np.zeros(len(objects), dtype=bool)
            for pred_index in pred_indices[np.argsort(-scores[b, pred_indices])]:
                ious = box_iou(pred_boxes[pred_index], gt_boxes)
                ious[used] = -1
                if len(ious) == 0 or ious.max() < self.iou_threshold:
                    continue
                match = int(ious.argmax())
                used[match] = True
                gt_depth = float(objects[match].pos[2])
                self.final_predictions.append(float(final_depth[b, pred_index]))
                self.map_predictions.append(float(map_depth[b, pred_index]))
                self.depth_targets.append(gt_depth)
                self.target_depths.append(gt_depth)
                if extent is not None:
                    corners = objects[match].generate_corners3d()
                    self.extent_predictions.append(float(extent[b, pred_index]))
                    self.extent_targets.append(float(corners[:, 2].max() - corners[:, 2].min()))

    def metrics(self):
        result = error_summary('final_depth', self.final_predictions,
                               self.depth_targets, self.target_depths)
        result.update(error_summary('map_depth', self.map_predictions,
                                    self.depth_targets, self.target_depths))
        if self.extent_predictions:
            predictions = np.asarray(self.extent_predictions)
            targets = np.asarray(self.extent_targets)
            errors = predictions - targets
            result.update({
                'extent_MAE': float(np.abs(errors).mean()),
                'extent_RMSE': float(np.sqrt(np.mean(errors ** 2))),
                'extent_GT_mean': float(targets.mean()),
                'extent_pred_mean': float(predictions.mean()),
            })
        else:
            result.update({key: float('nan') for key in (
                'extent_MAE', 'extent_RMSE', 'extent_GT_mean', 'extent_pred_mean')})
        return result


@torch.no_grad()
def validate(model, loader, tester_cfg, result_dir, logger, min_delta=None):
    from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
    import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
    model.eval()
    helper = Tester(tester_cfg, model, loader, logger,
                    train_cfg={'save_path': str(result_dir.parent)},
                    model_name=result_dir.name)
    helper.output_dir = str(result_dir)
    accumulator = DepthMetricAccumulator(
        topk=tester_cfg['topk'], threshold=tester_cfg.get('threshold', 0.2))
    helper.analysis_callback = accumulator
    inference_start = time.perf_counter()
    helper.inference()
    inference_time = time.perf_counter() - inference_start
    prediction_dir = result_dir / 'outputs' / 'data'
    image_ids = [int(value) for value in loader.dataset.idx_list]
    gt_annos = kitti.get_label_annos(loader.dataset.label_dir, image_ids)
    dt_annos = kitti.get_label_annos(str(prediction_dir), image_ids)
    evaluator_start = time.perf_counter()
    result_text, ap, _ = get_official_eval_result(gt_annos, dt_annos, 0)
    evaluator_time = time.perf_counter() - evaluator_start
    logger.info(result_text)
    metrics = parse_car_ap40(ap)
    metrics.update(accumulator.metrics())
    metrics['validation_inference_time_sec'] = inference_time
    metrics['validation_evaluator_time_sec'] = evaluator_time
    metrics['validation_total_time_sec'] = inference_time + evaluator_time
    print_validation_metrics(metrics, min_delta=min_delta)
    print(f'[Validation Runtime] inference={inference_time:.1f}s '
          f'evaluator={evaluator_time:.1f}s total={inference_time + evaluator_time:.1f}s '
          f'({len(image_ids)} images)')
    logger.info('Validation runtime: inference=%.2fs evaluator=%.2fs total=%.2fs (%d images)',
                inference_time, evaluator_time, inference_time + evaluator_time, len(image_ids))
    return metrics


def append_csv(path, columns, row):
    new_file = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in columns})


def phase0_checkpoint_state(model, optimizer, epoch, controller):
    state = get_checkpoint_state(
        model, optimizer, epoch, controller.absolute_best,
        controller.absolute_best_epoch)
    state['phase0_controller'] = controller.state_dict()
    return state


def plot_phase0_curves(rows, result_dir, lr_reduction_epochs):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    epochs = [row['epoch'] for row in rows]
    aps = [row['Car_3D_AP40_Moderate'] for row in rows]
    best = np.maximum.accumulate(aps)
    figure, axis = plt.subplots()
    axis.plot(epochs, aps, marker='o', label='validation AP40 Moderate')
    axis.plot(epochs, best, linestyle='--', label='absolute best')
    for epoch in lr_reduction_epochs:
        axis.axvline(epoch, color='tab:red', alpha=0.35)
    axis.set(xlabel='epoch', ylabel='AP points')
    axis.legend()
    figure.tight_layout()
    figure.savefig(result_dir / 'phase0_ap_curve.png')
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.plot(epochs, [row['learning_rate'] for row in rows], marker='o')
    axis.set(xlabel='epoch', ylabel='learning rate', yscale='log')
    figure.tight_layout()
    figure.savefig(result_dir / 'phase0_lr_curve.png')
    plt.close(figure)
    return True


def plot_screening_curves(rows, result_dir, prefix, lr_milestones=()):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    epochs = [row['epoch'] for row in rows]

    def render(values, ylabel, filename):
        figure, axis = plt.subplots()
        axis.plot(epochs, values, marker='o')
        for milestone in lr_milestones:
            axis.axvline(milestone, color='tab:red', alpha=0.35, linestyle='--')
        axis.set(xlabel='epoch', ylabel=ylabel)
        figure.tight_layout()
        figure.savefig(result_dir / filename)
        plt.close(figure)

    render([row['Car_3D_AP40_Moderate'] for row in rows],
           'Car 3D AP40 Moderate (AP points)', f'{prefix}_ap3d_moderate_curve.png')
    render([row.get('map_depth_MAE', float('nan')) for row in rows],
           'map_depth_MAE (m)', f'{prefix}_map_depth_mae_curve.png')
    render([row.get('final_depth_MAE', float('nan')) for row in rows],
           'final_depth_MAE (m)', f'{prefix}_final_depth_mae_curve.png')
    if any(row.get('extent_MAE') == row.get('extent_MAE') and row.get('extent_MAE') is not None
           for row in rows):
        render([row.get('extent_MAE', float('nan')) for row in rows],
               'extent_MAE (m)', f'{prefix}_extent_mae_curve.png')
    return True


def parse_car_ap40(ap):
    """Select official R40 AP points by name; evaluator already multiplies by 100."""
    mapping = {
        'Car_3D_AP40_Easy': 'Car_3d_easy_R40',
        'Car_3D_AP40_Moderate': 'Car_3d_moderate_R40',
        'Car_3D_AP40_Hard': 'Car_3d_hard_R40',
        'Car_BEV_AP40_Easy': 'Car_bev_easy_R40',
        'Car_BEV_AP40_Moderate': 'Car_bev_moderate_R40',
        'Car_BEV_AP40_Hard': 'Car_bev_hard_R40',
    }
    missing = [source for source in mapping.values() if source not in ap]
    if missing:
        raise KeyError(f'Official KITTI AP40 keys missing: {missing}')
    return {target: float(ap[source]) for target, source in mapping.items()}


def print_validation_metrics(metrics, min_delta=None):
    print('[Official KITTI Evaluation]')
    for family in ('3D', 'BEV'):
        print(f'Car {family} AP40 [AP points]')
        for difficulty in ('Easy', 'Moderate', 'Hard'):
            key = f'Car_{family}_AP40_{difficulty}'
            print(f'{difficulty:<8} = {metrics[key]:.4f}')
        print()
    print('Early stopping metric:')
    print('Car 3D AP40 Moderate = {:.4f} AP'.format(
        metrics['Car_3D_AP40_Moderate']))
    if min_delta is not None:
        print(f'min_delta = {min_delta:g} AP')


def evaluator_preflight(result_dir):
    """Check the untouched official evaluator and persist a staged diagnosis."""
    debug_dir = result_dir.parent / 'evaluator_debug'
    debug_dir.mkdir(parents=True, exist_ok=True)
    status = {
        'checkpoint_loaded': False,
        'validation_inference_completed': False,
        'prediction_txt_created': False,
        'evaluator_import_succeeded': False,
        'numba_cuda_jit_succeeded': False,
        'official_evaluation_completed': False,
    }
    try:
        from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result  # noqa: F401
        status['evaluator_import_succeeded'] = True
        status['numba_cuda_jit_succeeded'] = True
    except Exception as error:
        message = traceback.format_exc()
        status.update({
            'failure_stage': 'B_NUMBA_CUDA_JIT',
            'error_type': type(error).__name__,
            'error': str(error),
            'traceback': message,
        })
        with (result_dir / 'smoke_status.json').open('w') as stream:
            json.dump(status, stream, indent=2)
        print('Evaluator smoke: FAIL (B: Numba CUDA JIT compilation)')
        print(message)
        print('Status:', result_dir / 'smoke_status.json')
        return status
    with (debug_dir / 'import_smoke.json').open('w') as stream:
        json.dump({
            'rotate_iou_import': True,
            'official_eval_import': True,
            'cuda_available': torch.cuda.is_available(),
        }, stream, indent=2)
    return status


def inspect_prediction_files(prediction_dir, expected_ids):
    """Validate KITTI result serialization without changing its format."""
    paths = [prediction_dir / f'{int(image_id):06d}.txt'
             for image_id in expected_ids]
    empty, malformed, non_finite, missing = 0, 0, 0, 0
    for path in paths:
        if not path.exists():
            missing += 1
            continue
        lines = [line.strip() for line in path.read_text().splitlines()
                 if line.strip()]
        if not lines:
            empty += 1
        for line in lines:
            fields = line.split()
            if len(fields) != 16:
                malformed += 1
                continue
            try:
                values = np.asarray([float(value) for value in fields[1:]])
                if not np.isfinite(values).all():
                    non_finite += 1
            except ValueError:
                malformed += 1
    return {
        'num_prediction_files': sum(path.exists() for path in paths),
        'missing_prediction_files': missing,
        'empty_prediction_files': empty,
        'malformed_prediction_lines': malformed,
        'non_finite_prediction_lines': non_finite,
    }


def configure(args):
    with open(args.config, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.Loader)
    mode, directory = EXPERIMENTS[args.experiment]
    cfg['model']['pseudo_depth_mode'] = mode
    cfg['trainer']['save_frequency'] = 5
    if args.experiment == 'budget_search':
        cfg['trainer']['max_epoch'] = 195
    elif args.experiment.startswith('screening_'):
        # Stage 1 screening: fixed budget for every model (E0/E1/E2 alike),
        # no adaptive plateau search, no early stopping. Compressed LR
        # milestones reuse MonoDETR's 125/195 and 165/195 decay ratios.
        epochs = SCREENING_EPOCHS
        cfg['trainer']['max_epoch'] = epochs
        cfg['dataset']['train_split'] = SCREENING_TRAIN_SPLIT
        cfg['lr_scheduler']['decay_list'] = [
            math.floor(epochs * 125 / 195), math.floor(epochs * 165 / 195)]
        cfg['split_seed'] = load_split_seed(SCREENING_TRAIN_SPLIT)
    elif args.experiment.startswith('stage2_'):
        # Stage 2 confirmation: same fixed-budget philosophy as screening,
        # larger subset (50%) and longer budget (40 epochs) to check whether
        # the Stage 1 signal reproduces. Only E0/E1 are compared here.
        epochs = STAGE2_EPOCHS
        cfg['trainer']['max_epoch'] = epochs
        cfg['dataset']['train_split'] = STAGE2_TRAIN_SPLIT
        cfg['lr_scheduler']['decay_list'] = [
            math.floor(epochs * 125 / 195), math.floor(epochs * 165 / 195)]
        cfg['split_seed'] = load_split_seed(STAGE2_TRAIN_SPLIT)
    else:
        if not args.budget_json:
            raise ValueError('--budget-json is required for Phase 1 experiments')
        with open(args.budget_json, 'r') as stream:
            budget = json.load(stream)
        epochs = int(budget['E_budget'])
        cfg['trainer']['max_epoch'] = epochs
        cfg['lr_scheduler']['decay_list'] = [
            math.floor(epochs * 125 / 195), math.floor(epochs * 165 / 195)]
    return cfg, ROOT / args.results_root / directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True,
                        choices=list(EXPERIMENTS) + ['evaluator_smoke'])
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'monodetr.yaml'))
    parser.add_argument('--budget-json')
    parser.add_argument('--results-root', default='experiment_results')
    parser.add_argument('--checkpoint', help='checkpoint for evaluator_smoke')
    parser.add_argument('--resume', help='resume a Phase 0 latest checkpoint')
    args = parser.parse_args()
    if args.experiment == 'evaluator_smoke':
        if not args.checkpoint:
            parser.error('--checkpoint is required for evaluator_smoke')
        with open(args.config, 'r') as stream:
            cfg = yaml.load(stream, Loader=yaml.Loader)
        cfg['model']['pseudo_depth_mode'] = 'center'
        result_dir = ROOT / args.results_root / 'evaluator_smoke'
        result_dir.mkdir(parents=True, exist_ok=True)
        logger = create_logger(str(result_dir / 'smoke.log'))
        set_random_seed(cfg.get('random_seed', 444))
        _, val_loader = build_dataloader(cfg['dataset'])
        model, _ = build_model(cfg['model'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        load_checkpoint(model, None, args.checkpoint, device, logger)
        smoke_status = evaluator_preflight(result_dir)
        smoke_status['checkpoint_loaded'] = True
        if not smoke_status['evaluator_import_succeeded']:
            with (result_dir / 'smoke_status.json').open('w') as stream:
                json.dump(smoke_status, stream, indent=2)
            return
        metrics = validate(model, val_loader, cfg['tester'], result_dir, logger)
        prediction_dir = result_dir / 'outputs' / 'data'
        image_ids = [int(value) for value in val_loader.dataset.idx_list]
        file_status = inspect_prediction_files(prediction_dir, image_ids)
        smoke_status.update({
            'success': True,
            'checkpoint': str(Path(args.checkpoint).resolve()),
            'num_validation_images': len(image_ids),
            'validation_inference_completed': True,
            'prediction_txt_created': True,
            'official_evaluation_completed': True,
            'Car_3d_easy_R40': metrics['Car_3D_AP40_Easy'],
            'Car_3d_moderate_R40': metrics['Car_3D_AP40_Moderate'],
            'Car_3d_hard_R40': metrics['Car_3D_AP40_Hard'],
            'Car_bev_easy_R40': metrics['Car_BEV_AP40_Easy'],
            'Car_bev_moderate_R40': metrics['Car_BEV_AP40_Moderate'],
            'Car_bev_hard_R40': metrics['Car_BEV_AP40_Hard'],
            'early_stopping_metric': 'Car_3d_moderate_R40',
            'AP_unit': KITTI_AP_UNIT,
            # The official evaluator dictionary already contains AP points.
            # These are equal by design; both are recorded to prevent a future
            # mistaken fraction-to-percent conversion.
            'raw_evaluator_metrics': {
                'Car_3d_easy_R40': metrics['Car_3D_AP40_Easy'],
                'Car_3d_moderate_R40': metrics['Car_3D_AP40_Moderate'],
                'Car_3d_hard_R40': metrics['Car_3D_AP40_Hard'],
                'Car_bev_easy_R40': metrics['Car_BEV_AP40_Easy'],
                'Car_bev_moderate_R40': metrics['Car_BEV_AP40_Moderate'],
                'Car_bev_hard_R40': metrics['Car_BEV_AP40_Hard'],
            },
            'experiment_metrics_ap_points': {
                'Car_3d_easy_R40': metrics['Car_3D_AP40_Easy'],
                'Car_3d_moderate_R40': metrics['Car_3D_AP40_Moderate'],
                'Car_3d_hard_R40': metrics['Car_3D_AP40_Hard'],
                'Car_bev_easy_R40': metrics['Car_BEV_AP40_Easy'],
                'Car_bev_moderate_R40': metrics['Car_BEV_AP40_Moderate'],
                'Car_bev_hard_R40': metrics['Car_BEV_AP40_Hard'],
            },
            'environment': {
                'python': sys.version.split()[0],
                'torch': torch.__version__,
                'torch_cuda': torch.version.cuda,
                'numba': __import__('numba').__version__,
            },
        })
        smoke_status.update(file_status)
        with (result_dir / 'metrics.json').open('w') as stream:
            json.dump(metrics, stream, indent=2)
        with (result_dir / 'smoke_status.json').open('w') as stream:
            json.dump(smoke_status, stream, indent=2)
        print('Evaluator smoke: PASS')
        print('Prediction directory:', result_dir / 'outputs' / 'data')
        return
    cfg, result_dir = configure(args)
    seed = int(cfg.get('random_seed', 444))
    cfg['experiment_initialization'] = {
        'seed': seed,
        'python_random_seed': seed,
        'numpy_seed': seed ** 2,
        'pytorch_cpu_seed': seed ** 3,
        'pytorch_cuda_seed': seed ** 4,
        'backbone_initialization': 'torchvision ResNet-50 ImageNet pretrained',
        'pretrained_backbone': True,
        'full_model_checkpoint_initialization': bool(args.resume),
        'resume_checkpoint': args.resume,
    }
    checkpoints = result_dir / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    with (result_dir / 'config_snapshot.yaml').open('w') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    logger = create_logger(str(result_dir / 'training.log'))
    set_random_seed(seed)

    train_loader, val_loader = build_dataloader(cfg['dataset'])
    model, criterion = build_model(cfg['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, criterion = model.to(device), criterion.to(device)
    optimizer = build_optimizer(cfg['optimizer'], model)
    if args.experiment == 'budget_search':
        scheduler, warmup = None, None
        phase0_cfg = cfg.get('phase0', {})
        controller = Phase0Controller(
            min_delta=phase0_cfg.get('min_delta', 0.5),
            lr_patience=phase0_cfg.get('lr_plateau_patience', 2),
            early_patience=phase0_cfg.get('early_stop_patience', 6),
            lr_factor=phase0_cfg.get('lr_decay_factor', 0.1),
            min_lr=phase0_cfg.get('min_lr', 1e-6))
        validation_interval = int(phase0_cfg.get('validation_interval', 5))
    else:
        scheduler, warmup = build_lr_scheduler(cfg['lr_scheduler'], optimizer, -1)
        controller = None
        validation_interval = 5

    columns = BASE_COLUMNS + (E2_COLUMNS if EXPERIMENTS[args.experiment][0] == 'center_extent' else [])
    metrics_path = result_dir / 'metrics.csv'
    epoch_metrics_path = result_dir / 'training_epochs.csv'
    absolute_best = -float('inf')
    best_epoch = 0
    start_epoch = 1
    stopped_epoch = cfg['trainer']['max_epoch']
    best_metrics = {}
    validation_history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = int(checkpoint['epoch']) + 1
        absolute_best = float(checkpoint.get('best_result', -float('inf')))
        best_epoch = int(checkpoint.get('best_epoch', 0))
        if controller is not None:
            controller.load_state_dict(checkpoint['phase0_controller'])
        if scheduler is not None and 'scheduler_state' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        if warmup is not None and 'warmup_state' in checkpoint:
            warmup.load_state_dict(checkpoint['warmup_state'])
        logger.info('Resumed from %s: epoch=%d absolute_best=%.4f best_epoch=%d current_lr=%.8g',
                    args.resume, start_epoch - 1, absolute_best, best_epoch,
                    optimizer.param_groups[0]['lr'])

    stage_prefix = stage_prefix_of(args.experiment)
    if stage_prefix is not None:
        summary = f"""[{STAGE_LABELS[stage_prefix]}]
Experiment: {args.experiment}
Pseudo depth mode: {EXPERIMENTS[args.experiment][0]}
Dataset: KITTI
Train split: {cfg['dataset']['train_split']} ({len(train_loader.dataset)} images)
Validation split: {cfg['dataset']['test_split']} ({len(val_loader.dataset)} images)
Subset seed: {cfg.get('split_seed')}
Max epochs: {cfg['trainer']['max_epoch']}
Validation interval: {validation_interval}
Validation epochs: {[e for e in range(validation_interval, cfg['trainer']['max_epoch'] + 1, validation_interval)]}
Early stopping: disabled
Best-checkpoint metric: Car 3D AP40 Moderate (absolute maximum, no min_delta)
LR milestones (compressed from 125/195, 165/195 ratios): {cfg['lr_scheduler']['decay_list']}
LR decay factor: {cfg['lr_scheduler'].get('decay_rate', cfg['lr_scheduler'].get('decay_factor'))}
AP unit: {KITTI_AP_UNIT}
Seed: {seed} (Python={seed}, NumPy={seed ** 2}, Torch CPU={seed ** 3}, CUDA={seed ** 4})
Backbone: ResNet-50 ImageNet pretrained
Full model checkpoint initialization: {bool(args.resume)}
Optimizer: {cfg['optimizer']['type']}
Initial LR: {optimizer.param_groups[0]['lr']}
Weight decay: {cfg['optimizer']['weight_decay']}
Batch size: {cfg['dataset']['batch_size']}
Input resolution: 1280x384
Number of queries: {cfg['model'].get('num_queries')}
Number of depth bins: {cfg['model'].get('num_depth_bins')}"""
        print(summary)
        logger.info(summary)

    if args.experiment == 'budget_search':
        summary = f"""[Phase 0 Budget Search]
Pseudo depth mode: center
Dataset: KITTI
Train split: {cfg['dataset']['train_split']}
Validation split: {cfg['dataset']['test_split']} ({len(val_loader.dataset)} images)
Max epochs: {cfg['trainer']['max_epoch']}
Validation interval: {validation_interval}
Metric: Car 3D AP40 Moderate
AP unit: {KITTI_AP_UNIT}
min_delta: {controller.min_delta} AP
LR plateau patience: {controller.lr_patience} validations
LR factor: {controller.lr_factor}
Early stop patience: {controller.early_patience} validations
Minimum LR: {controller.min_lr}
Seed: {seed} (Python={seed}, NumPy={seed ** 2}, Torch CPU={seed ** 3}, CUDA={seed ** 4})
Backbone: ResNet-50 ImageNet pretrained
Full model checkpoint initialization: {bool(args.resume)}
Optimizer: {cfg['optimizer']['type']}
Initial LR: {optimizer.param_groups[0]['lr']}
Weight decay: {cfg['optimizer']['weight_decay']}
Batch size: {cfg['dataset']['batch_size']}"""
        print(summary)
        logger.info(summary)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    run_start = time.perf_counter()
    runtime_records = {}
    for epoch in range(start_epoch, cfg['trainer']['max_epoch'] + 1):
        epoch_index = epoch - 1
        train_metrics = train_one_epoch(model, criterion, optimizer, train_loader, device)
        train_row = {
            'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr'],
            'train_loss_total': train_metrics['train_loss_total'],
            'train_loss_depth_map': train_metrics.get('loss_depth_map', float('nan')),
            'train_loss_object_depth': train_metrics.get('loss_depth', float('nan')),
            'epoch_time_sec': train_metrics.get('epoch_time_sec', float('nan')),
            'avg_iter_time_sec': train_metrics.get('avg_iter_time_sec', float('nan')),
        }
        append_csv(epoch_metrics_path, list(train_row), train_row)
        logger.info('Epoch %d train total=%.6f depth_map=%.6f object_depth=%.6f lr=%.8g '
                    'epoch_time=%.1fs avg_iter_time=%.3fs',
                    epoch, train_row['train_loss_total'], train_row['train_loss_depth_map'],
                    train_row['train_loss_object_depth'], train_row['learning_rate'],
                    train_row['epoch_time_sec'], train_row['avg_iter_time_sec'])
        print(f'[Epoch {epoch}] train_time={train_row["epoch_time_sec"]:.1f}s '
              f'avg_iter_time={train_row["avg_iter_time_sec"]:.3f}s '
              f'loss_total={train_row["train_loss_total"]:.4f}')
        runtime_records[epoch] = {
            'epoch': epoch,
            'train_epoch_time_sec': train_row['epoch_time_sec'],
            'train_avg_iter_time_sec': train_row['avg_iter_time_sec'],
        }
        if args.experiment != 'budget_search':
            if warmup is not None and epoch_index < 5:
                warmup.step()
            else:
                scheduler.step()
        if epoch % validation_interval != 0:
            if controller is not None:
                save_checkpoint(phase0_checkpoint_state(
                    model, optimizer, epoch, controller), str(checkpoints / 'latest'))
            continue

        validation = validate(
            model, val_loader, cfg['tester'], result_dir, logger,
            min_delta=controller.min_delta if controller is not None else None)
        row = {'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr'],
               'AP_unit': KITTI_AP_UNIT,
               'train_loss_total': train_metrics['train_loss_total'],
               'train_loss_depth_map': train_metrics.get(
                   'loss_depth_map', train_metrics.get('loss_depth_map_center', float('nan'))),
               'loss_total': train_metrics['train_loss_total'],
               'loss_depth_map': train_metrics.get(
                   'loss_depth_map', train_metrics.get('loss_depth_map_center', float('nan'))),
               'loss_object_depth': train_metrics.get('loss_depth', float('nan')),
               'loss_depth_map_center': train_metrics.get('loss_depth_map_center', ''),
               'loss_depth_extent': train_metrics.get('loss_depth_extent', ''),
               'weighted_loss_depth_extent': train_metrics.get('weighted_loss_depth_extent', '')}
        row.update(validation)
        runtime_records[epoch].update({
            'validation_inference_time_sec': validation.get('validation_inference_time_sec'),
            'validation_evaluator_time_sec': validation.get('validation_evaluator_time_sec'),
            'validation_total_time_sec': validation.get('validation_total_time_sec'),
        })
        current = validation['Car_3D_AP40_Moderate']
        decision = None
        if args.experiment == 'budget_search':
            decision = controller.step(current, epoch, optimizer)
        if current > absolute_best:
            absolute_best, best_epoch, best_metrics = current, epoch, dict(row)
            if controller is not None:
                state = phase0_checkpoint_state(model, optimizer, epoch, controller)
            else:
                state = get_checkpoint_state(model, optimizer, epoch, absolute_best, best_epoch)
                if scheduler is not None:
                    state['scheduler_state'] = scheduler.state_dict()
                if warmup is not None:
                    state['warmup_state'] = warmup.state_dict()
            save_checkpoint(state, str(checkpoints / 'best'))
        if args.experiment == 'budget_search':
            row.update({
                'learning_rate': optimizer.param_groups[0]['lr'],
                'absolute_best_AP': controller.absolute_best,
                'absolute_best_epoch': controller.absolute_best_epoch,
                'significant_best_AP': controller.significant_best,
                'scheduler_bad_count': controller.scheduler_bad_count,
                'early_bad_count': controller.early_bad_count,
                'lr_reduced': decision['lr_reduced'],
            })
            logger.info(
                'Phase0 state: significant_best=%.4f absolute_best=%.4f '
                'scheduler_bad=%d early_bad=%d lr=%.8g reduced=%s',
                controller.significant_best, controller.absolute_best,
                controller.scheduler_bad_count, controller.early_bad_count,
                optimizer.param_groups[0]['lr'], decision['lr_reduced'])
            print(f'''[Validation State]
Epoch: {epoch}
Moderate AP40: {current:.4f} AP
Absolute best: {controller.absolute_best:.4f} AP @ epoch {controller.absolute_best_epoch}
Significant best: {controller.significant_best:.4f} AP
Scheduler bad count: {controller.scheduler_bad_count} / {controller.lr_patience}
Early-stop bad count: {controller.early_bad_count} / {controller.early_patience}
LR: {optimizer.param_groups[0]['lr']:.8g}
LR reduced: {decision['lr_reduced']}''')
        validation_history.append(dict(row))
        append_csv(metrics_path, columns, row)
        if args.experiment == 'budget_search':
            append_csv(ROOT / args.results_root / 'baseline_budget_curve.csv', columns, row)
            save_checkpoint(phase0_checkpoint_state(
                model, optimizer, epoch, controller), str(checkpoints / 'latest'))
        else:
            latest_state = get_checkpoint_state(
                model, optimizer, epoch, absolute_best, best_epoch)
            if scheduler is not None:
                latest_state['scheduler_state'] = scheduler.state_dict()
            if warmup is not None:
                latest_state['warmup_state'] = warmup.state_dict()
            save_checkpoint(latest_state, str(checkpoints / 'latest'))
        if epoch == best_epoch:
            best_metrics = dict(row)
            with (result_dir / 'best_metrics.json').open('w') as stream:
                json.dump(best_metrics, stream, indent=2)
        if args.experiment == 'budget_search':
            if decision['should_stop']:
                stopped_epoch = epoch
                break

    if args.experiment == 'budget_search':
        budget_result = {
            'experiment': 'budget_search',
            'pseudo_depth_mode': 'center',
            'metric': 'Car_3D_AP40_Moderate', 'max_epochs': 195,
            'AP_unit': KITTI_AP_UNIT,
            'validation_interval': validation_interval,
            'min_delta': controller.min_delta,
            'lr_plateau_patience': controller.lr_patience,
            'lr_decay_factor': controller.lr_factor,
            'min_lr': controller.min_lr,
            'early_stop_patience': controller.early_patience,
            'E_budget': stopped_epoch, 'E_best': best_epoch,
            'absolute_best_AP': controller.absolute_best,
            'significant_best_AP': controller.significant_best,
            'lr_reduction_epochs': controller.lr_reduction_epochs,
            'seed': seed,
            'final_learning_rate': optimizer.param_groups[0]['lr'],
        }
        with (result_dir / 'budget_search.json').open('w') as stream:
            json.dump(budget_result, stream, indent=2)
        # Requested compatibility alias.
        alias = ROOT / args.results_root / 'budget_search.json'
        with alias.open('w') as stream:
            json.dump(budget_result, stream, indent=2)
        plot_phase0_curves(
            validation_history, result_dir, controller.lr_reduction_epochs)
        print('\nLast validation history:')
        for item in validation_history[-6:]:
            print(f"Epoch {item['epoch']:3d}: "
                  f"{item['Car_3D_AP40_Moderate']:.4f} AP, "
                  f"LR={item['learning_rate']:.8g}")
        last_ap = validation_history[-1]['Car_3D_AP40_Moderate']
        print(f'''\nE_budget: {stopped_epoch}
E_best: {best_epoch}
E_budget - E_best: {stopped_epoch - best_epoch}
Best AP: {absolute_best:.4f}
Last AP: {last_ap:.4f}
LR reduction epochs: {controller.lr_reduction_epochs}''')
        if best_epoch == stopped_epoch:
            print('WARNING: Best validation performance occurred at the training boundary. '
                  'The selected budget may be too short.')
        if (len(validation_history) >= 3 and all(
                validation_history[index]['Car_3D_AP40_Moderate'] <
                validation_history[index + 1]['Car_3D_AP40_Moderate']
                for index in range(len(validation_history) - 3,
                                   len(validation_history) - 1))):
            print('WARNING: Validation AP is still improving near E_budget.')
        if stopped_epoch < 30:
            print('WARNING: Early stopping selected a very small training budget. '
                  'Review convergence before Phase 1.')
        if stopped_epoch >= 180:
            print('WARNING: Budget search provided little computational saving '
                  'relative to the original 195-epoch schedule.')

    if stage_prefix is not None:
        total_wall_time = time.perf_counter() - run_start
        train_times = [record['train_epoch_time_sec'] for record in runtime_records.values()
                       if record.get('train_epoch_time_sec') is not None]
        validation_times = [record['validation_total_time_sec'] for record in runtime_records.values()
                            if record.get('validation_total_time_sec') is not None]
        peak_gpu_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2
                       if device.type == 'cuda' else float('nan'))
        moderate_aps = [item['Car_3D_AP40_Moderate'] for item in validation_history]
        runtime_summary = {
            'experiment': args.experiment,
            'pseudo_depth_mode': EXPERIMENTS[args.experiment][0],
            'train_images': len(train_loader.dataset),
            'validation_images': len(val_loader.dataset),
            'max_epochs': cfg['trainer']['max_epoch'],
            'validation_interval': validation_interval,
            'lr_milestones': cfg['lr_scheduler']['decay_list'],
            'seed': seed,
            'split_seed': cfg.get('split_seed'),
            'average_epoch_train_time_sec': sum(train_times) / len(train_times) if train_times else float('nan'),
            'total_training_time_sec': sum(train_times),
            'average_validation_time_sec': sum(validation_times) / len(validation_times) if validation_times else float('nan'),
            'total_validation_time_sec': sum(validation_times),
            'overall_wall_time_sec': total_wall_time,
            'peak_gpu_memory_MB': peak_gpu_mb,
            'best_epoch': best_epoch,
            'best_Car_3D_AP40_Moderate': absolute_best,
            'final_epoch_Car_3D_AP40_Moderate': moderate_aps[-1] if moderate_aps else float('nan'),
            'mean_validation_Car_3D_AP40_Moderate': (
                sum(moderate_aps) / len(moderate_aps) if moderate_aps else float('nan')),
            'per_epoch_runtime': list(runtime_records.values()),
        }
        with (result_dir / 'runtime.json').open('w') as stream:
            json.dump(runtime_summary, stream, indent=2)
        prefix = result_dir.name.split('_')[0]
        plot_screening_curves(validation_history, result_dir, prefix,
                              lr_milestones=cfg['lr_scheduler']['decay_list'])
        print(f'''\n[Runtime Summary: {args.experiment}]
Average epoch train time: {runtime_summary['average_epoch_train_time_sec']:.1f}s
Total training time: {runtime_summary['total_training_time_sec']:.1f}s
Average validation time: {runtime_summary['average_validation_time_sec']:.1f}s
Total validation time: {runtime_summary['total_validation_time_sec']:.1f}s
Overall wall time: {runtime_summary['overall_wall_time_sec']:.1f}s
Peak GPU memory: {peak_gpu_mb:.1f} MB
Best epoch: {best_epoch} (Moderate AP40={absolute_best:.4f})
Final epoch Moderate AP40: {runtime_summary['final_epoch_Car_3D_AP40_Moderate']:.4f}
Mean validation Moderate AP40: {runtime_summary['mean_validation_Car_3D_AP40_Moderate']:.4f}''')
        logger.info('Runtime summary: %s', json.dumps(runtime_summary))


if __name__ == '__main__':
    main()
