"""Pre-training sanity checks for E0/E1/E2 pseudo-depth supervision."""
import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault('NUMBA_DISABLE_JIT', '1')

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.helpers.dataloader_helper import build_dataloader
from lib.models.monodetr.depth_predictor.ddn_loss.ddn_loss import DDNLoss
from lib.models.monodetr.depth_predictor.ddn_loss.focalloss import FocalLoss
from tools.run_pseudo_depth_experiment import prepare_targets
from utils import box_ops


def legacy_center_map(depth_logits, gt_boxes2d, gt_center_depth, counts):
    """Independent copy of the official pre-experiment painter for comparison."""
    batch, _, height, width = depth_logits.shape
    result = torch.zeros((batch, height, width), device=depth_logits.device,
                         dtype=depth_logits.dtype)
    boxes = gt_boxes2d.clone()
    boxes[:, :2] = torch.floor(boxes[:, :2])
    boxes[:, 2:] = torch.ceil(boxes[:, 2:])
    boxes = boxes.long().split(counts, dim=0)
    depths = gt_center_depth.split(counts, dim=0)
    for b, (batch_boxes, batch_depths) in enumerate(zip(boxes, depths)):
        sorted_depths, order = torch.sort(batch_depths, descending=True)
        for box, value in zip(batch_boxes[order], sorted_depths):
            u1, v1, u2, v2 = box
            result[b, v1:v2, u1:u2] = value
    return result


def target_checks(cfg):
    dataset_cfg = dict(cfg['dataset'])
    dataset_cfg.update({'aug_pd': False, 'aug_crop': False,
                        'random_flip': 0.0, 'random_crop': 0.0})
    dataset = KITTI_Dataset('train', dataset_cfg)
    for sample_index in range(min(100, len(dataset))):
        _, _, targets, info = dataset[sample_index]
        valid = np.flatnonzero(targets['mask_2d'])
        if len(valid) >= 3:
            break
    if len(valid) < 3:
        raise RuntimeError('Fewer than three valid objects found in the first 100 training samples')
    print('Dataset sample:', info['img_id'])
    ddn = DDNLoss()
    logits = torch.zeros(1, 81, 24, 80)
    boxes = torch.as_tensor(targets['boxes'][valid]) * torch.tensor([80, 24, 80, 24])
    boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    center = torch.as_tensor(targets['depth'][valid, 0])
    near = torch.as_tensor(targets['depth_near'][valid, 0])
    far = torch.as_tensor(targets['depth_far'][valid, 0])
    extent = torch.as_tensor(targets['depth_extent'][valid, 0])
    assert torch.all(near <= center + 1e-5)
    assert torch.all(center <= far + 1e-5)
    assert torch.all(extent >= 0)
    for index in range(min(5, len(valid))):
        bins = [int(ddn.bin_depths(value[None], target=True)[0])
                for value in (near[index], center[index], far[index])]
        cls = dataset.class_name[int(targets['labels'][valid[index]])]
        print(cls, f'near={near[index]:.3f}', f'center={center[index]:.3f}',
              f'far={far[index]:.3f}', f'extent={extent[index]:.3f}',
              f'bins={bins}')

    original = legacy_center_map(logits, boxes, center, [len(valid)])
    baseline = ddn.build_target_depth_from_3dcenter(
        logits, boxes.clone(), center, [len(valid)])
    difference = (original - baseline).abs().max()
    assert difference == 0, 'baseline target map changed'
    print('Baseline target max difference:', float(difference))
    soft = ddn.build_soft_ncf_target(
        logits, boxes, near, center, far, [len(valid)], (0.2, 0.6, 0.2))
    assert torch.isfinite(soft).all() and soft.min() >= 0
    assert torch.allclose(soft.sum(1), torch.ones_like(soft[:, 0]), atol=1e-6)

    # Explicit same-bin degeneracy.
    one_box = torch.tensor([[1., 1., 5., 5.]])
    one_depth = torch.tensor([10.])
    collapsed = ddn.build_soft_ncf_target(
        logits, one_box, one_depth, one_depth, one_depth, [1], (0.2, 0.6, 0.2))
    pixel = collapsed[0, :, 2, 2]
    assert torch.isclose(pixel.max(), torch.tensor(1.0)) and (pixel > 0).sum() == 1

    focal = FocalLoss(alpha=.25, gamma=2, reduction='none')
    random_logits = torch.randn(2, 5, 3, 4)
    hard = torch.randint(0, 5, (2, 3, 4))
    one_hot = torch.nn.functional.one_hot(hard, 5).permute(0, 3, 1, 2).float()
    hard_loss = focal(random_logits, hard)
    soft_one_hot_loss = focal(random_logits, one_hot)
    # Legacy hard one_hot adds eps to every class; tolerance reflects that only.
    assert torch.allclose(hard_loss, soft_one_hot_loss, atol=2e-5, rtol=2e-5)
    print('Target/loss checks: PASS')


def one_iteration(cfg, mode, device):
    from lib.helpers.model_helper import build_model
    local = yaml.safe_load(yaml.safe_dump(cfg))
    local['model']['pseudo_depth_mode'] = mode
    local['dataset']['batch_size'] = 1
    train_loader, _ = build_dataloader(local['dataset'], workers=0)
    model, criterion = build_model(local['model'])
    model, criterion = model.to(device), criterion.to(device)
    inputs, calibs, targets, _ = next(iter(train_loader))
    inputs, calibs = inputs.to(device), calibs.to(device)
    targets = {key: value.to(device) for key, value in targets.items()}
    prepared = prepare_targets(targets, inputs.shape[0])
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    outputs = model(inputs, calibs, prepared, targets['img_size'], dn_args=None)
    losses = criterion(outputs, prepared, None)
    total = sum(losses[key] * criterion.weight_dict[key]
                for key in losses if key in criterion.weight_dict)
    assert torch.isfinite(total)
    total.backward()
    memory = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == 'cuda' else float('nan')
    selected = {key: float(value.detach()) for key, value in losses.items()
                if key in ('loss_depth_map', 'loss_depth_map_center',
                           'loss_depth_extent', 'weighted_loss_depth_extent')}
    print(mode, 'losses=', selected, f'peak_memory_MB={memory:.1f}')
    return memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'monodetr.yaml'))
    parser.add_argument('--skip-model', action='store_true')
    args = parser.parse_args()
    with open(args.config) as stream:
        cfg = yaml.load(stream, Loader=yaml.Loader)
    target_checks(cfg)
    if args.skip_model:
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memory = {mode: one_iteration(cfg, mode, device)
              for mode in ('center', 'soft_ncf', 'center_extent')}
    if device.type == 'cuda':
        print('GPU memory deltas (MB):',
              {mode: value - memory['center'] for mode, value in memory.items()})


if __name__ == '__main__':
    main()
