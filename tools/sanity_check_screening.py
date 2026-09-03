"""Few-iteration sanity check that the screening config/dataset wiring works.

Not a short epoch: only a handful of batches per mode, enough to prove the
train_screening_20 split, dataloader, model, loss, and optimizer are all
connected correctly for E0/E1/E2 before committing to a real 25-epoch run.
"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.utils_helper import set_random_seed
from tools.run_pseudo_depth_experiment import EXPERIMENTS, prepare_targets

STAGE_CONFIG = {
    'stage1': {'epochs': 25, 'train_split': 'train_screening_20',
               'experiments': ('screening_baseline', 'screening_soft_ncf', 'screening_center_extent')},
    'stage2': {'epochs': 40, 'train_split': 'train_confirmation_50',
               'experiments': ('stage2_baseline', 'stage2_soft_ncf')},
}


def run_mode(cfg, experiment_key, num_iterations, device, epochs, train_split):
    mode, _ = EXPERIMENTS[experiment_key]
    local = yaml.safe_load(yaml.safe_dump(cfg))
    local['model']['pseudo_depth_mode'] = mode
    local['dataset']['train_split'] = train_split
    local['lr_scheduler']['decay_list'] = [
        math.floor(epochs * 125 / 195), math.floor(epochs * 165 / 195)]
    seed = int(local.get('random_seed', 444))
    set_random_seed(seed)

    train_loader, _ = build_dataloader(local['dataset'], workers=0)
    print(f'[{experiment_key}] mode={mode} train_split={train_split} '
          f'images={len(train_loader.dataset)} batch_size={local["dataset"]["batch_size"]}')

    model, criterion = build_model(local['model'])
    model, criterion = model.to(device), criterion.to(device)
    optimizer = build_optimizer(local['optimizer'], model)
    model.train()
    criterion.train()

    losses = []
    for step, (inputs, calibs, targets, _) in enumerate(itertools.islice(train_loader, num_iterations)):
        inputs, calibs = inputs.to(device), calibs.to(device)
        targets = {key: value.to(device) for key, value in targets.items()}
        prepared = prepare_targets(targets, inputs.shape[0])
        optimizer.zero_grad()
        outputs = model(inputs, calibs, prepared, targets['img_size'], dn_args=None)
        loss_dict = criterion(outputs, prepared, None)
        total_loss = sum(loss_dict[key] * criterion.weight_dict[key]
                          for key in loss_dict if key in criterion.weight_dict)
        assert torch.isfinite(total_loss), f'non-finite loss at step {step}: {total_loss}'
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.detach()))
        print(f'  iter {step + 1}/{num_iterations} total_loss={losses[-1]:.6f}')

    assert len(losses) == num_iterations, (
        f'expected {num_iterations} iterations, only ran {len(losses)} '
        f'(dataset/batch size too small for this many batches)')
    print(f'[{experiment_key}] PASS ({num_iterations} iterations, no NaN/Inf, optimizer step succeeded)')
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'monodetr.yaml'))
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--stage', choices=list(STAGE_CONFIG), default='stage1')
    args = parser.parse_args()
    with open(args.config) as stream:
        cfg = yaml.load(stream, Loader=yaml.Loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    stage = STAGE_CONFIG[args.stage]
    results = {}
    for experiment_key in stage['experiments']:
        results[experiment_key] = run_mode(cfg, experiment_key, args.iterations, device,
                                           stage['epochs'], stage['train_split'])

    print('\n=== Sanity summary ===')
    for experiment_key, losses in results.items():
        print(f'{experiment_key}: losses={["%.4f" % value for value in losses]}')
    print(f'All {args.stage} modes wired correctly: PASS')


if __name__ == '__main__':
    main()
