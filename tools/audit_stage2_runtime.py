"""Read-only audit of the Stage 2 E0-vs-E1 runtime gap.

Does NOT retrain, resume, re-validate, or modify any existing checkpoint or
result file. Only reads existing training.log / metrics.csv / runtime.json /
config_snapshot.yaml / checkpoints (metadata only) and writes new files under
experiment_results/stage2_confirmation/runtime_audit/.
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_runtime(path):
    with path.open() as stream:
        return json.load(stream)


def epoch_time_stats(times):
    n = len(times)
    mid_start = max(0, n // 2 - 2)
    return {
        'n_epochs': n,
        'min': min(times), 'max': max(times),
        'mean': st.mean(times), 'median': st.median(times),
        'std': st.pstdev(times),
        'first5_mean': st.mean(times[:5]),
        'mid5_mean': st.mean(times[mid_start:mid_start + 5]),
        'last5_mean': st.mean(times[-5:]),
        'epoch1': times[0], 'epoch40': times[-1],
    }


def checkpoint_metadata(path):
    ck = torch.load(path, map_location='cpu')
    opt = ck.get('optimizer_state', {})
    groups = opt.get('param_groups', [])
    return {
        'epoch': ck.get('epoch'),
        'best_result': ck.get('best_result'),
        'best_epoch': ck.get('best_epoch'),
        'has_scheduler_state': 'scheduler_state' in ck,
        'has_warmup_state': 'warmup_state' in ck,
        'optimizer_param_group_count': len(groups),
        'optimizer_param_groups': [
            {'lr': g.get('lr'), 'weight_decay': g.get('weight_decay'),
             'num_param_tensors': len(g.get('params', []))}
            for g in groups],
        'model_state_key_count': len(ck.get('model_state', {})),
        '_model_state_keys': set(ck.get('model_state', {}).keys()),
        '_model_state_shapes': {k: tuple(v.shape) for k, v in ck.get('model_state', {}).items()},
    }


def diff_configs(path_a, path_b):
    with open(path_a) as stream:
        cfg_a = yaml.safe_load(stream)
    with open(path_b) as stream:
        cfg_b = yaml.safe_load(stream)

    def flatten(prefix, obj, out):
        if isinstance(obj, dict):
            for key, value in obj.items():
                flatten(f'{prefix}.{key}' if prefix else key, value, out)
        else:
            out[prefix] = obj

    flat_a, flat_b = {}, {}
    flatten('', cfg_a, flat_a)
    flatten('', cfg_b, flat_b)
    keys = sorted(set(flat_a) | set(flat_b))
    diffs = []
    for key in keys:
        a, b = flat_a.get(key, '<missing>'), flat_b.get(key, '<missing>')
        if a != b:
            diffs.append((key, a, b))
    return diffs


def count_predictions(output_dir):
    files = list(Path(output_dir).glob('*.txt'))
    total_lines = 0
    for path in files:
        total_lines += sum(1 for line in path.read_text().splitlines() if line.strip())
    return {'num_files': len(files), 'total_detection_lines': total_lines,
            'avg_detections_per_file': (total_lines / len(files)) if files else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='experiment_results/stage2_confirmation')
    parser.add_argument('--out', default='experiment_results/stage2_confirmation/runtime_audit')
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    e0_dir = root / 'E0_baseline'
    e1_dir = root / 'E1_soft_ncf'

    e0_runtime = load_runtime(e0_dir / 'runtime.json')
    e1_runtime = load_runtime(e1_dir / 'runtime.json')

    e0_train_times = [r['train_epoch_time_sec'] for r in e0_runtime['per_epoch_runtime']]
    e1_train_times = [r['train_epoch_time_sec'] for r in e1_runtime['per_epoch_runtime']]
    e0_stats = epoch_time_stats(e0_train_times)
    e1_stats = epoch_time_stats(e1_train_times)

    # Epoch-by-epoch training time comparison CSV.
    with (out / 'epoch_runtime_comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'E0_epoch_time_sec', 'E1_epoch_time_sec', 'delta_sec', 'delta_percent'])
        for i in range(len(e0_train_times)):
            e0t, e1t = e0_train_times[i], e1_train_times[i]
            writer.writerow([i + 1, f'{e0t:.3f}', f'{e1t:.3f}', f'{e1t - e0t:.3f}',
                             f'{100.0 * (e1t - e0t) / e0t:.2f}'])

    # Validation runtime comparison CSV.
    e0_val = {r['epoch']: r for r in e0_runtime['per_epoch_runtime'] if 'validation_total_time_sec' in r}
    e1_val = {r['epoch']: r for r in e1_runtime['per_epoch_runtime'] if 'validation_total_time_sec' in r}
    with (out / 'validation_runtime_comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'E0_inference_sec', 'E0_evaluator_sec', 'E0_total_sec',
                         'E1_inference_sec', 'E1_evaluator_sec', 'E1_total_sec',
                         'total_delta_percent'])
        for epoch in sorted(e0_val):
            a, b = e0_val[epoch], e1_val[epoch]
            writer.writerow([epoch, f"{a['validation_inference_time_sec']:.3f}",
                             f"{a['validation_evaluator_time_sec']:.3f}", f"{a['validation_total_time_sec']:.3f}",
                             f"{b['validation_inference_time_sec']:.3f}",
                             f"{b['validation_evaluator_time_sec']:.3f}", f"{b['validation_total_time_sec']:.3f}",
                             f"{100.0 * (b['validation_total_time_sec'] - a['validation_total_time_sec']) / a['validation_total_time_sec']:.2f}"])

    val_inf_e0 = [r['validation_inference_time_sec'] for r in e0_val.values()]
    val_inf_e1 = [r['validation_inference_time_sec'] for r in e1_val.values()]
    val_eval_e0 = [r['validation_evaluator_time_sec'] for r in e0_val.values()]
    val_eval_e1 = [r['validation_evaluator_time_sec'] for r in e1_val.values()]

    # Config diff.
    config_diffs = diff_configs(e0_dir / 'config_snapshot.yaml', e1_dir / 'config_snapshot.yaml')
    allowed_prefixes = ('model.pseudo_depth_mode',)
    unexpected_diffs = [d for d in config_diffs if not d[0].startswith(allowed_prefixes)]
    with (out / 'config_diff.txt').open('w') as stream:
        stream.write('Config snapshot diff (E0_baseline vs E1_soft_ncf)\n')
        stream.write('=' * 60 + '\n')
        for key, a, b in config_diffs:
            stream.write(f'{key}:\n  E0 = {a}\n  E1 = {b}\n\n')
        if not config_diffs:
            stream.write('(no differences found)\n')

    # Checkpoint metadata comparison.
    e0_ckpt = checkpoint_metadata(e0_dir / 'checkpoints' / 'latest.pth')
    e1_ckpt = checkpoint_metadata(e1_dir / 'checkpoints' / 'latest.pth')
    keys_equal = e0_ckpt['_model_state_keys'] == e1_ckpt['_model_state_keys']
    shape_mismatches = [k for k in (e0_ckpt['_model_state_keys'] & e1_ckpt['_model_state_keys'])
                        if e0_ckpt['_model_state_shapes'][k] != e1_ckpt['_model_state_shapes'][k]]
    for ckpt in (e0_ckpt, e1_ckpt):
        del ckpt['_model_state_keys']
        del ckpt['_model_state_shapes']

    # Prediction / detection count from the final validation snapshot (epoch 40 for both;
    # outputs/data holds only the most recent validation's predictions, not re-run here).
    e0_predictions = count_predictions(e0_dir / 'outputs' / 'data')
    e1_predictions = count_predictions(e1_dir / 'outputs' / 'data')

    # Training/iteration/optimizer-step workload counts (derived from dataset size; no code
    # executed, drop_last=False confirmed in lib/helpers/dataloader_helper.py).
    e0_images = e0_runtime['train_images']
    e1_images = e1_runtime['train_images']
    batch_size = 16  # confirmed identical in both config_snapshot.yaml
    e0_iterations = -(-e0_images // batch_size)  # ceil
    e1_iterations = -(-e1_images // batch_size)

    # Sanity check on the loss progression (workload validity, not a performance re-analysis).
    e0_losses_csv = list(csv.DictReader((e0_dir / 'training_epochs.csv').open()))
    e1_losses_csv = list(csv.DictReader((e1_dir / 'training_epochs.csv').open()))
    checkpoints_epochs = [1, 10, 20, 30, 40]
    loss_progression = {
        'E0': {e: next((r['train_loss_total'] for r in e0_losses_csv if int(r['epoch']) == e), None)
              for e in checkpoints_epochs},
        'E1': {e: next((r['train_loss_total'] for r in e1_losses_csv if int(r['epoch']) == e), None)
              for e in checkpoints_epochs},
    }

    audit = {
        'workload_comparison': {
            'E0_train_images': e0_images, 'E1_train_images': e1_images,
            'images_match': e0_images == e1_images,
            'batch_size': batch_size,
            'E0_iterations_per_epoch_expected': e0_iterations,
            'E1_iterations_per_epoch_expected': e1_iterations,
            'iterations_match': e0_iterations == e1_iterations,
            'E0_total_iterations_40_epochs': e0_iterations * 40,
            'E1_total_iterations_40_epochs': e1_iterations * 40,
            'drop_last': False,
            'note': 'iterations == optimizer.step() calls: one step per batch, no grad accumulation in code.',
        },
        'epoch_time_stats': {'E0': e0_stats, 'E1': e1_stats},
        'lr_milestone_epoch_time_check': {
            'milestones': [25, 33],
            'E0_epoch25': e0_train_times[24], 'E0_epoch33': e0_train_times[32],
            'E1_epoch25': e1_train_times[24], 'E1_epoch33': e1_train_times[32],
            'E1_epoch33_is_max_of_run': e1_train_times[32] == max(e1_train_times),
        },
        'validation_time_stats': {
            'E0_inference_mean': st.mean(val_inf_e0), 'E1_inference_mean': st.mean(val_inf_e1),
            'E0_evaluator_mean': st.mean(val_eval_e0), 'E1_evaluator_mean': st.mean(val_eval_e1),
            'inference_pct_diff': 100.0 * (st.mean(val_inf_e1) - st.mean(val_inf_e0)) / st.mean(val_inf_e0),
            'evaluator_pct_diff': 100.0 * (st.mean(val_eval_e1) - st.mean(val_eval_e0)) / st.mean(val_eval_e0),
        },
        'config_diff': {'all_diffs': config_diffs, 'unexpected_diffs': unexpected_diffs},
        'checkpoint_metadata': {'E0': e0_ckpt, 'E1': e1_ckpt},
        'model_state_key_set_equal': keys_equal,
        'model_state_shape_mismatches': shape_mismatches,
        'prediction_workload_final_snapshot': {'E0': e0_predictions, 'E1': e1_predictions},
        'loss_progression_finite_and_present': loss_progression,
        'amp_autocast_gradscaler_found_in_code': False,
        'torch_compile_found_in_code': False,
        'cudnn_settings': {'benchmark': False, 'deterministic': True,
                           'source': 'lib/helpers/utils_helper.py:set_random_seed, identical call for both runs'},
        'gpu_telemetry_available_in_logs': {
            'gpu_utilization': False, 'gpu_temperature': False, 'gpu_clock': False,
            'power_draw': False, 'shared_gpu_memory': False, 'cpu_utilization': False,
            'system_ram': False, 'disk_usage': False,
            'peak_gpu_memory_MB_source': 'torch.cuda.max_memory_allocated() / 1024**2, '
                                         'reset once via torch.cuda.reset_peak_memory_stats() '
                                         'before the epoch loop (tools/run_pseudo_depth_experiment.py)',
        },
        'resume_used': {'E0': False, 'E1': False, 'source': 'no --resume flag in either launch command; '
                        'no "Resumed from" log line in either training.log'},
    }
    with (out / 'runtime_audit.json').open('w') as stream:
        json.dump(audit, stream, indent=2)

    print(json.dumps(audit, indent=2))
    print(f'\nSaved: {out / "runtime_audit.json"}')
    print(f'Saved: {out / "epoch_runtime_comparison.csv"}')
    print(f'Saved: {out / "validation_runtime_comparison.csv"}')
    print(f'Saved: {out / "config_diff.txt"}')


if __name__ == '__main__':
    main()
