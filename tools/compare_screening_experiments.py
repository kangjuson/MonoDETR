"""Full E0/E1/E2 Stage 1 screening comparison.

Produces summary.csv, screening_report.json, and comparison plots from the
per-experiment metrics.csv / best_metrics.json / runtime.json files under
experiment_results/screening/{E0_baseline,E1_soft_ncf,E2_center_extent}/.
"""
import argparse
import csv
import json
import math
from pathlib import Path

MODELS = [
    ('E0', 'Baseline', 'center', 'E0_baseline'),
    ('E1', 'Soft-NCF', 'soft_ncf', 'E1_soft_ncf'),
    ('E2', 'Center+Extent', 'center_extent', 'E2_center_extent'),
]
VALIDATION_EPOCHS = [5, 10, 15, 20, 25]
DISTANCE_BUCKETS = [('0_20', '0-20m'), ('20_40', '20-40m'), ('40_plus', '40m+')]


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows = []
        for raw in reader:
            row = {}
            for key, value in raw.items():
                if value in (None, ''):
                    row[key] = None
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
        return rows


def read_json(path):
    if not path.exists():
        return None
    with path.open() as stream:
        return json.load(stream)


def load_model(root, directory):
    base = root / directory
    return {
        'metrics_rows': read_csv_rows(base / 'metrics.csv'),
        'best_metrics': read_json(base / 'best_metrics.json'),
        'runtime': read_json(base / 'runtime.json'),
        'dir': str(base),
    }


def trajectory(rows, key='Car_3D_AP40_Moderate'):
    by_epoch = {int(row['epoch']): row.get(key) for row in rows if row.get('epoch') is not None}
    return {epoch: by_epoch.get(epoch) for epoch in VALIDATION_EPOCHS}


def mean_of(values):
    clean = [value for value in values if value is not None and not (isinstance(value, float) and math.isnan(value))]
    return sum(clean) / len(clean) if clean else float('nan')


def classify_signal(delta_best_ap, ap_improved_count, total_checkpoints,
                     easy_improved, hard_improved, map_mae_delta, final_mae_delta):
    depth_improved = (final_mae_delta is not None and final_mae_delta < 0) or \
                     (map_mae_delta is not None and map_mae_delta < 0)
    easy_or_hard_improved_count = int(bool(easy_improved)) + int(bool(hard_improved))
    if delta_best_ap >= 0.5 and (easy_or_hard_improved_count >= 1 or
                                  ap_improved_count >= 4 or depth_improved):
        return 'STRONG POSITIVE'
    if delta_best_ap < 0.5 and (ap_improved_count >= 3 or depth_improved) and delta_best_ap > -0.1:
        return 'POSITIVE/POSSIBLE'
    if delta_best_ap <= -0.3 and ap_improved_count <= 1:
        return 'NEGATIVE'
    return 'NO CLEAR SIGNAL'


def build_core_row(tag, label, mode, data):
    best = data['best_metrics'] or {}
    return {
        'Model': tag, 'Label': label, 'Pseudo-depth': mode,
        'Best Epoch': best.get('epoch'),
        'AP3D Easy': best.get('Car_3D_AP40_Easy'),
        'AP3D Moderate': best.get('Car_3D_AP40_Moderate'),
        'AP3D Hard': best.get('Car_3D_AP40_Hard'),
        'BEV Moderate': best.get('Car_BEV_AP40_Moderate'),
        'Map Depth MAE': best.get('map_depth_MAE'),
        'Final Depth MAE': best.get('final_depth_MAE'),
        'Extent MAE': best.get('extent_MAE') if mode == 'center_extent' else '',
    }


def delta_block(core_e0, core_ex):
    def delta(key):
        a, b = core_e0.get(key), core_ex.get(key)
        if a is None or b is None:
            return None
        return b - a
    return {
        'delta_AP3D_Easy': delta('AP3D Easy'),
        'delta_AP3D_Moderate': delta('AP3D Moderate'),
        'delta_AP3D_Hard': delta('AP3D Hard'),
        'delta_BEV_Moderate': delta('BEV Moderate'),
        'delta_map_depth_MAE': delta('Map Depth MAE'),
        'delta_map_depth_MAE_note': 'negative = improvement (lower MAE)',
        'delta_final_depth_MAE': delta('Final Depth MAE'),
        'delta_final_depth_MAE_note': 'negative = improvement (lower MAE)',
    }


def distance_wise(models):
    result = {'map_depth_MAE': {}, 'final_depth_MAE': {}}
    for tag, _, _, data in models:
        best = data['best_metrics'] or {}
        for metric_prefix in ('map_depth_MAE', 'final_depth_MAE'):
            result[metric_prefix][tag] = {
                label: best.get(f'{metric_prefix}_{suffix}')
                for suffix, label in DISTANCE_BUCKETS
            }
    return result


def extent_analysis(e2_data):
    rows = e2_data['metrics_rows']
    if not rows or not any(row.get('extent_MAE') is not None for row in rows):
        return None
    traj = []
    for epoch in VALIDATION_EPOCHS:
        row = next((r for r in rows if int(r.get('epoch', -1)) == epoch), None)
        if row is None:
            continue
        traj.append({
            'epoch': epoch,
            'extent_MAE': row.get('extent_MAE'),
            'extent_RMSE': row.get('extent_RMSE'),
            'extent_GT_mean': row.get('extent_GT_mean'),
            'extent_pred_mean': row.get('extent_pred_mean'),
        })
    if not traj:
        return None
    first, last = traj[0], traj[-1]
    mae_decreasing = (last['extent_MAE'] is not None and first['extent_MAE'] is not None and
                      last['extent_MAE'] < first['extent_MAE'])
    gap_last = (abs(last['extent_pred_mean'] - last['extent_GT_mean'])
               if last['extent_pred_mean'] is not None and last['extent_GT_mean'] is not None
               else None)
    collapse_suspected = gap_last is not None and last['extent_GT_mean'] not in (None, 0) and \
        gap_last / abs(last['extent_GT_mean']) > 0.5
    return {
        'trajectory': traj,
        'mae_decreasing_over_training': mae_decreasing,
        'final_pred_mean_vs_gt_mean_gap': gap_last,
        'collapse_suspected': collapse_suspected,
    }


def runtime_comparison(models):
    result = {}
    e0_runtime = models[0][3]['runtime']
    for tag, _, _, data in models:
        runtime = data['runtime']
        if runtime is None:
            result[tag] = None
            continue
        entry = dict(runtime)
        if e0_runtime is not None and tag != 'E0':
            epoch_pct = (100.0 * (runtime['average_epoch_train_time_sec'] -
                                  e0_runtime['average_epoch_train_time_sec']) /
                        e0_runtime['average_epoch_train_time_sec'])
            memory_delta = runtime['peak_gpu_memory_MB'] - e0_runtime['peak_gpu_memory_MB']
            entry['epoch_time_pct_change_vs_E0'] = epoch_pct
            entry['peak_gpu_memory_delta_vs_E0_MB'] = memory_delta
            entry['epoch_time_warning'] = epoch_pct >= 20.0
        result[tag] = entry
    return result


def stage2_recommendation(signal_e1, signal_e2, delta_e1, delta_e2, mean_e1, mean_e0, mean_e2):
    order = {'STRONG POSITIVE': 3, 'POSITIVE/POSSIBLE': 2, 'NO CLEAR SIGNAL': 1, 'NEGATIVE': 0}
    if order[signal_e1] <= 1 and order[signal_e2] <= 1:
        return {'choice': 'D', 'reason': 'Neither E1 nor E2 shows a positive signal over E0; '
                'review pseudo-depth target/loss design before committing GPU budget to Stage 2.'}
    if order[signal_e1] > order[signal_e2]:
        return {'choice': 'A', 'reason': 'E1 (soft_ncf) shows the stronger signal over E0. '
                'Stage 2 candidate: E0 vs E1 on 50% KITTI train, 40 epochs.'}
    if order[signal_e2] > order[signal_e1]:
        return {'choice': 'B', 'reason': 'E2 (center_extent) shows the stronger signal over E0. '
                'Stage 2 candidate: E0 vs E2 on 50% KITTI train, 40 epochs.'}
    # Tie: prefer the one with the higher, more consistent mean validation AP.
    winner = 'E1' if mean_e1 >= mean_e2 else 'E2'
    return {'choice': 'C', 'reason': f'E1 and E2 show comparably positive signals ({signal_e1} / {signal_e2}). '
            f'Mean validation AP favors {winner} (E1={mean_e1:.4f}, E2={mean_e2:.4f}); given limited GPU budget, '
            f'{winner} is the more defensible single Stage 2 candidate unless GPU budget allows both.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='experiment_results/screening')
    args = parser.parse_args()
    root = Path(args.root)

    models = []
    for tag, label, mode, directory in MODELS:
        data = load_model(root, directory)
        models.append((tag, label, mode, data))

    missing = [tag for tag, _, _, data in models if data['best_metrics'] is None]
    if missing:
        raise SystemExit(f'Missing best_metrics.json for: {missing}. Run all three experiments first.')

    core_rows = [build_core_row(tag, label, mode, data) for tag, label, mode, data in models]
    fields = ['Model', 'Label', 'Pseudo-depth', 'Best Epoch', 'AP3D Easy', 'AP3D Moderate',
              'AP3D Hard', 'BEV Moderate', 'Map Depth MAE', 'Final Depth MAE', 'Extent MAE']
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(core_rows)

    core_by_tag = {row['Model']: row for row in core_rows}
    delta_e1 = delta_block(core_by_tag['E0'], core_by_tag['E1'])
    delta_e2 = delta_block(core_by_tag['E0'], core_by_tag['E2'])

    traj = {tag: trajectory(data['metrics_rows']) for tag, _, _, data in models}
    mean_val_ap = {tag: mean_of(traj[tag].values()) for tag in traj}

    def checkpoint_wins(challenger):
        wins = sum(1 for epoch in VALIDATION_EPOCHS
                   if traj[challenger].get(epoch) is not None and traj['E0'].get(epoch) is not None
                   and traj[challenger][epoch] > traj['E0'][epoch])
        diffs = [traj[challenger][epoch] - traj['E0'][epoch] for epoch in VALIDATION_EPOCHS
                 if traj[challenger].get(epoch) is not None and traj['E0'].get(epoch) is not None
                 and traj[challenger][epoch] > traj['E0'][epoch]]
        return wins, (sum(diffs) / len(diffs) if diffs else 0.0)

    e1_wins, e1_win_margin = checkpoint_wins('E1')
    e2_wins, e2_win_margin = checkpoint_wins('E2')

    dist = distance_wise(models)
    extent = extent_analysis(models[2][3])
    runtime_cmp = runtime_comparison(models)

    easy_improved_e1 = (delta_e1['delta_AP3D_Easy'] or 0) > 0
    hard_improved_e1 = (delta_e1['delta_AP3D_Hard'] or 0) > 0
    easy_improved_e2 = (delta_e2['delta_AP3D_Easy'] or 0) > 0
    hard_improved_e2 = (delta_e2['delta_AP3D_Hard'] or 0) > 0

    signal_e1 = classify_signal(delta_e1['delta_AP3D_Moderate'] or 0.0, e1_wins, 5,
                                easy_improved_e1, hard_improved_e1,
                                delta_e1['delta_map_depth_MAE'], delta_e1['delta_final_depth_MAE'])
    signal_e2 = classify_signal(delta_e2['delta_AP3D_Moderate'] or 0.0, e2_wins, 5,
                                easy_improved_e2, hard_improved_e2,
                                delta_e2['delta_map_depth_MAE'], delta_e2['delta_final_depth_MAE'])

    stage2 = stage2_recommendation(signal_e1, signal_e2, delta_e1, delta_e2,
                                   mean_val_ap['E1'], mean_val_ap['E0'], mean_val_ap['E2'])

    report = {
        'core_results': core_rows,
        'delta_vs_E0': {'E1': delta_e1, 'E2': delta_e2},
        'validation_trajectory_Car_3D_AP40_Moderate': traj,
        'mean_validation_Car_3D_AP40_Moderate': mean_val_ap,
        'checkpoints_beating_E0': {
            'E1': {'count': e1_wins, 'of': len(VALIDATION_EPOCHS), 'mean_margin_when_winning': e1_win_margin},
            'E2': {'count': e2_wins, 'of': len(VALIDATION_EPOCHS), 'mean_margin_when_winning': e2_win_margin},
        },
        'distance_wise_depth_error_at_best_epoch': dist,
        'extent_analysis_E2': extent,
        'runtime_comparison': runtime_cmp,
        'signal_classification': {'E1_vs_E0': signal_e1, 'E2_vs_E0': signal_e2},
        'stage2_recommendation': stage2,
    }
    with (root / 'screening_report.json').open('w') as stream:
        json.dump(report, stream, indent=2)

    try:
        import matplotlib.pyplot as plt
        colors = {'E0': 'tab:gray', 'E1': 'tab:blue', 'E2': 'tab:orange'}

        def render(metric_key, ylabel, filename, only_tags=None):
            figure, axis = plt.subplots()
            for tag, _, _, data in models:
                if only_tags and tag not in only_tags:
                    continue
                rows = {int(row['epoch']): row.get(metric_key) for row in data['metrics_rows']
                       if row.get('epoch') is not None}
                xs = [epoch for epoch in VALIDATION_EPOCHS if rows.get(epoch) is not None]
                ys = [rows[epoch] for epoch in xs]
                if xs:
                    axis.plot(xs, ys, marker='o', label=tag, color=colors[tag])
            axis.set(xlabel='epoch', ylabel=ylabel)
            axis.legend()
            figure.tight_layout()
            figure.savefig(root / filename)
            plt.close(figure)

        render('Car_3D_AP40_Moderate', 'Car 3D AP40 Moderate (AP points)', 'screening_ap3d_moderate_curve.png')
        render('map_depth_MAE', 'map_depth_MAE (m)', 'screening_map_depth_mae_curve.png')
        render('final_depth_MAE', 'final_depth_MAE (m)', 'screening_final_depth_mae_curve.png')
        if extent is not None:
            render('extent_MAE', 'extent_MAE (m)', 'screening_extent_mae_curve.png', only_tags={'E2'})
    except ImportError:
        pass

    print(f'Saved: {root / "summary.csv"}')
    print(f'Saved: {root / "screening_report.json"}')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
