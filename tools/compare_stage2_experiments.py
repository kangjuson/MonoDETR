"""Stage 2 E0/E1 confirmation comparison, plus Stage 1 vs Stage 2 delta comparison."""
import argparse
import csv
import json
import math
from pathlib import Path

MODELS = [
    ('E0', 'Baseline', 'center', 'E0_baseline'),
    ('E1', 'Soft-NCF', 'soft_ncf', 'E1_soft_ncf'),
]
VALIDATION_EPOCHS = [5, 10, 15, 20, 25, 30, 35, 40]
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


def build_core_row(tag, label, mode, data):
    best = data['best_metrics'] or {}
    return {
        'Model': tag, 'Label': label, 'Pseudo-depth': mode,
        'Best Epoch': best.get('epoch'),
        'AP3D Easy': best.get('Car_3D_AP40_Easy'),
        'AP3D Moderate': best.get('Car_3D_AP40_Moderate'),
        'AP3D Hard': best.get('Car_3D_AP40_Hard'),
        'BEV Easy': best.get('Car_BEV_AP40_Easy'),
        'BEV Moderate': best.get('Car_BEV_AP40_Moderate'),
        'BEV Hard': best.get('Car_BEV_AP40_Hard'),
        'Map Depth MAE': best.get('map_depth_MAE'),
        'Final Depth MAE': best.get('final_depth_MAE'),
    }


def delta_block(core_e0, core_e1):
    def delta(key):
        a, b = core_e0.get(key), core_e1.get(key)
        if a is None or b is None:
            return None
        return b - a
    return {
        'delta_AP3D_Easy': delta('AP3D Easy'),
        'delta_AP3D_Moderate': delta('AP3D Moderate'),
        'delta_AP3D_Hard': delta('AP3D Hard'),
        'delta_BEV_Easy': delta('BEV Easy'),
        'delta_BEV_Moderate': delta('BEV Moderate'),
        'delta_BEV_Hard': delta('BEV Hard'),
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
        result[tag] = entry
    return result


def classify_stage2(delta_moderate, wins, total, depth_supports, easy_or_hard_improved):
    if delta_moderate >= 0.5 and easy_or_hard_improved and wins > total / 2:
        return 'CONFIRMED STRONG'
    if delta_moderate > 0 and wins > total / 2 and depth_supports:
        return 'CONFIRMED POSITIVE'
    if delta_moderate <= 0 and wins <= total / 2 and not depth_supports:
        return 'NOT CONFIRMED'
    return 'WEAK / INCONCLUSIVE'


def load_stage1_report(stage1_root):
    path = Path(stage1_root) / 'screening_report.json'
    return read_json(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='experiment_results/stage2_confirmation')
    parser.add_argument('--stage1-root', default='experiment_results/screening')
    args = parser.parse_args()
    root = Path(args.root)

    models = [(tag, label, mode, load_model(root, directory))
              for tag, label, mode, directory in MODELS]
    missing = [tag for tag, _, _, data in models if data['best_metrics'] is None]
    if missing:
        raise SystemExit(f'Missing best_metrics.json for: {missing}. Run both Stage 2 experiments first.')

    core_rows = [build_core_row(tag, label, mode, data) for tag, label, mode, data in models]
    fields = ['Model', 'Label', 'Pseudo-depth', 'Best Epoch', 'AP3D Easy', 'AP3D Moderate',
              'AP3D Hard', 'BEV Easy', 'BEV Moderate', 'BEV Hard', 'Map Depth MAE', 'Final Depth MAE']
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(core_rows)

    core_by_tag = {row['Model']: row for row in core_rows}
    delta_e1 = delta_block(core_by_tag['E0'], core_by_tag['E1'])

    traj = {tag: trajectory(data['metrics_rows']) for tag, _, _, data in models}
    mean_val_ap = {tag: mean_of(traj[tag].values()) for tag in traj}

    wins = sum(1 for epoch in VALIDATION_EPOCHS
              if traj['E1'].get(epoch) is not None and traj['E0'].get(epoch) is not None
              and traj['E1'][epoch] > traj['E0'][epoch])
    losses = sum(1 for epoch in VALIDATION_EPOCHS
                if traj['E1'].get(epoch) is not None and traj['E0'].get(epoch) is not None
                and traj['E1'][epoch] < traj['E0'][epoch])
    ties = len(VALIDATION_EPOCHS) - wins - losses
    win_margins = [traj['E1'][epoch] - traj['E0'][epoch] for epoch in VALIDATION_EPOCHS
                  if traj['E1'].get(epoch) is not None and traj['E0'].get(epoch) is not None
                  and traj['E1'][epoch] > traj['E0'][epoch]]

    dist = distance_wise(models)
    runtime_cmp = runtime_comparison(models)

    easy_or_hard_improved = (delta_e1['delta_AP3D_Easy'] or 0) > 0 or (delta_e1['delta_AP3D_Hard'] or 0) > 0
    depth_supports = (delta_e1['delta_final_depth_MAE'] or 0) < 0 or (delta_e1['delta_map_depth_MAE'] or 0) < 0
    classification = classify_stage2(delta_e1['delta_AP3D_Moderate'] or 0.0, wins, len(VALIDATION_EPOCHS),
                                     depth_supports, easy_or_hard_improved)

    stage1_report = load_stage1_report(args.stage1_root)
    stage1_vs_stage2 = None
    if stage1_report is not None:
        s1 = stage1_report['delta_vs_E0']['E1']
        stage1_vs_stage2 = {
            'stage1_delta_AP3D_Moderate': s1['delta_AP3D_Moderate'],
            'stage2_delta_AP3D_Moderate': delta_e1['delta_AP3D_Moderate'],
            'moderate_sign_preserved': (s1['delta_AP3D_Moderate'] > 0) == (delta_e1['delta_AP3D_Moderate'] > 0),
            'moderate_magnitude_change': (delta_e1['delta_AP3D_Moderate'] - s1['delta_AP3D_Moderate']
                                          if delta_e1['delta_AP3D_Moderate'] is not None else None),
            'stage1_delta_AP3D_Hard': s1['delta_AP3D_Hard'],
            'stage2_delta_AP3D_Hard': delta_e1['delta_AP3D_Hard'],
            'hard_sign_preserved': (s1['delta_AP3D_Hard'] > 0) == (delta_e1['delta_AP3D_Hard'] > 0),
            'stage1_delta_final_depth_MAE': s1['delta_final_depth_MAE'],
            'stage2_delta_final_depth_MAE': delta_e1['delta_final_depth_MAE'],
            'final_depth_sign_preserved': (s1['delta_final_depth_MAE'] < 0) == (delta_e1['delta_final_depth_MAE'] < 0),
            'stage1_final_depth_MAE_40_plus': {
                'E0': stage1_report['distance_wise_depth_error_at_best_epoch']['final_depth_MAE']['E0']['40m+'],
                'E1': stage1_report['distance_wise_depth_error_at_best_epoch']['final_depth_MAE']['E1']['40m+'],
            },
            'stage2_final_depth_MAE_40_plus': {
                'E0': dist['final_depth_MAE']['E0']['40m+'],
                'E1': dist['final_depth_MAE']['E1']['40m+'],
            },
        }
        far_improved_stage1 = (stage1_vs_stage2['stage1_final_depth_MAE_40_plus']['E1'] <
                               stage1_vs_stage2['stage1_final_depth_MAE_40_plus']['E0'])
        far_improved_stage2 = (stage1_vs_stage2['stage2_final_depth_MAE_40_plus']['E1'] <
                               stage1_vs_stage2['stage2_final_depth_MAE_40_plus']['E0'])
        stage1_vs_stage2['far_depth_improvement_preserved'] = far_improved_stage1 and far_improved_stage2

    report = {
        'core_results': core_rows,
        'delta_E1_vs_E0': delta_e1,
        'validation_trajectory_Car_3D_AP40_Moderate': traj,
        'mean_validation_Car_3D_AP40_Moderate': mean_val_ap,
        'checkpoint_comparison': {'E1_gt_E0': wins, 'E1_eq_E0': ties, 'E1_lt_E0': losses,
                                  'of': len(VALIDATION_EPOCHS), 'mean_margin_when_winning':
                                  (sum(win_margins) / len(win_margins) if win_margins else 0.0)},
        'distance_wise_depth_error_at_best_epoch': dist,
        'runtime_comparison': runtime_cmp,
        'stage1_vs_stage2': stage1_vs_stage2,
        'confirmation_classification': classification,
    }
    with (root / 'stage2_report.json').open('w') as stream:
        json.dump(report, stream, indent=2)

    try:
        import matplotlib.pyplot as plt
        colors = {'E0': 'tab:gray', 'E1': 'tab:blue'}

        def render(metric_key, ylabel, filename):
            figure, axis = plt.subplots()
            for tag, _, _, data in models:
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

        render('Car_3D_AP40_Moderate', 'Car 3D AP40 Moderate (AP points)', 'stage2_ap3d_moderate_curve.png')
        render('map_depth_MAE', 'map_depth_MAE (m)', 'stage2_map_depth_mae_curve.png')
        render('final_depth_MAE', 'final_depth_MAE (m)', 'stage2_final_depth_mae_curve.png')
    except ImportError:
        pass

    print(f'Saved: {root / "summary.csv"}')
    print(f'Saved: {root / "stage2_report.json"}')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
