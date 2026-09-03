"""Compare completed E0/E1/E2 best validation results."""
import argparse
import csv
import json
from pathlib import Path


RUNS = [
    ('Baseline', 'center', 'E0_baseline'),
    ('Soft-NCF', 'soft_ncf', 'E1_soft_ncf'),
    ('Center+Extent', 'center_extent', 'E2_center_extent'),
]
FIELDS = ['Model', 'Pseudo-depth', 'Best Epoch', 'Car AP3D Easy',
          'Car AP3D Moderate', 'Car AP3D Hard', 'BEV AP40 Moderate',
          'Map Depth MAE', 'Map Depth RMSE', 'Final Depth MAE',
          'Final Depth RMSE', 'Extent MAE', 'Extent RMSE']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='experiment_results')
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for model, mode, directory in RUNS:
        path = root / directory / 'best_metrics.json'
        if not path.exists():
            continue
        with path.open() as stream:
            metrics = json.load(stream)
        rows.append({
            'Model': model, 'Pseudo-depth': mode,
            'Best Epoch': metrics['epoch'],
            'Car AP3D Easy': metrics['Car_3D_AP40_Easy'],
            'Car AP3D Moderate': metrics['Car_3D_AP40_Moderate'],
            'Car AP3D Hard': metrics['Car_3D_AP40_Hard'],
            'BEV AP40 Moderate': metrics['Car_BEV_AP40_Moderate'],
            'Map Depth MAE': metrics['map_depth_MAE'],
            'Map Depth RMSE': metrics['map_depth_RMSE'],
            'Final Depth MAE': metrics['final_depth_MAE'],
            'Final Depth RMSE': metrics['final_depth_RMSE'],
            'Extent MAE': metrics.get('extent_MAE', ''),
            'Extent RMSE': metrics.get('extent_RMSE', ''),
        })
    if not rows:
        raise SystemExit('No completed best_metrics.json files found.')
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    widths = {field: max(len(field), *(len(f'{row[field]:.4f}') if isinstance(row[field], float)
                                      else len(str(row[field])) for row in rows)) for field in FIELDS}
    print('  '.join(field.ljust(widths[field]) for field in FIELDS))
    for row in rows:
        print('  '.join((f'{row[field]:.4f}' if isinstance(row[field], float)
                         else str(row[field])).ljust(widths[field]) for field in FIELDS))


if __name__ == '__main__':
    main()
