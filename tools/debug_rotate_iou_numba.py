"""Minimal import/JIT/kernel smoke test for the official KITTI rotate IoU.

This does not implement or replace any evaluator calculation.  It exercises
the repository's existing CUDA kernel with three small rotated-box cases.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='experiment_results/evaluator_debug')
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    traceback_path = output_dir / 'numba_minimal_traceback.txt'
    result_path = output_dir / 'rotate_iou_synthetic.json'
    result = {'module_import': False, 'kernel_execution': False}
    try:
        from lib.datasets.kitti.kitti_eval_python.rotate_iou import rotate_iou_gpu_eval
        result['module_import'] = True
        boxes = np.asarray([
            [0.0, 0.0, 2.0, 4.0, 0.0],
            [20.0, 20.0, 2.0, 4.0, 0.0],
            [1.0, 0.0, 2.0, 4.0, 0.0],
        ], dtype=np.float32)
        reference = boxes[:1]
        values = rotate_iou_gpu_eval(boxes, reference).reshape(-1)
        result.update({
            'kernel_execution': True,
            'identical_iou': float(values[0]),
            'disjoint_iou': float(values[1]),
            'partial_iou': float(values[2]),
            'checks_passed': bool(
                np.isclose(values[0], 1.0, atol=1e-5)
                and np.isclose(values[1], 0.0, atol=1e-5)
                and 0.0 < values[2] < 1.0),
        })
    except Exception as error:
        trace = traceback.format_exc()
        traceback_path.write_text(trace, encoding='utf-8')
        result.update({'error_type': type(error).__name__, 'error': str(error)})
        print(trace)
    result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result.get('checks_passed') else 1)


if __name__ == '__main__':
    main()
