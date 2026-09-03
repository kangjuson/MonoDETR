import math
import unittest

import torch

from tools.run_pseudo_depth_experiment import (
    BASE_COLUMNS,
    DepthMetricAccumulator,
    KITTI_AP_UNIT,
    Phase0Controller,
    error_summary,
    parse_car_ap40,
)


class ExperimentRunnerTest(unittest.TestCase):
    def test_parse_car_ap40_uses_exact_named_keys(self):
        ap = {
            'Car_3d_easy_R40': 1.0,
            'Car_3d_moderate_R40': 2.0,
            'Car_3d_hard_R40': 3.0,
            'Car_bev_easy_R40': 4.0,
            'Car_bev_moderate_R40': 5.0,
            'Car_bev_hard_R40': 6.0,
            'unrelated': 999.0,
        }
        parsed = parse_car_ap40(ap)
        self.assertEqual(parsed['Car_3D_AP40_Moderate'], 2.0)
        self.assertEqual(parsed['Car_BEV_AP40_Moderate'], 5.0)

    def test_official_ap_points_are_not_rescaled(self):
        raw = {
            'Car_3d_easy_R40': 0.2622207,
            'Car_3d_moderate_R40': 0.2163629,
            'Car_3d_hard_R40': 0.1906331,
            'Car_bev_easy_R40': 1.2582420,
            'Car_bev_moderate_R40': 1.0653365,
            'Car_bev_hard_R40': 0.8068312,
        }
        parsed = parse_car_ap40(raw)
        self.assertEqual(parsed['Car_3D_AP40_Moderate'], 0.2163629)

    def test_min_delta_boundary_and_absolute_best(self):
        controller = Phase0Controller(min_delta=0.5)
        decisions = [controller.step(value, 5 * (index + 1))
                     for index, value in enumerate(
                         [20.00, 20.20, 20.40, 20.49, 20.50, 20.70])]
        self.assertEqual([item['significant'] for item in decisions],
                         [True, False, False, False, True, False])
        self.assertEqual(controller.significant_best, 20.50)
        self.assertEqual(controller.absolute_best, 20.70)
        self.assertEqual(controller.absolute_best_epoch, 30)

    def test_ap_point_scheduler_sequence(self):
        controller = Phase0Controller(min_delta=0.5)
        values = [18.0, 18.3, 18.4, 18.6, 18.8, 19.1, 19.2, 19.3]
        decisions = [controller.step(value, 5 * (index + 1))
                     for index, value in enumerate(values)]
        self.assertFalse(decisions[1]['significant'])
        self.assertFalse(decisions[2]['significant'])
        self.assertTrue(decisions[3]['significant'])
        self.assertTrue(decisions[5]['significant'])
        self.assertEqual(controller.significant_best, 19.1)
        self.assertEqual(controller.absolute_best, 19.3)

    def test_csv_and_json_ap_unit_label(self):
        self.assertIn('AP_unit', BASE_COLUMNS)
        self.assertEqual(KITTI_AP_UNIT, 'AP points (0-100)')

    def test_phase0_scheduler_reduces_resets_then_stops_at_min_lr(self):
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([parameter], lr=2e-4)
        controller = Phase0Controller(min_delta=0.5, lr_patience=2,
                                      early_patience=6, lr_factor=0.1,
                                      min_lr=1e-6)
        sequence = [10.0] + [10.1] * 12
        decisions = []
        for index, ap in enumerate(sequence):
            decision = controller.step(ap, 5 * (index + 1), optimizer)
            decisions.append(decision)
            if decision['should_stop']:
                break

        self.assertEqual(controller.absolute_best, 10.1)
        self.assertEqual(controller.significant_best, 10.0)
        self.assertEqual(controller.lr_reduction_epochs, [15, 25, 35])
        self.assertTrue(math.isclose(optimizer.param_groups[0]['lr'], 1e-6))
        self.assertTrue(decisions[-1]['should_stop'])
        self.assertEqual(controller.early_bad_count, 6)
        self.assertEqual(5 * len(decisions), 65)

    def test_depth_metric_values_and_distance_buckets(self):
        metrics = error_summary('final_depth', [11.0, 38.0, 55.0],
                                [10.0, 40.0, 50.0], [10.0, 40.0, 50.0])
        self.assertTrue(math.isclose(metrics['final_depth_MAE'], 8 / 3))
        self.assertTrue(math.isclose(metrics['final_depth_RMSE'], math.sqrt(10)))
        self.assertEqual(metrics['final_depth_MAE_0_20'], 1.0)
        self.assertTrue(math.isnan(metrics['final_depth_MAE_20_40']))
        self.assertEqual(metrics['final_depth_MAE_40_plus'], 3.5)

    def test_extent_metric_values(self):
        accumulator = DepthMetricAccumulator()
        accumulator.extent_predictions = [4.0, 4.0, 7.0]
        accumulator.extent_targets = [4.0, 5.0, 6.0]
        metrics = accumulator.metrics()
        self.assertTrue(math.isclose(metrics['extent_MAE'], 2 / 3))
        self.assertTrue(math.isclose(metrics['extent_RMSE'], math.sqrt(2 / 3)))
        self.assertEqual(metrics['extent_GT_mean'], 5.0)
        self.assertEqual(metrics['extent_pred_mean'], 5.0)


if __name__ == '__main__':
    unittest.main()
