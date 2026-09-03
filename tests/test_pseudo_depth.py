import unittest

import torch

from lib.models.monodetr.depth_predictor.ddn_loss.ddn_loss import DDNLoss
from lib.models.monodetr.depth_predictor.ddn_loss.focalloss import FocalLoss


class PseudoDepthTargetTest(unittest.TestCase):
    def setUp(self):
        self.ddn = DDNLoss()
        self.logits = torch.zeros(1, 81, 4, 6)
        self.box = torch.tensor([[1., 1., 5., 4.]])

    def test_soft_distribution_and_colliding_bins(self):
        depth = torch.tensor([10.])
        target = self.ddn.build_soft_ncf_target(
            self.logits, self.box, depth, depth, depth, [1], (0.2, 0.6, 0.2))
        self.assertTrue(torch.all(target >= 0))
        self.assertTrue(torch.isfinite(target).all())
        self.assertTrue(torch.allclose(target.sum(1), torch.ones_like(target[:, 0])))
        pixel = target[0, :, 2, 2]
        self.assertEqual(int((pixel > 0).sum()), 1)
        self.assertAlmostEqual(float(pixel.max()), 1.0)

    def test_soft_one_hot_matches_hard_focal(self):
        logits = torch.randn(2, 5, 3, 4)
        hard = torch.randint(0, 5, (2, 3, 4))
        soft = torch.nn.functional.one_hot(hard, 5).permute(0, 3, 1, 2).float()
        focal = FocalLoss(alpha=0.25, gamma=2, reduction='none')
        self.assertTrue(torch.allclose(
            focal(logits, hard), focal(logits, soft), atol=2e-5, rtol=2e-5))

    def test_extent_target_is_non_negative(self):
        prediction = torch.ones(1, 1, 4, 6)
        extent = torch.tensor([4.2])
        target = self.ddn.build_extent_target(
            prediction, self.box, torch.tensor([10.]), extent, [1])
        self.assertGreaterEqual(float(target.min()), 0.0)
        self.assertAlmostEqual(float(target[0, 2, 2]), 4.2, places=5)


if __name__ == '__main__':
    unittest.main()
