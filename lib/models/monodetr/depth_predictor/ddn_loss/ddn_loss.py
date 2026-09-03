import torch
import torch.nn as nn
import math

from .balancer import Balancer, compute_fg_mask
from .focalloss import FocalLoss

# based on:
# https://github.com/TRAILab/CaDDN/blob/master/pcdet/models/backbones_3d/ffe/ddn_loss/ddn_loss.py


class DDNLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 fg_weight=13,
                 bg_weight=1,
                 downsample_factor=1):
        """
        Initializes DDNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            disc_cfg [dict]: Depth discretiziation configuration
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.balancer = Balancer(
            downsample_factor=downsample_factor,
            fg_weight=fg_weight,
            bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")

    def build_target_depth_from_3dcenter(self, depth_logits, gt_boxes2d, gt_center_depth, num_gt_per_img):
        B, _, H, W = depth_logits.shape
        depth_maps = torch.zeros((B, H, W), device=depth_logits.device, dtype=depth_logits.dtype)

        # Set box corners
        gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
        gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
        gt_boxes2d = gt_boxes2d.long()

        # Set all values within each box to True
        gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)
        gt_center_depth = gt_center_depth.split(num_gt_per_img, dim=0)
        B = len(gt_boxes2d)
        for b in range(B):
            center_depth_per_batch = gt_center_depth[b]
            center_depth_per_batch, sorted_idx = torch.sort(center_depth_per_batch, dim=0, descending=True)
            gt_boxes_per_batch = gt_boxes2d[b][sorted_idx]
            for n in range(gt_boxes_per_batch.shape[0]):
                u1, v1, u2, v2 = gt_boxes_per_batch[n]
                depth_maps[b, v1:v2, u1:u2] = center_depth_per_batch[n]

        return depth_maps

    def build_soft_ncf_target(self, depth_logits, gt_boxes2d, gt_near_depth,
                              gt_center_depth, gt_far_depth, num_gt_per_img,
                              weights=(0.2, 0.6, 0.2)):
        """Build per-pixel near/center/far depth-bin distributions.

        Background remains the original final depth class. Objects use the
        baseline far-to-near painter order, so overlap semantics are unchanged.
        ``scatter_add_`` naturally merges coincident LID bins.
        """
        B, C, H, W = depth_logits.shape
        num_bins = C - 1
        target = depth_logits.new_zeros((B, C, H, W))
        target[:, num_bins] = 1.0

        boxes = gt_boxes2d.clone()
        boxes[:, :2] = torch.floor(boxes[:, :2])
        boxes[:, 2:] = torch.ceil(boxes[:, 2:])
        boxes = boxes.long()
        bins = torch.stack([
            self.bin_depths(gt_near_depth, num_bins=num_bins, target=True),
            self.bin_depths(gt_center_depth, num_bins=num_bins, target=True),
            self.bin_depths(gt_far_depth, num_bins=num_bins, target=True),
        ], dim=1)
        bin_weights = depth_logits.new_tensor(weights)
        if not torch.isclose(bin_weights.sum(), bin_weights.new_tensor(1.0)):
            raise ValueError(f'soft NCF weights must sum to 1, got {weights}')
        if torch.any(bin_weights < 0):
            raise ValueError(f'soft NCF weights must be non-negative, got {weights}')

        box_batches = boxes.split(num_gt_per_img, dim=0)
        center_batches = gt_center_depth.split(num_gt_per_img, dim=0)
        bin_batches = bins.split(num_gt_per_img, dim=0)
        for b, (batch_boxes, batch_centers, batch_bins) in enumerate(
                zip(box_batches, center_batches, bin_batches)):
            _, order = torch.sort(batch_centers, descending=True)
            for box, object_bins in zip(batch_boxes[order], batch_bins[order]):
                distribution = depth_logits.new_zeros(C)
                distribution.scatter_add_(0, object_bins, bin_weights)
                u1, v1, u2, v2 = box
                target[b, :, v1:v2, u1:u2] = distribution[:, None, None]
        return target

    def build_extent_target(self, extent_logits, gt_boxes2d, gt_center_depth,
                            gt_depth_extent, num_gt_per_img):
        """Paint scalar extent using the exact baseline overlap ordering."""
        B, _, H, W = extent_logits.shape
        target = extent_logits.new_zeros((B, H, W))
        boxes = gt_boxes2d.clone()
        boxes[:, :2] = torch.floor(boxes[:, :2])
        boxes[:, 2:] = torch.ceil(boxes[:, 2:])
        boxes = boxes.long()
        box_batches = boxes.split(num_gt_per_img, dim=0)
        center_batches = gt_center_depth.split(num_gt_per_img, dim=0)
        extent_batches = gt_depth_extent.split(num_gt_per_img, dim=0)
        for b, (batch_boxes, batch_centers, batch_extents) in enumerate(
                zip(box_batches, center_batches, extent_batches)):
            _, order = torch.sort(batch_centers, descending=True)
            for box, extent in zip(batch_boxes[order], batch_extents[order]):
                u1, v1, u2, v2 = box
                target[b, v1:v2, u1:u2] = extent
        return target

    def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=60, num_bins=80, target=False):
        """
        Converts depth map into bin indices
        Args:
            depth_map [torch.Tensor(H, W)]: Depth Map
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            indices [torch.Tensor(H, W)]: Depth bin indices
        """
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        elif mode == "SID":
            indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
                      (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)
       
        return indices

    def forward(self, depth_logits, gt_boxes2d, num_gt_per_img, gt_center_depth,
                mode='center', gt_near_depth=None, gt_far_depth=None,
                soft_weights=(0.2, 0.6, 0.2)):
        """
        Gets depth_map loss
        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
            num_gt_per_img:
            gt_center_depth:
        Returns:
            loss [torch.Tensor(1)]: Depth classification network loss
        """

        if mode in ('center', 'center_extent'):
            # Keep the original hard-label path byte-for-byte equivalent.
            depth_maps = self.build_target_depth_from_3dcenter(
                depth_logits, gt_boxes2d, gt_center_depth, num_gt_per_img)
            depth_target = self.bin_depths(depth_maps, target=True)
            loss = self.loss_func(depth_logits, depth_target)
        elif mode == 'soft_ncf':
            if gt_near_depth is None or gt_far_depth is None:
                raise ValueError('soft_ncf requires near and far depth targets')
            depth_target = self.build_soft_ncf_target(
                depth_logits, gt_boxes2d, gt_near_depth, gt_center_depth,
                gt_far_depth, num_gt_per_img, soft_weights)
            loss = self.loss_func(depth_logits, depth_target)
        else:
            raise ValueError(f'Unknown pseudo depth mode: {mode}')
        #ipdb.set_trace()
        # Compute foreground/background balancing
        loss = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)

        return loss

    def extent_loss(self, extent_prediction, gt_boxes2d, num_gt_per_img,
                    gt_center_depth, gt_depth_extent):
        target = self.build_extent_target(
            extent_prediction, gt_boxes2d, gt_center_depth,
            gt_depth_extent, num_gt_per_img)
        prediction = extent_prediction.squeeze(1)
        loss_map = torch.nn.functional.smooth_l1_loss(
            prediction, target, reduction='none')
        fg_mask = compute_fg_mask(
            gt_boxes2d.clone(), loss_map.shape, num_gt_per_img,
            downsample_factor=1, device=loss_map.device)
        if not torch.any(fg_mask):
            return prediction.sum() * 0.0
        return loss_map[fg_mask].mean()
