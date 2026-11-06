import torch
from torch.nn import functional as F

from .loss_base import segmentation_edges


class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255, boundary_tolerance=2):
        self.num_classes, self.ignore_index, self.tolerance = num_classes, ignore_index, boundary_tolerance
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        self.gt_pixels = torch.zeros(num_classes, dtype=torch.long)
        self.predicted_pixels = torch.zeros(num_classes, dtype=torch.long)
        self.boundary_pred_matched = self.boundary_gt_matched = 0
        self.boundary_pred = self.boundary_gt = 0

    @torch.no_grad()
    def update(self, predictions, labels):
        if predictions.ndim == labels.ndim + 1:
            predictions = predictions.argmax(dim=1)
        valid = labels != self.ignore_index
        encoded = self.num_classes * labels[valid] + predictions[valid]
        self.confusion += torch.bincount(encoded.cpu(), minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.gt_pixels += torch.bincount(labels[valid].cpu(), minlength=self.num_classes)
        self.predicted_pixels += torch.bincount(predictions[valid].cpu(), minlength=self.num_classes)

        predictions = predictions.masked_fill(~valid, self.ignore_index)
        pred_edge = segmentation_edges(predictions, self.ignore_index).bool()
        gt_edge = segmentation_edges(labels, self.ignore_index).bool()
        kernel = 2 * self.tolerance + 1
        pred_match = pred_edge & (F.max_pool2d(gt_edge.float(), kernel, stride=1, padding=self.tolerance) > 0)
        gt_match = gt_edge & (F.max_pool2d(pred_edge.float(), kernel, stride=1, padding=self.tolerance) > 0)
        self.boundary_pred_matched += int(pred_match.sum())
        self.boundary_gt_matched += int(gt_match.sum())
        self.boundary_pred += int(pred_edge.sum())
        self.boundary_gt += int(gt_edge.sum())

    def compute(self):
        matrix = self.confusion.float()
        tp = matrix.diag()
        actual, predicted = matrix.sum(1), matrix.sum(0)
        iou = tp / (actual + predicted - tp).clamp_min(1)
        accuracy = tp / actual.clamp_min(1)
        present = actual > 0
        precision = self.boundary_pred_matched / max(self.boundary_pred, 1)
        recall = self.boundary_gt_matched / max(self.boundary_gt, 1)
        return {
            "miou": float(iou[present].mean()),
            "macc": float(accuracy[present].mean()),
            "pacc": float(tp.sum() / matrix.sum().clamp_min(1)),
            "bf1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "per_class_iou": [float(value) if is_present else None for value, is_present in zip(iou, present)],
            "per_class_acc": [float(value) if is_present else None for value, is_present in zip(accuracy, present)],
            "gt_pixels": self.gt_pixels.tolist(),
            "predicted_pixels": self.predicted_pixels.tolist(),
        }
