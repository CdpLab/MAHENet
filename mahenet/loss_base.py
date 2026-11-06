import torch
from torch import nn
from torch.nn import functional as F


def segmentation_edges(labels, ignore_index=255):
    """Derive class-transition boundaries directly from discrete segmentation labels."""
    valid = labels != ignore_index
    edge = torch.zeros_like(labels, dtype=torch.bool)
    edge[:, 1:] |= valid[:, 1:] & valid[:, :-1] & (labels[:, 1:] != labels[:, :-1])
    edge[:, :-1] |= valid[:, 1:] & valid[:, :-1] & (labels[:, 1:] != labels[:, :-1])
    edge[:, :, 1:] |= valid[:, :, 1:] & valid[:, :, :-1] & (labels[:, :, 1:] != labels[:, :, :-1])
    edge[:, :, :-1] |= valid[:, :, 1:] & valid[:, :, :-1] & (labels[:, :, 1:] != labels[:, :, :-1])
    return edge.float().unsqueeze(1)


def edge_targets(labels, output_size, ignore_index=255):
    edge = F.interpolate(segmentation_edges(labels, ignore_index), size=output_size, mode="nearest")
    valid = F.interpolate((labels != ignore_index).float().unsqueeze(1), size=output_size, mode="nearest").bool()
    # Avoid supervising cells adjacent to padded or unlabeled pixels.
    valid = F.max_pool2d((~valid).float(), 3, stride=1, padding=1) == 0
    return edge, valid


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, eps=1e-6):
        super().__init__()
        self.num_classes, self.ignore_index, self.eps = num_classes, ignore_index, eps

    def forward(self, logits, labels):
        valid = labels != self.ignore_index
        safe_labels = labels.masked_fill(~valid, 0)
        probabilities = logits.softmax(dim=1)
        target = F.one_hot(safe_labels, self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        probabilities, target = probabilities * valid, target * valid
        intersection = (probabilities * target).sum((0, 2, 3))
        denominator = probabilities.square().sum((0, 2, 3)) + target.square().sum((0, 2, 3))
        present = target.sum((0, 2, 3)) > 0
        if not present.any():
            return logits.sum() * 0
        return (1 - (2 * intersection[present] + self.eps) / (denominator[present] + self.eps)).mean()


class JointLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, edge_weight=0.25, ignore_index=255):
        super().__init__()
        self.alpha, self.edge_weight, self.ignore_index = alpha, edge_weight, ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index)

    def forward(self, logits, edge_logits, labels):
        # Reductions stay fp32 under AMP to avoid Dice/BCE overflow.
        logits, edge_logits = logits.float(), edge_logits.float()
        valid = labels != self.ignore_index
        if not valid.any():
            zero = logits.sum() * 0
            return zero, {"ce": zero.detach(), "dice": zero.detach(), "edge": zero.detach()}
        ce = self.cross_entropy(logits, labels)
        dice = self.dice(logits, labels) if self.alpha < 1 else logits.sum() * 0
        if self.edge_weight:
            edge_target, edge_valid = edge_targets(labels, edge_logits.shape[-2:], self.ignore_index)
            positives = edge_target[edge_valid].sum()
            negatives = edge_valid.sum() - positives
            pos_weight = (negatives / positives.clamp_min(1)).clamp(max=100)
            edge_loss = F.binary_cross_entropy_with_logits(
                edge_logits[edge_valid], edge_target[edge_valid], pos_weight=pos_weight
            )
        else:
            # Preserve a zero-gradient path through the edge head. Without it,
            # DDP receives edge_logits as a model output but no edge-head
            # gradient when edge supervision is disabled.
            edge_loss = edge_logits.sum() * 0
        total = self.alpha * ce + (1 - self.alpha) * dice + self.edge_weight * edge_loss
        return total, {"ce": ce.detach(), "dice": dice.detach(), "edge": edge_loss.detach()}
