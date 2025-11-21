import torch


class PointCloudEvaluator:
    def __init__(self, num_classes, device, ignore_index=-1):
        self.num_classes = num_classes
        self.device = device
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long, device=self.device)

    def update(self, pred_labels, gt_labels):
        mask = (gt_labels != self.ignore_index)
        pred = pred_labels[mask]
        gt = gt_labels[mask]

        valid = (pred >= 0) & (pred < self.num_classes)
        pred = pred[valid]
        gt = gt[valid]
        
        if len(gt) == 0: return
        
        indices = gt * self.num_classes + pred
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self.confusion_matrix += counts.view(self.num_classes, self.num_classes)

    def compute_metrics(self):
        cm = self.confusion_matrix.float()
        total_correct = torch.diag(cm).sum()
        total = cm.sum()
        oa = total_correct / (total + 1e-6)
        
        intersection = torch.diag(cm)
        union = cm.sum(1) + cm.sum(0) - intersection
        iou = intersection / (union + 1e-6)
        
        class_acc = intersection / (cm.sum(1) + 1e-6)
        
        return {
            "mIoU": iou.mean().item(),
            "OA": oa.item(),
            "mAcc": class_acc.mean().item(),
            "IoU_per_class": iou.cpu().numpy()
        }
