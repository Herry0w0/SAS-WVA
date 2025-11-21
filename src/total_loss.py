import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from .loss_func import (
    SuperpointDiscriminativeLoss,
    ContrastiveBoundaryLoss
)


class TotalSemanticLoss(nn.Module):
    """
    总语义损失函数 (Total Semantic Loss)
    它结合了三种损失：
    1. CrossEntropy Loss (CE): 用于点级别的语义分割 (基于 seg_logits)。
    2. ContrastiveBoundaryLoss (CBL): 在 K-NN 边上运行，强制边界清晰。
    3. SuperpointDiscriminativeLoss (SP Disc): 强制超点内部特征一致 (pull) 且
       不同语义部分的超点特征分离 (push)。
    """
    def __init__(self, 
                 num_classes: int,
                 ignore_label: int = -1,
                 w_ce: float = 1.0, 
                 w_cbl: float = 1.0,
                 w_sp_disc: float = 1.0,
                 sp_margin_pull: float = 0.01,
                 sp_margin_push: float = 0.2,
                 sp_weight_pull: float = 1.0,
                 sp_weight_push: float = 1.0,
                 cbl_temperature: float = 0.3,
                 ce_class_weights: torch.Tensor = None
                 ):
        """
        Args:
            num_classes (int): 语义类别的总数。
            ignore_label (int): 应被所有损失忽略的标签索引。
            w_ce (float): 交叉熵损失的权重。
            w_cbl (float): 对比边界损失的权重。
            w_sp_disc (float): 超点判别损失的权重。
            sp_margin_pull (float): L_pull 的 L2 阈值。
            sp_margin_push (float): L_push 的余弦相似度阈值。
            sp_weight_pull (float): L_pull 的内部权重。
            sp_weight_push (float): L_push 的内部权重。
            cbl_temperature (float): CBL 的温度参数。
            ce_class_weights (torch.Tensor): 交叉熵的类别权重 (可选)。
        """
        super().__init__()
        
        self.ignore_label = ignore_label

        self.w_ce = w_ce
        self.w_cbl = w_cbl
        self.w_sp_disc = w_sp_disc
        self.cbl_temperature = cbl_temperature

        self.ce_loss = nn.CrossEntropyLoss(
            weight=ce_class_weights, 
            ignore_index=self.ignore_label
        )

        self.cbl_loss = ContrastiveBoundaryLoss(
            temperature=self.cbl_temperature,
            ignore_label=self.ignore_label
        )

        self.sp_disc_loss = SuperpointDiscriminativeLoss(
            num_classes=num_classes,
            margin_pull_l2=sp_margin_pull,
            margin_push_cos=sp_margin_push,
            weight_pull_l2=sp_weight_pull,
            weight_push_cos=sp_weight_push,
            ignore_index=self.ignore_label
        )

    def forward(self, results: dict, labels: torch.Tensor):
        device = labels.device

        valid_mask = (labels != self.ignore_label)
        if valid_mask.sum() == 0:
            loss_dict = {
                'total_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'ce_loss': torch.tensor(0.0, device=device),
                'cbl_loss': torch.tensor(0.0, device=device),
                'sp_disc_loss': torch.tensor(0.0, device=device),
            }
            return loss_dict

        seg_logits = results.get('seg_logits') 
        if seg_logits is not None:
            loss_ce = self.ce_loss(seg_logits, labels)
        else:
            if results.get('seg_probs') is not None:
                seg_probs = results.get('seg_probs')     # (N_total, num_classes)
                eps = 1e-12
                log_probs = (seg_probs + eps).log()

                loss_ce = F.nll_loss(
                    log_probs,
                    labels,
                    ignore_index=self.ignore_label
                )
            else:
                loss_ce = torch.tensor(0.0, device=device)

        edge_index = results.get('knn_edge_index')
        edge_logits = results.get('knn_edge_logits')
        
        if edge_index is not None and edge_logits is not None:
            loss_cbl = self.cbl_loss(edge_index, edge_logits, labels)
        else:
            loss_cbl = torch.tensor(0.0, device=device)

        p_fea = results.get('p_fea') 
        sp_ids = results.get('asso_data')
        cc_ids = results.get('cc_ids')

        if p_fea is not None and sp_ids is not None and cc_ids is not None:
            loss_sp_disc = self.sp_disc_loss(
                p_fea, 
                sp_ids, 
                labels 
            )
            loss_cc_disc = self.sp_disc_loss(
                p_fea,
                cc_ids,
                labels
            )
            loss_disc = loss_sp_disc + loss_cc_disc
        else:
            loss_disc = torch.tensor(0.0, device=device)

        total_loss = (
            self.w_ce * loss_ce +
            self.w_cbl * loss_cbl +
            self.w_sp_disc * loss_disc
        )

        loss_dict = {
            'total_loss': total_loss,
            'ce_loss': loss_ce.detach(),
            'cbl_loss': loss_cbl.detach(),
            'sp_disc_loss': loss_sp_disc.detach(),
        }
        
        return loss_dict
