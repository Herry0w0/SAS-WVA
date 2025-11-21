import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch_scatter import scatter_mean
import dgl
import dgl.ops as DOP


class SuperpointDiscriminativeLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 margin_pull_l2: float = 0.01,
                 margin_push_cos: float = 0.2,
                 weight_pull_l2: float = 1.0,
                 weight_push_cos: float = 1.0,
                 ignore_index: int = -1):
        super().__init__()
        assert num_classes > 0, "num_classes 必须是正整数"
        self.num_classes = int(num_classes)
        self.margin_pull_l2 = float(margin_pull_l2)
        self.margin_push_cos = float(margin_push_cos)
        self.weight_pull_l2 = float(weight_pull_l2)
        self.weight_push_cos = float(weight_push_cos)
        self.ignore_index = int(ignore_index)

    @torch.no_grad()
    def _build_point_group_pairs(self, sp_ids_valid, group_id_remapped, unique_group_ids,
                                N_valid, G, device):
        sp_of_j = sp_ids_valid.to(torch.long)                                       # (N_valid,)
        sp_of_k = torch.div(unique_group_ids, self.num_classes, rounding_mode='floor').to(torch.long)  # (G,)

        perm_p = torch.argsort(sp_of_j)             # (N_valid,)
        perm_g = torch.argsort(sp_of_k)             # (G,)

        sp_p_sorted = sp_of_j[perm_p]               # (N_valid,)
        sp_g_sorted = sp_of_k[perm_g]               # (G,)

        M = int(torch.max(sp_p_sorted.max() if sp_p_sorted.numel() else torch.tensor(0, device=device), sp_g_sorted.max() if sp_g_sorted.numel() else torch.tensor(0, device=device)).item()) + 1

        cnt_p = torch.bincount(sp_p_sorted, minlength=M)            # (M,)
        cnt_g = torch.bincount(sp_g_sorted, minlength=M)            # (M,)

        ng_per_point_sorted = cnt_g[sp_p_sorted]                    # (N_valid,)
        if ng_per_point_sorted.sum().item() == 0:
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device), 0)

        rows_sorted = perm_p.repeat_interleave(ng_per_point_sorted)   # (E_all,)

        off_g = torch.empty(M + 1, dtype=torch.long, device=device)
        off_g[0] = 0
        off_g[1:] = torch.cumsum(cnt_g, dim=0)
        start_g_per_point_sorted = off_g[:-1][sp_p_sorted]           # (N_valid,)

        start_rep = start_g_per_point_sorted.repeat_interleave(ng_per_point_sorted)    # (E_all,)

        csum = torch.cumsum(ng_per_point_sorted, dim=0)              # (N_valid,)
        base = torch.repeat_interleave(csum - ng_per_point_sorted, ng_per_point_sorted) # (E_all,)
        local = torch.arange(rows_sorted.numel(), device=device, dtype=torch.long) - base

        take_idx = start_rep + local          # 在 perm_g 上的下标
        cols_sorted = perm_g.take(take_idx)   # (E_all,) -> 全部 (j, 所在超点的所有组)

        self_k_per_point_sorted = group_id_remapped[perm_p] # (N_valid,)
        self_k_rep = self_k_per_point_sorted.repeat_interleave(ng_per_point_sorted)      # (E_all,)
        keep = (cols_sorted != self_k_rep)

        rows_u = rows_sorted[keep]
        cols_u = cols_sorted[keep]
        E_push = int(rows_u.numel())

        return rows_u, cols_u, E_push

    def forward(self, p_fea: torch.Tensor, sp_ids: torch.Tensor, p_labels: torch.Tensor):
        device = p_fea.device

        valid_mask = (p_labels != self.ignore_index)
        N_valid = int(valid_mask.sum().item())
        if N_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        p_fea_valid = p_fea[valid_mask]
        sp_ids_valid = sp_ids[valid_mask].to(torch.long)
        p_labels_valid = p_labels[valid_mask].to(torch.long)

        p_fn = F.normalize(p_fea_valid.to(torch.float32), p=2, dim=-1) # (N_valid, C)

        group_id = sp_ids_valid * self.num_classes + p_labels_valid
        unique_group_ids, group_id_remapped = torch.unique(group_id, return_inverse=True)
        G = int(unique_group_ids.numel())
        C = p_fn.size(1)

        z_unnorm = scatter_mean(p_fn, group_id_remapped, dim=0, dim_size=G)  # (G, C)
        z_k = F.normalize(z_unnorm, p=2, dim=-1)   # (G, C)

        z_k_per_point = z_k[group_id_remapped]   # (N_valid, C)
        l2_dist = torch.norm(p_fn - z_k_per_point, p=2, dim=1)    # (N_valid,)
        loss_pull = torch.clamp(l2_dist - self.margin_pull_l2, min=0).pow(2).mean()

        rows_u, cols_u, E_push = self._build_point_group_pairs(
            sp_ids_valid, group_id_remapped, unique_group_ids, N_valid, G, device
        )

        if E_push == 0:
            loss_push = torch.tensor(0.0, device=device)
        else:
            with autocast(enabled=False):
                g = dgl.heterograph(
                    {('point', 'p2g', 'group'): (rows_u, cols_u)},
                    num_nodes_dict={'point': N_valid, 'group': G},
                    idtype=torch.int32, device=device
                )
                # u_dot_v：在每条边 (j,k') 上计算 <p_fn[j], z_k[k']>，返回形状 (E, 1) 或 (E,)
                cos_sim_push = DOP.u_dot_v(g, p_fn, z_k).reshape(-1) # (E_push,)

            loss_push = torch.clamp(cos_sim_push - self.margin_push_cos, min=0).pow(2).mean()

        total = self.weight_pull_l2 * loss_pull + self.weight_push_cos * loss_push
        return total


class ContrastiveBoundaryLoss(nn.Module):
    def __init__(self, temperature=0.07, ignore_label=-1):
        super().__init__()
        self.temperature = temperature
        self.ignore_label = ignore_label
        self.eps = 1e-8

    def forward(self, edge_index, edge_logits, label):
        N_total = label.size(0)
        device = label.device

        idx_i = edge_index[0] # Source nodes
        idx_j = edge_index[1] # Neighbor nodes
        
        label_i = label[idx_i]
        label_j = label[idx_j]

        valid_mask = (label_i != self.ignore_label) & (label_j != self.ignore_label)
        if valid_mask.sum() == 0:
             return torch.tensor(0.0, device=device, requires_grad=True)
             
        idx_i = idx_i[valid_mask]
        label_i = label_i[valid_mask]
        label_j = label_j[valid_mask]
        edge_logits = edge_logits[valid_mask]

        is_pos = (label_i == label_j) # mask for neighbors with SAME label
        is_neg = ~is_pos              # mask for neighbors with DIFFERENT label

        is_boundary = torch.zeros(N_total, dtype=torch.bool, device=device)

        is_boundary[idx_i[is_neg]] = True

        is_boundary = is_boundary & (label != self.ignore_label)

        num_boundary = is_boundary.sum()
        if num_boundary == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        edge_is_boundary = is_boundary[idx_i]
        
        if edge_is_boundary.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        idx_i_boundary = idx_i[edge_is_boundary]
        is_pos_boundary = is_pos[edge_is_boundary]
        edge_logits_boundary = edge_logits[edge_is_boundary]

        exp_logits = torch.exp(edge_logits_boundary / self.temperature)

        denominator = torch.zeros(N_total, device=device, dtype=exp_logits.dtype)
        denominator.index_add_(0, idx_i_boundary, exp_logits)

        numerator = torch.zeros(N_total, device=device, dtype=exp_logits.dtype)
        numerator.index_add_(0, idx_i_boundary[is_pos_boundary], exp_logits[is_pos_boundary])

        valid_boundary_mask = is_boundary & (denominator > 0)
        if valid_boundary_mask.sum() == 0:
             return torch.tensor(0.0, device=device, requires_grad=True)

        loss_per_node = torch.log(denominator[valid_boundary_mask] + self.eps) - torch.log(numerator[valid_boundary_mask] + self.eps)

        return loss_per_node.mean()
