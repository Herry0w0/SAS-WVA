import torch
from torch_scatter import scatter_add, scatter_max
from pytorch3d.ops import knn_points


@torch.no_grad()
def compute_metrics(
    outputs,
    labels,
    p_xyz,
    point_offsets,
    sp_offsets=None,
    k: int = 8,
    tolerance: int = 1,
    ignore_index: int = -1,
    device: str = "cuda",
):
    """
    oversegmentation metrics: OOA(ASA), BR, BP — batched KNN + symmetric tolerance.

    支持两种 outputs 结构：
    1) 稀疏指派（旧版）:
        outputs['asso_data'] = (edge_index, final_asso)
        - edge_index: (2, E_total) [point_id, superpoint_id]
        - final_asso: (E_total,)   对应边的分数
        - sp_offsets:  (B+1,) 可由参数或 outputs['sp_offsets'] 提供
    2) 稠密指派（新版）:
        outputs['asso_data'] = pt2sp_global，形状 (N_total,)
        - 直接给每个点的超点ID（可为 -1 表示未分配）
        - 可给 outputs['num_superpoints']（整型或长度为 B 的列表/张量），但不是必需
        - sp_offsets 可省略（函数内部按 batch 自行做局部唯一化映射）

    Args:
        labels: (N_total,) LongTensor 语义标签（可含 ignore_index）
        p_xyz:  (N_total, 3) float32 点坐标
        point_offsets: (B+1,) 每个 batch 的点起止偏移
        sp_offsets:    (B+1,) 每个 batch 的超点起止偏移（仅稀疏模式需要）
        k: KNN 的 K
        tolerance: 边界匹配的对称“膨胀”跳数（0 表示不膨胀）
        ignore_index: 忽略的标签值
        device: 'cuda' or 'cpu'

    Returns:
        dict: {'OOA': float, 'BR': float, 'BP': float,
               'counts': {'ooa_num': float, 'ooa_den': int, 'br_num': int, 'br_den': int, 'bp_num': int, 'bp_den': int}}
    """
    point_offsets = torch.as_tensor(point_offsets, dtype=torch.long, device=device)
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    p_xyz = torch.as_tensor(p_xyz, dtype=torch.float32, device=device)

    B = int(point_offsets.numel() - 1)

    asso = outputs["asso_data"]
    is_dense_assign = torch.is_tensor(asso) and asso.dim() == 1
    is_sparse_assign = isinstance(asso, (tuple, list)) and len(asso) >= 2

    if not (is_dense_assign or is_sparse_assign):
        raise ValueError(
            "Unsupported outputs['asso_data'] format. "
            "Expect 1D tensor for dense assignment, or (edge_index, scores) for sparse."
        )

    if is_sparse_assign:
        if sp_offsets is None:
            if "sp_offsets" in outputs:
                sp_offsets = outputs["sp_offsets"]
            else:
                raise ValueError("Sparse 'asso_data' requires 'sp_offsets' (argument or outputs['sp_offsets']).")
        sp_offsets = torch.as_tensor(sp_offsets, dtype=torch.long, device=device)
        edge_index = asso[0].to(device)
        final_asso = asso[1].to(device)
    else:
        pt2sp_global = torch.as_tensor(asso, dtype=torch.long, device=device)

    def _safe_point_to_sp(local_asso, local_p_idx, local_sp_idx, n_batch):
        # Per point argmax over incident edges; points without edges -> -1
        conf, max_edge = scatter_max(local_asso, local_p_idx, dim=0, dim_size=n_batch)
        valid = torch.isfinite(conf)
        pt2sp = local_sp_idx.new_full((n_batch,), -1)
        if valid.any():
            pt2sp[valid] = local_sp_idx[max_edge[valid]]
        return pt2sp

    def _remap_dense(lbl):
        valid = lbl != ignore_index
        if valid.any():
            uniq, inv = torch.unique(lbl[valid], sorted=True, return_inverse=True)
            dense = torch.full_like(lbl, -1)
            dense[valid] = inv
            return dense, int(uniq.numel())
        else:
            return torch.full_like(lbl, -1), 0

    def _dilate_edges(mark_e: torch.Tensor, edges: torch.Tensor, num_nodes: int, hops: int):
        # Hop-based dilation on an edge mask (E,)
        if hops <= 0 or mark_e.numel() == 0:
            return mark_e
        dil = mark_e.clone()
        for _ in range(hops):
            nodes_src = scatter_add(dil.float(), edges[0], dim=0, dim_size=num_nodes) > 0
            nodes_dst = scatter_add(dil.float(), edges[1], dim=0, dim_size=num_nodes) > 0
            nodes = nodes_src | nodes_dst
            new_edges = nodes[edges[0]] | nodes[edges[1]]
            dil = dil | new_edges
        return dil

    lengths = point_offsets[1:] - point_offsets[:-1]
    max_N = int(lengths.max().item()) if B > 0 else 0

    # padded buffers for batched KNN
    padded_xyz = torch.zeros(B, max_N, 3, device=device)
    padded_labels = torch.full((B, max_N), ignore_index, dtype=torch.long, device=device)
    padded_pt2sp = torch.full((B, max_N), -1, dtype=torch.long, device=device)
    M_batches = torch.zeros(B, dtype=torch.long, device=device)

    # global accumulators (weighted)
    tot = {"ooa_num": 0.0, "ooa_den": 0, "br_num": 0, "br_den": 0, "bp_num": 0, "bp_den": 0}

    for b in range(B):
        pt_beg, pt_end = point_offsets[b].item(), point_offsets[b + 1].item()
        N_batch = pt_end - pt_beg
        if N_batch <= 0:
            continue

        padded_xyz[b, :N_batch] = p_xyz[pt_beg:pt_end]
        padded_labels[b, :N_batch] = labels[pt_beg:pt_end]

        if is_sparse_assign:
            sp_beg, sp_end = sp_offsets[b].item(), sp_offsets[b + 1].item()
            M_batch = sp_end - sp_beg
            if M_batch <= 0:
                continue
            M_batches[b] = M_batch

            mask = (edge_index[0] >= pt_beg) & (edge_index[0] < pt_end) & \
                   (edge_index[1] >= sp_beg) & (edge_index[1] < sp_end)
            if mask.any():
                local_p_idx = edge_index[0][mask] - pt_beg
                local_sp_idx = edge_index[1][mask] - sp_beg
                local_asso = final_asso[mask]
                pt2sp_local = _safe_point_to_sp(local_asso, local_p_idx, local_sp_idx, N_batch)
                padded_pt2sp[b, :N_batch] = pt2sp_local
            else:
                pt2sp_local = padded_pt2sp[b, :N_batch]  # 全 -1

        else:
            pt2sp_g = pt2sp_global[pt_beg:pt_end]  # (N_batch,)
            valid = pt2sp_g >= 0
            if valid.any():
                uniq, inv = torch.unique(pt2sp_g[valid], sorted=True, return_inverse=True)
                pt2sp_local = pt2sp_g.new_full((N_batch,), -1)
                pt2sp_local[valid] = inv  # [0..M_b-1]
                M_b = int(uniq.numel())
            else:
                pt2sp_local = pt2sp_g.new_full((N_batch,), -1)
                M_b = 0
            padded_pt2sp[b, :N_batch] = pt2sp_local
            M_batches[b] = M_b

        # ---- OOA/ASA ----
        labels_b = padded_labels[b, :N_batch]
        labels_dense_b, C_b = _remap_dense(labels_b)
        valid_pts = (labels_dense_b >= 0) & (pt2sp_local >= 0)
        if valid_pts.any() and C_b > 0 and int(M_batches[b].item()) > 0:
            lbl_v = labels_dense_b[valid_pts]  # (P,)
            sp_v  = pt2sp_local[valid_pts]     # (P,) in [0..M_b-1]
            # hash pairs -> bincount then reshape to (M_b, C_b)
            M_b = int(M_batches[b].item())
            idx = (sp_v * C_b + lbl_v).to(torch.long)
            counts = torch.bincount(idx, minlength=M_b * C_b)
            counts = counts.view(M_b, C_b)
            per_sp_max = counts.max(dim=1)[0]
            tot["ooa_num"] += float(per_sp_max.sum().item())
            tot["ooa_den"] += int(valid_pts.sum().item())

    if B > 0 and max_N > 0:
        knn = knn_points(padded_xyz, padded_xyz, lengths1=lengths, lengths2=lengths, K=k)
        idx_all = knn.idx  # (B, max_N, k)
    else:
        idx_all = None

    for b in range(B):
        N_batch = int(lengths[b].item())
        if N_batch <= 0 or int(M_batches[b].item()) <= 0:
            continue
        labels_b = padded_labels[b, :N_batch]
        labels_dense_b, _ = _remap_dense(labels_b)
        pt2sp_b = padded_pt2sp[b, :N_batch]

        # undirected KNN edges: keep dst>src to dedup
        idx_b = idx_all[b, :N_batch]  # (N,k)
        src = torch.arange(N_batch, device=device).unsqueeze(1).expand(N_batch, k).reshape(-1)
        dst = idx_b.reshape(-1)
        mask = (dst >= 0) & (dst != src) & (dst > src)
        if not mask.any():
            continue
        src, dst = src[mask], dst[mask]
        edges = torch.stack([src, dst], dim=0)

        # GT boundary edges
        li, lj = labels_dense_b[edges[0]], labels_dense_b[edges[1]]
        real_inter = (li >= 0) & (lj >= 0) & (li != lj)

        # Pred boundary edges
        si, sj = pt2sp_b[edges[0]], pt2sp_b[edges[1]]
        pred_inter = (si >= 0) & (sj >= 0) & (si != sj)

        # symmetric dilation for tolerance
        dil_gt   = _dilate_edges(real_inter, edges, N_batch, tolerance)
        dil_pred = _dilate_edges(pred_inter, edges, N_batch, tolerance)

        # counts
        tot["br_num"] += int((real_inter & dil_pred).sum().item())
        tot["br_den"] += int(real_inter.sum().item())
        tot["bp_num"] += int((pred_inter & dil_gt).sum().item())
        tot["bp_den"] += int(pred_inter.sum().item())

    OOA = float(tot["ooa_num"] / tot["ooa_den"]) if tot["ooa_den"] > 0 else 0.0
    BR  = float(tot["br_num"]  / tot["br_den"])  if tot["br_den"]  > 0 else 0.0
    BP  = float(tot["bp_num"]  / tot["bp_den"])  if tot["bp_den"]  > 0 else 0.0

    return {"OOA": OOA, "BR": BR, "BP": BP, "counts": tot}
