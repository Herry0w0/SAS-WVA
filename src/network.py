import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch_scatter import scatter, scatter_min, scatter_mean
from pytorch3d.ops import knn_points, sample_farthest_points
import dgl
import dgl.ops as DOP

from .backbone import PointTransformerV3
from .dataset.transform import Compose, ToTensor

# Ampere/Hopper 上启用 TF32（对 Linear/Conv/SDPA 友好）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 让 PyTorch 倾向用更快的 matmul 实现（含 TF32）
torch.set_float32_matmul_precision("high")


@torch.no_grad()
def compute_knn(points_coord, points_offsets, k_p):
    device = points_coord.device
    N_total = points_coord.size(0)
    B = len(points_offsets) - 1

    lens_p = torch.from_numpy(points_offsets[1:] - points_offsets[:-1]).to(device, dtype=torch.long)
    P_max = lens_p.max().item()

    P = torch.zeros((B, P_max, points_coord.size(1)), device=device, dtype=points_coord.dtype)
    if N_total > 0:
        p_batch_idx = torch.arange(B, device=device).repeat_interleave(lens_p)
        p_offsets_tensor = torch.from_numpy(points_offsets[:-1]).to(device, dtype=torch.long)
        p_local_idx = torch.arange(N_total, device=device) - p_offsets_tensor[p_batch_idx]
        P[p_batch_idx, p_local_idx] = points_coord

    if N_total > 0:
        Kp = int(k_p)
        K = min(Kp + 1, P_max)
        _, idx_p2p, _ = knn_points(P, P, K=K, lengths1=lens_p, lengths2=lens_p, return_nn=False, return_sorted=True)  # (B, P_max, K)

        p_mask = torch.arange(P_max, device=device)[None, :] < lens_p[:, None]  # (B, P_max)
        local_indices = idx_p2p[p_mask]  # (N_total, K)

        self_local = p_local_idx.unsqueeze(1).expand(-1, K)            # (N_total, K)
        self_mask = (local_indices == self_local)                     # (N_total, K)

        self_pos = torch.argmax(self_mask.to(torch.int32), dim=1)    # (N_total,)

        all_cols = torch.arange(K, device=device).view(1, -1).expand(N_total, -1)  # (N_total,K)
        drop_mask = all_cols == self_pos.unsqueeze(1)               # (N_total,K)

        nonself_cols = all_cols[~drop_mask].view(N_total, K - 1)
        local_neighbors = torch.gather(local_indices, 1, nonself_cols)  # (N_total, K-1)

        valid_k_per_batch = torch.minimum(lens_p - 1, torch.tensor(Kp, device=device))

        num_neighbors = K - 1
        valid_neighbor_mask = torch.arange(num_neighbors, device=device).unsqueeze(0) < valid_k_per_batch[p_batch_idx].unsqueeze(1)
        p_offsets_broadcast = p_offsets_tensor[p_batch_idx].unsqueeze(1)
        global_neighbors = local_neighbors + p_offsets_broadcast
        knn_p2p = torch.full((N_total, k_p), -1, dtype=torch.long, device=device)
        knn_p2p[:, :num_neighbors] = torch.where(valid_neighbor_mask, global_neighbors, -1)

        src_p = torch.arange(N_total, device=device, dtype=torch.long).unsqueeze(1).expand(-1, num_neighbors)
        edge_p2p_all = torch.stack([src_p[valid_neighbor_mask], global_neighbors[valid_neighbor_mask]], dim=0)

        assert (edge_p2p_all[0] != edge_p2p_all[1]).all().item()

    edge_index_coo = torch.sparse_coo_tensor(edge_p2p_all, torch.ones(edge_p2p_all.size(1), device=device), size=(N_total, N_total)).coalesce()
    edge_index = edge_index_coo.indices()
    assert (knn_p2p >= 0).all().item()
    assert edge_index.size(1) == N_total * k_p, "Number of edges != N_total * k_p"

    return edge_index, knn_p2p


@torch.no_grad()
def voxel_quota_seeds_select(
    p_xyz: torch.Tensor,    # (N,3) float/half on cuda
    comps: list,            # [(idx:LongTensor, n_c:int, s_c:int), ...]
    device: torch.device,
    points_cap_per_group: int = 5_120_000,
    n_bisect: int = 8
):
    assert p_xyz.is_cuda, "voxel_quota_seeds_select expects CUDA tensors"
    assert p_xyz.dim() == 2 and p_xyz.size(1) == 3

    def _get_closest_to_centroid(xyz, inv, num_voxels):
        vox_centroids = scatter_mean(xyz, inv, dim=0, dim_size=num_voxels)
        dists_sq = torch.sum((xyz - vox_centroids[inv]) ** 2, dim=1)
        _, argmin = scatter_min(dists_sq, inv, dim=0, dim_size=num_voxels)

        assert argmin.max().item() < xyz.size(0), "argmin index out of bounds"
        
        return argmin

    def _hash_voxels_with_comp(xyz, comp_local_ids, mins_c, voxel_c):
        v = voxel_c[comp_local_ids]                     
        mn = mins_c[comp_local_ids]                     

        grid = torch.floor((xyz - mn) / v.unsqueeze(1)).to(torch.int64)

        p1, p2, p3 = 73856093, 19349663, 83492791
        hashed = (grid[:, 0] * p1) ^ (grid[:, 1] * p2) ^ (grid[:, 2] * p3)
        hashed = hashed & ((1 << 40) - 1)

        keys = (comp_local_ids.to(torch.int64) << 40) | hashed 
        return keys

    groups = []
    cur, cur_pts = [], 0
    for (idx, n_c, s_c) in comps:
        if cur and (cur_pts + n_c > int(points_cap_per_group)):
            groups.append(cur)
            cur, cur_pts = [], 0
        cur.append((idx, n_c, s_c))
        cur_pts += n_c
    if cur:
        groups.append(cur)

    seeds_gid_list = []
    seeds_xyz_list = []

    for grp in groups:
        idx_all_list, comp_lid_list, n_list, s_list = [], [], [], []
        
        for lid, (idx, n_c, s_c) in enumerate(grp):
            idx_all_list.append(idx)
            comp_lid_list.append(torch.full((n_c,), lid, device=device, dtype=torch.long))
            n_list.append(int(n_c))
            s_list.append(int(s_c))

        idx_all = torch.cat(idx_all_list, dim=0)
        comp_lid = torch.cat(comp_lid_list, dim=0)
        xyz_g = p_xyz[idx_all].to(torch.float32).contiguous()
        
        G = len(grp)
        s_c_t = torch.tensor(s_list, device=device, dtype=torch.long)

        mins = torch.empty((G, 3), dtype=torch.float32, device=device)
        maxs = torch.empty((G, 3), dtype=torch.float32, device=device)
        for d in range(3):
            mins[:, d] = scatter(xyz_g[:, d], comp_lid, dim=0, dim_size=G, reduce='min')
            maxs[:, d] = scatter(xyz_g[:, d], comp_lid, dim=0, dim_size=G, reduce='max')
        span = (maxs - mins).clamp_(min=1e-6)

        safe_span = torch.max(span, torch.tensor(0.05, device=device))
        safe_vol  = safe_span[:, 0] * safe_span[:, 1] * safe_span[:, 2]

        v0 = torch.pow(safe_vol / torch.clamp(s_c_t.to(torch.float32), min=1.0), 1.0 / 3.0)

        v_lo = torch.clamp(v0 * 0.1, min=1e-4) 
        v_hi = torch.clamp(v0 * 64.0, min=1e-4)

        best_v = v0.clone()
        best_diff = torch.full((G,), 1 << 30, dtype=torch.long, device=device)

        for _ in range(int(n_bisect)):
            v_mid = (v_lo + v_hi) * 0.5

            keys = _hash_voxels_with_comp(xyz_g, comp_lid, mins, v_mid)
            uniq, inv = torch.unique(keys, return_inverse=True, sorted=False)

            pos = torch.arange(keys.numel(), device=device, dtype=torch.long)
            fast_first = scatter(pos, inv, dim=0, dim_size=uniq.numel(), reduce='min')

            comp_of_voxel = comp_lid[fast_first]

            cnt = scatter(torch.ones_like(comp_of_voxel, dtype=torch.long),
                            comp_of_voxel, dim=0, dim_size=G, reduce='sum')

            curr_diff = (cnt - s_c_t).abs()

            improved = curr_diff < best_diff

            if improved.any():
                best_diff[improved] = curr_diff[improved]
                best_v[improved]    = v_mid[improved]

            more = cnt > s_c_t
            v_lo = torch.where(more, v_mid, v_lo)
            v_hi = torch.where(~more, v_mid, v_hi)
   
        keys = _hash_voxels_with_comp(xyz_g, comp_lid, mins, best_v)
        uniq, inv = torch.unique(keys, return_inverse=True, sorted=False)

        num_voxels = uniq.numel()
        best_first_local = _get_closest_to_centroid(xyz_g, inv, num_voxels)

        if best_first_local.numel() > 0:
            final_global_ids = idx_all[best_first_local]
            seeds_gid_list.append(final_global_ids)
            seeds_xyz_list.append(p_xyz[final_global_ids].to(torch.float32))

    if len(seeds_gid_list) == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, 3, dtype=torch.float32, device=device),
                0)

    all_seeds_gid = torch.cat(seeds_gid_list, dim=0).to(torch.long).contiguous()
    all_seeds_xyz = torch.cat(seeds_xyz_list, dim=0).to(torch.float32).contiguous()
    
    return all_seeds_gid, all_seeds_xyz


class AssignModule(nn.Module):
    def __init__(self,
                 target_sp_size=400,
                 target_max_sp_cc=60,
                 edge_threshold=[0.75, 0.52],
                 w_cos=1.0,
                 w_geo=0.24,
                 k_seed=64,
                 group_points_cap=5_120_000):
        super().__init__()

        self.target_sp_size = target_sp_size
        self.target_max_sp_cc = target_max_sp_cc
        self.edge_threshold = edge_threshold
        self.w_cos = w_cos
        self.w_geo = w_geo
        self.k_seed = k_seed
        self.group_points_cap = group_points_cap

    @staticmethod
    def _offsets_to_batch(offsets, length: int, device):
        offsets_t = torch.as_tensor(offsets, dtype=torch.long, device=device) 
        assert offsets_t.dim() == 1 and offsets_t.numel() >= 2, "offsets must be 1-D with length >= 2"
        assert offsets_t[0] == 0, "Offsets must start with 0"
        
        B = int(offsets_t.numel() - 1)

        counts = offsets_t[1:] - offsets_t[:-1]

        assert counts.sum() == length, f"Offsets sum ({counts.sum()}) does not match expected length ({length})"

        batch_indices = torch.arange(B, device=device, dtype=torch.long)
        batch = torch.repeat_interleave(batch_indices, counts)
        
        return batch, B

    @staticmethod
    def _torch_renumber_to_zero_based(labels_gpu_tensor):
        unique_labels, inverse_indices = torch.unique(labels_gpu_tensor, return_inverse=True) 
        n_comp = unique_labels.shape[0]
        return inverse_indices, n_comp

    @staticmethod
    def torch_mnn_and_cc(strong_edges_directed, N_total, mnn=False):
        dev = strong_edges_directed.device
        N = N_total
        N_t = torch.tensor(N, device=dev, dtype=torch.long)

        i = strong_edges_directed[0].to(torch.long) # (E_topk,)
        j = strong_edges_directed[1].to(torch.long) # (E_topk,)
        
        if mnn:
            keys = i * N_t + j 
            rev  = j * N_t + i 
            keys_sorted, _ = torch.sort(keys) 
            pos = torch.searchsorted(keys_sorted, rev) 
            inb = pos < keys_sorted.numel() 
            eq = torch.zeros_like(inb, dtype=torch.bool) 
            if inb.any():
                eq[inb] = (keys_sorted[pos[inb]] == rev[inb]) 
            mask_dir = eq # (E_topk,)

            ui = i[mask_dir]
            vj = j[mask_dir]
            u = torch.minimum(ui, vj) 
            v = torch.maximum(ui, vj)
        else:
            mask_dir=None
            u = torch.minimum(i, j)
            v = torch.maximum(i, j)

        uv_keys = u * N_t + v 
        uv_keys_u = torch.unique(uv_keys) 
        u = torch.div(uv_keys_u, N_t, rounding_mode='floor') 
        v = uv_keys_u - u * N_t 

        if u.numel() == 0:
            labels_gpu = torch.arange(N, device=dev, dtype=torch.long) 
            n_comp = N 
        else:
            src = torch.cat([u, v], dim=0) 
            dst = torch.cat([v, u], dim=0) 
            labels_gpu = torch.arange(N, device=dev, dtype=torch.long) 
            BIG = torch.iinfo(labels_gpu.dtype).max
            max_iter = 1024
            for _ in range(max_iter):
                msg = torch.full((N,), BIG, dtype=labels_gpu.dtype, device=dev)
                msg.scatter_reduce_(0, dst, labels_gpu[src], reduce="amin", include_self=False) 
                new_labels = torch.minimum(labels_gpu, msg) 
                if torch.equal(new_labels, labels_gpu): 
                    break
                labels_gpu = new_labels
            labels_gpu, n_comp = AssignModule._torch_renumber_to_zero_based(labels_gpu.detach()) 
        
        return {
            "E_recip": int(u.numel()),
            "labels": labels_gpu,
            "n_comp": int(n_comp),
            "mask_dir": mask_dir # (E_topk,)
        }

    @staticmethod
    def _weighted_voronoi_assignment(p_xyz: torch.Tensor,
                                     p_fea: torch.Tensor,
                                     comp_ids_n: torch.Tensor,
                                     s_alloc: torch.Tensor,
                                     k_seed: int = 64,
                                     w_cos: float = 1.0,
                                     w_geo: float = 0.24,
                                     assign_chunk: int = 524288,
                                     group_points_cap: int = 5_120_000
                                     ):
        device = p_xyz.device
        N_total = p_xyz.size(0)
        assert p_fea.size(0) == N_total, "p_fea 与 p_xyz 点数不一致"
        M_cc = int(comp_ids_n.max().item()) + 1 if comp_ids_n.numel() > 0 else 0
        assert M_cc == s_alloc.numel(), "s_alloc 长度应等于 CC 数量"

        singleton_gid = []
        singleton_xyz = []
        filtered = []

        for cid in range(M_cc):
            idx = (comp_ids_n == cid).nonzero(as_tuple=True)[0]
            n_c = int(idx.numel())
            if n_c == 0:
                continue
            s_c = int(s_alloc[cid].item())
            s_c = max(1, min(s_c, n_c)) 
            
            if n_c == 1:
                singleton_gid.append(idx.view(-1))
                singleton_xyz.append(p_xyz[idx].view(1, 3))
            else:
                filtered.append((idx, n_c, s_c))

        vq_gid, vq_xyz = voxel_quota_seeds_select(
            p_xyz=p_xyz,
            comps=filtered,
            device=device,
            points_cap_per_group=group_points_cap,
            n_bisect=8
        )

        seeds_gid_list = []
        seeds_xyz_list = []

        if len(singleton_gid) > 0:
            seeds_gid_list.append(torch.cat(singleton_gid, dim=0))
            seeds_xyz_list.append(torch.cat(singleton_xyz, dim=0))

        seeds_gid_list.append(vq_gid)
        seeds_xyz_list.append(vq_xyz)

        all_seeds_gid = torch.cat(seeds_gid_list, dim=0).to(torch.long).contiguous()
        all_seeds_xyz = torch.cat(seeds_xyz_list, dim=0).to(torch.float32).contiguous()

        S_eff = int(all_seeds_gid.numel())
        S_tar = 0
        for (_, n_c, s_c) in filtered:
            S_tar += int(s_c)
        S_tar += sum(1 for _ in singleton_gid)

        seeds_fea = p_fea[all_seeds_gid].to(torch.float32).contiguous()   # (S_eff, C)
        p_fn = F.normalize(p_fea.to(torch.float32), p=2, dim=-1)          # (N, C)
        s_fn = F.normalize(seeds_fea, p=2, dim=-1)                        # (S_eff, C)

        pts_1 = p_xyz.unsqueeze(0).to(torch.float32)            # (1, N, 3)
        seeds_1 = all_seeds_xyz.unsqueeze(0).to(torch.float32)  # (1, S_eff, 3)
        len_pts_1 = torch.tensor([N_total], device=device, dtype=torch.long)
        len_seeds_1 = torch.tensor([S_eff], device=device, dtype=torch.long)
        Kc = min(int(k_seed), S_eff)
        
        # dists2: (1, N, Kc)
        dists2, knn_idx, _ = knn_points(pts_1, seeds_1, K=Kc, lengths1=len_pts_1, lengths2=len_seeds_1)  # dists2: (1, N, Kc) 为欧氏“平方距离”；knn_idx: (1, N, Kc) 为种子下标 [0..S_eff-1]

        d_geo = torch.sqrt(dists2[0] + 1e-12)   # (N, Kc)

        labels_pp = torch.empty(N_total, dtype=torch.long, device=device)
        step = int(assign_chunk)
        
        with autocast(enabled=False):
            for s in range(0, N_total, step):
                e = min(s + step, N_total)
                pf = p_fn[s:e].contiguous()              # (n, C)
                idx_block = knn_idx[0, s:e].contiguous() # (n, Kc)
                dg = d_geo[s:e]                          # (n, Kc)
                n = pf.size(0)
                if n == 0:
                    continue

                rows_local = torch.arange(n, device=device, dtype=torch.int32).repeat_interleave(idx_block.size(1))
                cols_global = idx_block.reshape(-1).to(torch.int32)

                g_blk = dgl.heterograph(
                    {('p', 'to', 's'): (rows_local, cols_global)},
                    num_nodes_dict={'p': n, 's': s_fn.size(0)},
                    device=device
                )

                cos_flat = DOP.u_dot_v(g_blk, pf, s_fn).reshape(n, -1)   # (n, Kc)
                cos_norm = (cos_flat + 1.0) * 0.5

                local_max_dist = dg.max(dim=1, keepdim=True)[0]          # (n, 1)
                dg_norm = dg / (local_max_dist + 1e-8)        # (n, Kc) -> [0, 1]

                # Score = Feature_Score - Geometry_Penalty
                scores = w_cos * cos_norm - w_geo * dg_norm
                
                best_k = torch.argmax(scores, dim=1)                     # (n,)
                labels_pp[s:e] = idx_block[torch.arange(n, device=device), best_k]

        labels_pp[all_seeds_gid] = torch.arange(S_eff, device=device, dtype=torch.long)
        M_new = S_eff
        
        return labels_pp, M_new

    def forward(self, p_fea, p_xyz, edge_index, scores_p_feature, scores_p_semantic, k_per_point):
        device = p_fea.device
        N_total = p_fea.size(0)

        with torch.no_grad():
            scores_p = scores_p_feature + scores_p_semantic  # (E,)
            a_ij_view = scores_p.view(N_total, -1) # (N_total, k_p)
            ranks_i = torch.argsort(torch.argsort(a_ij_view, dim=1, descending=True), dim=1)
            k_per_point_b = k_per_point.unsqueeze(1) # (N_total, 1)
            topk_mask = (ranks_i < k_per_point_b).view(-1) # (E,)
             
            knn_mask = topk_mask
            if (self.edge_threshold is not None):
                keep_feature = (scores_p_feature >= float(self.edge_threshold[0])) # (E,)
                keep_semantic = (scores_p_semantic >= float(self.edge_threshold[1])) # (E,)
                keep = keep_feature & keep_semantic
                if keep.any():
                    top1_mask = (ranks_i < 1).view(-1) # (E,)
                    knn_mask = (keep & topk_mask) | top1_mask

            strong_edges_directed = edge_index[:, knn_mask]

            if strong_edges_directed.numel() > 0:
                cc_results = AssignModule.torch_mnn_and_cc(strong_edges_directed, N_total, mnn=False)
            else:
                cc_results = {
                    "labels": torch.arange(N_total, device=device, dtype=torch.long), 
                    "n_comp": N_total,
                    "mask_dir": None
                }
            
            reciprocal_mask = cc_results["mask_dir"] # (E_topk,)
            assert reciprocal_mask is None
            if reciprocal_mask is not None:
                knn_mask[knn_mask.clone()] = reciprocal_mask

            comp_ids_n = cc_results["labels"] # (N_total,)
            M = cc_results["n_comp"]

            assert self.target_sp_size > 0.0, "目标超点大小 必须 > 0"
            assert self.target_max_sp_cc >= 1, "每个连通分量最大超点数 必须 >= 1"

            counts = torch.bincount(comp_ids_n, minlength=M)
            raw_seeds = counts.to(torch.float32) / self.target_sp_size
            soft_limit = self.target_max_sp_cc
            damping_ratio = 0.1 
            s_soft = torch.where(
                raw_seeds > soft_limit,
                soft_limit + (raw_seeds - soft_limit) * damping_ratio,
                raw_seeds
            )
            s_est = torch.floor(s_soft).to(torch.long)
            non_empty = counts > 0
            s_clamped = s_est.clamp(min=1) 
            s_alloc = torch.where(non_empty, s_clamped, torch.zeros_like(s_clamped))

            comp_ids_n, M_new = AssignModule._weighted_voronoi_assignment(
                p_xyz=p_xyz,
                p_fea=p_fea,
                comp_ids_n=comp_ids_n,
                s_alloc=s_alloc,
                k_seed=getattr(self, "k_seed"),
                w_cos=getattr(self, "w_cos"),
                w_geo=getattr(self, "w_geo"),
                assign_chunk=getattr(self, "assign_chunk", 524288),
                group_points_cap=getattr(self, "group_points_cap", 5120000)
            )

            comp_ids_n, M = AssignModule._torch_renumber_to_zero_based(comp_ids_n)
            assert M == M_new, f"Mismatch after renumber: {M} vs {M_new}"

            cluster_sizes = torch.bincount(comp_ids_n, minlength=M)

        results = {
            "sp_ids_n": comp_ids_n,   # (N_total,) - 每个点的超点ID
            "num_superpoints": M,     # (int) - 超点总数 M
            "cc_ids": cc_results["labels"], # (N_total,) - 每个点的连通分量ID
            "cluster_sizes": cluster_sizes  # (M,) - 每个超点的大小
        }

        return results


# ===== Main Network =====
class SPANetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        pool_reduce = cfg.get('pool_reduce', 'max')

        # Encoder parameters
        ptv3_enc_channels = cfg.get('ptv3_enc_channels', (32, 64, 128, 256, 512))
        ptv3_enc_depths = cfg.get('ptv3_enc_depths', (2, 2, 2, 6, 2))
        ptv3_enc_num_head = cfg.get('ptv3_enc_num_head', (2, 4, 8, 16, 32))
        ptv3_enc_patch_size = cfg.get('ptv3_enc_patch_size', (1024, 1024, 1024, 1024, 1024))

        # Decoder parameters
        # The final output dimension of PTV3 will be dec_channels[0]
        ptv3_dec_channels = cfg.get('ptv3_dec_channels', (64, 64, 128, 256))
        ptv3_dec_depths = cfg.get('ptv3_dec_depths', (2, 2, 2, 2))
        ptv3_dec_num_head = cfg.get('ptv3_dec_num_head', (4, 4, 8, 16))
        ptv3_dec_patch_size = cfg.get('ptv3_dec_patch_size', (1024, 1024, 1024, 1024))

        # Other parameters
        ptv3_mlp_ratio = cfg.get('ptv3_mlp_ratio', 4)
        drop_path = cfg.get('ptv3_drop_path', 0.3)
        enc_enable_rpe = cfg.get('enc_enable_rpe')
        enc_enable_flash = cfg.get('enc_enable_flash')
        dec_enable_rpe = cfg.get('dec_enable_rpe')
        dec_enable_flash = cfg.get('dec_enable_flash')

        self.backbone = PointTransformerV3(
            in_channels=9,
            enc_channels=ptv3_enc_channels,
            enc_depths=ptv3_enc_depths,
            enc_num_head=ptv3_enc_num_head,
            enc_patch_size=ptv3_enc_patch_size,
            dec_channels=ptv3_dec_channels,
            dec_depths=ptv3_dec_depths,
            dec_num_head=ptv3_dec_num_head,
            dec_patch_size=ptv3_dec_patch_size,
            mlp_ratio=ptv3_mlp_ratio,
            pool_reduce=pool_reduce,
            drop_path=drop_path,
            enc_enable_rpe=enc_enable_rpe,
            enc_enable_flash=enc_enable_flash,
            dec_enable_rpe=dec_enable_rpe,
            dec_enable_flash=dec_enable_flash
        )

        self.k_p = int(cfg.get('k_p', 16))
        self.strong_k_min = int(cfg.get('strong_k_min', 6))
        self.strong_k_max = int(cfg.get('strong_k_max', 10))
        assert self.strong_k_max <= self.k_p, "strong_k_max must be <= k_p (KNN)"
        self.density_gamma = cfg.get('density_gamma', 1.0)

        self.target_sp_size = cfg.get('target_sp_size', 400)
        self.target_max_sp_cc = cfg.get('target_max_sp_cc', 60)
        self.edge_threshold = cfg.get('edge_threshold', [0.75, 0.52])
        self.w_cos = cfg.get('w_cos', 1.0)
        self.w_geo = cfg.get('w_geo', 0.24)
        self.k_seed = cfg.get('k_seed', 64)
        self.group_points_cap = cfg.get('group_points_cap', 5_120_000)

        _dtype = str(cfg.get('feat_dtype', 'float32')).lower()
        if _dtype in ('bf16','bfloat16'):
            self.feat_dtype = torch.bfloat16
        elif _dtype in ('fp16','float16','half'):
            self.feat_dtype = torch.float16
        else:
            self.feat_dtype = torch.float32
        
        self.num_classes = int(cfg.get('num_classes', 13))
        self.seg_head = (
            nn.Linear(ptv3_dec_channels[0], self.num_classes)
        )

        self.sp_1 = AssignModule(target_sp_size=self.target_sp_size, target_max_sp_cc=self.target_max_sp_cc, edge_threshold=self.edge_threshold, w_cos=self.w_cos, w_geo=self.w_geo, k_seed=self.k_seed, group_points_cap=self.group_points_cap)

        self.use_dgl_for_scores = bool(cfg.get('use_dgl_for_scores', True))
        self.edge_chunk   = int(cfg.get('edge_chunk', 1_000_000))
        self.channel_chunk= int(cfg.get('channel_chunk', 64))
        self.scores_use_fp16 = bool(cfg.get('scores_use_fp16', True))

    @staticmethod
    def _offsets_to_batch(offsets, length: int, device):
        offsets_t = torch.as_tensor(offsets, dtype=torch.long, device=device) 
        assert offsets_t.dim() == 1 and offsets_t.numel() >= 2, "offsets must be 1-D with length >= 2"
        assert offsets_t[0] == 0, "Offsets must start with 0"
        
        B = int(offsets_t.numel() - 1)

        counts = offsets_t[1:] - offsets_t[:-1]

        assert counts.sum() == length, f"Offsets sum ({counts.sum()}) does not match expected length ({length})"

        batch_indices = torch.arange(B, device=device, dtype=torch.long)
        batch = torch.repeat_interleave(batch_indices, counts)
        
        return batch, B

    @torch.no_grad()
    def _edge_scores_stream(self, rows, cols, p_fea, seg_logits, seg_prob=None):
        dev = p_fea.device
        E = rows.numel()
        C = p_fea.size(1)

        if seg_logits is None:
            assert seg_prob is not None, "seg_prob must be provided if seg_logits is None"
            probs = seg_prob
        else:
            probs = torch.softmax(seg_logits, dim=1)
        if self.scores_use_fp16:
            probs = probs.half()
        scores_p_semantic = (probs[rows] * probs[cols]).sum(-1).float().clamp_(0, 1)
        base = F.normalize(p_fea, p=2, dim=1)
        if self.scores_use_fp16:
            base = base.half()

        out_feat = torch.empty(E, device=dev, dtype=torch.float32)

        for es in range(0, E, self.edge_chunk):
            ee = min(es + self.edge_chunk, E)
            acc = torch.zeros(ee - es, device=dev, dtype=torch.float32)

            r = rows[es:ee]; c = cols[es:ee]
            for cs in range(0, C, self.channel_chunk):
                ce = min(cs + self.channel_chunk, C)
                fr = base[r, cs:ce]   # (e_chunk, c_chunk)
                fc = base[c, cs:ce]
                acc += (fr * fc).sum(-1).float()

            out_feat[es:ee] = (acc + 1.0) * 0.5  # 映射到 [0,1]

        return out_feat, scores_p_semantic

    def knn_graph_construction(self, p_fea, p_xyz, point_offsets, seg_logits, seg_prob=None):
        device = p_fea.device
        compute_dtype = torch.float32

        edge_index, knn_p2p = compute_knn(p_xyz, point_offsets, int(self.k_p))  # E = N_total * k_p
        point_idx_e = edge_index[0]  # (E,)
        neighbor_idx_e = edge_index[1]  # (E,)
        E = edge_index.size(1)
        N_total = p_fea.size(0)

        with torch.no_grad():
            assert E == N_total * self.k_p, "edge_index 大小与 k_p 不匹配"
            dists = torch.norm(p_xyz[point_idx_e] - p_xyz[neighbor_idx_e], p=2, dim=1)  # (E,)
            dists_view = dists.view(N_total, int(self.k_p))                                  # (N, k)
            avg_dist_i = dists_view.mean(dim=1)                                         # (N,)

            point_batch_ids, B = self._offsets_to_batch(point_offsets, N_total, device)
            b_min = scatter(avg_dist_i, point_batch_ids, dim=0, dim_size=B, reduce='min')
            b_max = scatter(avg_dist_i, point_batch_ids, dim=0, dim_size=B, reduce='max')
            d_norm_i = (avg_dist_i - b_min[point_batch_ids]) / (b_max[point_batch_ids] - b_min[point_batch_ids] + 1e-8)

            k_range = self.strong_k_max - self.strong_k_min
            k_float_i = self.strong_k_min + (d_norm_i.pow(self.density_gamma) * k_range)
            k_per_point = torch.round(k_float_i).long().clamp(self.strong_k_min, self.strong_k_max)

        rows = point_idx_e.contiguous()
        cols = neighbor_idx_e.contiguous()

        if self.use_dgl_for_scores:
            g = dgl.graph((rows, cols), num_nodes=N_total, device=device)

            with autocast(enabled=False):
                p_unit = F.normalize(p_fea.to(compute_dtype), p=2, dim=1) # (N, C)
                cos_feat = DOP.u_dot_v(g, p_unit, p_unit).reshape(-1)  # (E,)
                scores_p_feature = (cos_feat + 1.0) * 0.5

                if seg_logits is None:
                    assert seg_prob is not None, "seg_prob must be provided if seg_logits is None"
                    seg_probs = seg_prob.to(compute_dtype)
                else:
                    seg_probs = torch.softmax(seg_logits.to(compute_dtype), dim=1)  # (N, K)
                prob_dot = DOP.u_dot_v(g, seg_probs, seg_probs).reshape(-1) # (E,)
                scores_p_semantic = prob_dot.clamp_(0.0, 1.0)
        else:
            scores_p_feature, scores_p_semantic = self._edge_scores_stream(
                rows, cols, p_fea, seg_logits, seg_prob
            )

        return edge_index, k_per_point, scores_p_feature, scores_p_semantic

    def forward(self, points_dict: dict, point_offsets: np.ndarray):
        """
        Args:
            points_dict: A dictionary containing point cloud data.
            point_offsets: (B+1,) batch offsets where point_offsets[0]=0
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        points_dict_t = ToTensor()(points_dict)
        for key in points_dict_t.keys():
            if isinstance(points_dict_t[key], torch.Tensor):
                points_dict_t[key] = points_dict_t[key].to(device, non_blocking=True)

        xyz = points_dict_t['coord']  # (N_total, 3) tensor
        
        with autocast(enabled=False):
            p_fea = self.backbone(points_dict_t).feat  # (N_total, C)

        p_fea = p_fea.to(self.feat_dtype).contiguous()

        seg_logits = self.seg_head(p_fea)

        edge_index, k_per_point, scores_p_feature, scores_p_semantic = self.knn_graph_construction(
            p_fea, xyz, point_offsets, seg_logits
        )

        assignments = self.sp_1(p_fea, xyz, edge_index, scores_p_feature, scores_p_semantic, k_per_point)
        
        results = {
            'p_fea': p_fea,
            'seg_logits': seg_logits,
            'knn_edge_index': edge_index,
            'knn_edge_logits': scores_p_feature,
            'num_superpoints': assignments['num_superpoints'],
            'cluster_sizes': assignments['cluster_sizes'],
            'asso_data': assignments['sp_ids_n'],
            'cc_ids': assignments['cc_ids']
        }

        return results
