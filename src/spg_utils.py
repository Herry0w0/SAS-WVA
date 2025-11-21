import torch
from torch_scatter import scatter_mean, scatter_add


def compute_superpoint_pca(pos, sp_ids, num_sp):
    centroid = scatter_mean(pos, sp_ids, dim=0, dim_size=num_sp)

    pos_centered = pos - centroid[sp_ids]
    outer_prod = torch.bmm(pos_centered.unsqueeze(2), pos_centered.unsqueeze(1)) 
    cov_sum = scatter_add(outer_prod, sp_ids, dim=0, dim_size=num_sp)
    
    counts = scatter_add(torch.ones_like(sp_ids, dtype=torch.float32), sp_ids, dim=0, dim_size=num_sp)
    counts_clamped = counts.clamp(min=2) 
    cov = cov_sum / (counts_clamped.view(-1, 1, 1) - 1)

    try:
        eigvals = torch.linalg.eigvalsh(cov) 
    except RuntimeError:
        cov = cov + torch.eye(3, device=pos.device).unsqueeze(0) * 1e-6
        eigvals = torch.linalg.eigvalsh(cov)

    l3 = eigvals[:, 0].clamp(min=0)
    l2 = eigvals[:, 1].clamp(min=0)
    l1 = eigvals[:, 2].clamp(min=0)

    length = l1
    surface = torch.sqrt(l1 * l2 + 1e-10)
    volume = torch.sqrt(l1 * l2 * l3 + 1e-10)
    
    return centroid, length, surface, volume, counts


def build_spg_edge_features(pos, sp_ids, num_sp, point_edge_index):
    device = pos.device

    centroid, length, surface, volume, counts = compute_superpoint_pca(pos, sp_ids, num_sp)

    row, col = point_edge_index
    sp_row = sp_ids[row]
    sp_col = sp_ids[col]

    mask = sp_row != sp_col
    if not mask.any():
        return torch.zeros((2, 0), device=device, dtype=torch.long), torch.zeros((0, 13), device=device)

    interface_row_pts = row[mask]
    interface_col_pts = col[mask]
    interface_sp_src = sp_row[mask]
    interface_sp_dst = sp_col[mask]

    sp_edge_tensor = torch.stack([interface_sp_src, interface_sp_dst], dim=0)
    unique_edges, inverse_indices = torch.unique(sp_edge_tensor, dim=1, return_inverse=True)
    
    num_sp_edges = unique_edges.size(1)
    src_sp = unique_edges[0]
    dst_sp = unique_edges[1]

    delta = pos[interface_row_pts] - pos[interface_col_pts] # (E_interface, 3)

    edge_counts = scatter_add(torch.ones_like(inverse_indices, dtype=torch.float32), inverse_indices, dim=0, dim_size=num_sp_edges)
    edge_counts = edge_counts.unsqueeze(1).clamp(min=1)
    
    delta_sum = scatter_add(delta, inverse_indices, dim=0, dim_size=num_sp_edges)
    delta_avg = delta_sum / edge_counts 

    delta_sq_sum = scatter_add(delta**2, inverse_indices, dim=0, dim_size=num_sp_edges)
    delta_var = (delta_sq_sum / edge_counts) - delta_avg**2
    delta_std = torch.sqrt(delta_var.clamp(min=1e-6))

    delta_centroid = centroid[src_sp] - centroid[dst_sp]

    def get_log_diff(attr):
        log_attr = torch.log(attr.clamp(min=1e-6))
        return (log_attr[src_sp] - log_attr[dst_sp]).unsqueeze(1)

    len_r = get_log_diff(length)
    surf_r = get_log_diff(surface)
    vol_r = get_log_diff(volume)
    size_r = get_log_diff(counts)

    edge_attr = torch.cat([
        delta_avg,      # 3
        delta_std,      # 3
        len_r,          # 1
        surf_r,         # 1
        vol_r,          # 1
        size_r,         # 1
        delta_centroid  # 3
    ], dim=1)           

    return unique_edges, edge_attr


def subgraph_sampling(sp_x, edge_index, edge_attr, sp_labels, num_sp, max_nodes=512, k_hops=3):
    device = sp_x.device
    if num_sp <= max_nodes:
        return sp_x, edge_index, edge_attr, sp_labels

    center_node = torch.randint(0, num_sp, (1,), device=device)
    
    subset = center_node
    current_frontier = center_node
    
    for _ in range(k_hops):
        mask = torch.isin(edge_index[0], current_frontier)
        neighbors = edge_index[1][mask].unique()
        
        new_nodes = neighbors[~torch.isin(neighbors, subset)]
        
        if new_nodes.numel() == 0:
            break
            
        subset = torch.cat([subset, new_nodes])
        current_frontier = new_nodes
        
        if subset.numel() >= max_nodes:
            subset = subset[:max_nodes]
            break
    
    new_sp_x = sp_x[subset]
    new_sp_labels = sp_labels[subset]
    
    mask_src = torch.isin(edge_index[0], subset)
    mask_dst = torch.isin(edge_index[1], subset)
    edge_mask = mask_src & mask_dst
    
    new_edge_index = edge_index[:, edge_mask]
    new_edge_attr = edge_attr[edge_mask]
    
    map_tensor = torch.full((num_sp,), -1, dtype=torch.long, device=device)
    map_tensor[subset] = torch.arange(subset.numel(), device=device)
    
    new_edge_index = map_tensor[new_edge_index]
    
    return new_sp_x, new_edge_index, new_edge_attr, new_sp_labels
