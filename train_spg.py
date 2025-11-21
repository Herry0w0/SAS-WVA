import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast
from torch_scatter import scatter_mean, scatter_add
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import json
import logging

from src.dataset.s3dis import S3dis_Dataset
from src.network import SPANetwork
from src.spg_utils import build_spg_edge_features, compute_superpoint_pca, subgraph_sampling
from src.spg_gcn import SPG_GCN
from src.metrics_spg import PointCloudEvaluator
from src.dataset.custom_sampler import concat_collate_fn

# S3DIS Class Weights
S3DIS_CLASS_WEIGHTS = [3.3, 2.8, 3.2, 3.4, 4.8, 4.2, 4.0, 4.3, 4.4, 4.6, 4.8, 4.7, 4.8]


def setup_logger(output_dir, rank):
    logger = logging.getLogger("SPG_Train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    if rank == 0:
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        log_file = os.path.join(output_dir, 'train.log')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def train_one_epoch(backbone, spg_head, loader, optimizer, criterion, device, rank, num_classes=13, use_augmentation=True):
    backbone.eval()
    spg_head.train()
    
    total_loss = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)
    
    iterator = tqdm(loader, desc="Train", leave=False, disable=(rank != 0))
    
    for batch in iterator:
        if batch is None: continue

        pc_dict = batch['points_dict']
        offsets_np = batch['offsets_np']
        labels = batch['labels'].to(device, non_blocking=True)
        
        feat = torch.from_numpy(pc_dict['feat']).to(device, non_blocking=True)
        pos = feat[:, 0:3].contiguous()
        color = feat[:, 3:6].contiguous()
        pc_dict.update({'grid_size': 0.03})

        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                results = backbone(pc_dict, offsets_np)
            sp_ids = results['asso_data']
            num_sp = results['num_superpoints']
            point_edge_index = results['knn_edge_index']

        centroid, length, surface, volume, counts = compute_superpoint_pca(pos, sp_ids, num_sp)
        sp_rgb = scatter_mean(color, sp_ids, dim=0, dim_size=num_sp)
        
        scene_center = scatter_mean(pos, torch.zeros(pos.shape[0], dtype=torch.long, device=device), dim=0)
        norm_centroid = centroid - scene_center
        
        log_length = torch.log(length.unsqueeze(1).clamp(min=1e-6) + 1)
        log_surface = torch.log(surface.unsqueeze(1).clamp(min=1e-6) + 1)
        log_volume = torch.log(volume.unsqueeze(1).clamp(min=1e-6) + 1)
        
        sp_x = torch.cat([
            norm_centroid, 
            sp_rgb, 
            log_length, log_surface, log_volume, 
            torch.log(counts.unsqueeze(1) + 1)
        ], dim=1).float()

        labels_onehot = torch.nn.functional.one_hot(labels, num_classes).float()
        sp_label_counts = scatter_add(labels_onehot, sp_ids, dim=0, dim_size=num_sp)
        sp_labels = torch.argmax(sp_label_counts, dim=1)
        
        edge_index, edge_attr = build_spg_edge_features(pos, sp_ids, num_sp, point_edge_index)

        if use_augmentation and num_sp > 128: 
            sub_x, sub_edge_index, sub_edge_attr, sub_labels = subgraph_sampling(
                sp_x, edge_index, edge_attr, sp_labels, num_sp, max_nodes=512, k_hops=3
            )
        else:
            sub_x, sub_edge_index, sub_edge_attr, sub_labels = sp_x, edge_index, edge_attr, sp_labels

        if sub_x.size(0) <= 1: continue

        sp_logits = spg_head(sub_x, sub_edge_index, sub_edge_attr)
        loss = criterion(sp_logits, sub_labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(spg_head.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.detach()
        count += 1
        
        if rank == 0:
            iterator.set_postfix({'loss': loss.item()})
            
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    
    return (total_loss / count).item() if count > 0 else 0.0


@torch.no_grad()
def evaluate(backbone, spg_head, loader, device, evaluator, criterion, rank):
    backbone.eval()
    spg_head.eval()
    evaluator.reset()
    
    local_loss = torch.tensor(0.0, device=device)
    local_count = torch.tensor(0.0, device=device)
    num_classes = evaluator.num_classes
    
    iterator = tqdm(loader, desc="Eval", leave=False, disable=(rank != 0))
    
    for batch in iterator:
        if batch is None: continue
        
        pc_dict = batch['points_dict']
        offsets_np = batch['offsets_np']
        labels = batch['labels'].to(device, non_blocking=True)
        
        feat = torch.from_numpy(pc_dict['feat']).to(device, non_blocking=True)
        pos = feat[:, 0:3].contiguous()
        color = feat[:, 3:6].contiguous()
        pc_dict.update({'grid_size': 0.03})

        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                results = backbone(pc_dict, offsets_np)
        
        sp_ids = results['asso_data']
        num_sp = results['num_superpoints']
        point_edge_index = results['knn_edge_index']

        centroid, length, surface, volume, counts = compute_superpoint_pca(pos, sp_ids, num_sp)
        sp_rgb = scatter_mean(color, sp_ids, dim=0, dim_size=num_sp)

        scene_center = scatter_mean(pos, torch.zeros(pos.shape[0], dtype=torch.long, device=device), dim=0)
        norm_centroid = centroid - scene_center
        
        log_length = torch.log(length.unsqueeze(1).clamp(min=1e-6) + 1)
        log_surface = torch.log(surface.unsqueeze(1).clamp(min=1e-6) + 1)
        log_volume = torch.log(volume.unsqueeze(1).clamp(min=1e-6) + 1)
        
        sp_x = torch.cat([
            norm_centroid, 
            sp_rgb, 
            log_length, log_surface, log_volume, 
            torch.log(counts.unsqueeze(1) + 1)
        ], dim=1).float()
        # ==============================

        edge_index, edge_attr = build_spg_edge_features(pos, sp_ids, num_sp, point_edge_index)

        if sp_x.size(0) == 0: continue

        sp_logits = spg_head(sp_x, edge_index, edge_attr)
        
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes).float()
        sp_label_counts = scatter_add(labels_onehot, sp_ids, dim=0, dim_size=num_sp)
        sp_labels = torch.argmax(sp_label_counts, dim=1)
        
        loss = criterion(sp_logits, sp_labels)
        local_loss += loss.detach()
        local_count += 1
        
        point_logits = sp_logits[sp_ids]
        point_preds = torch.argmax(point_logits, dim=1)

        evaluator.update(point_preds, labels)
    
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
    avg_loss = (local_loss / local_count).item() if local_count > 0 else 0.0
    
    dist.all_reduce(evaluator.confusion_matrix, op=dist.ReduceOp.SUM)
    
    metrics = evaluator.compute_metrics()
    metrics['loss'] = avg_loss
    return metrics


def run_fold(fold_idx, args, cfg, device, rank, world_size, logger):
    if rank == 0:
        logger.info(f"========================================")
        logger.info(f"Starting Fold {fold_idx} (Test Area: Area_{fold_idx})")
        logger.info(f"========================================")
    
    train_set = S3dis_Dataset(args.data_root, logger, mode='train', 
                              test_area_idx=fold_idx, 
                              grid_sample_cfg=cfg['data'].get('grid_sample'), 
                              transform_cfg=cfg['data'].get('train_transform'))
    
    val_set = S3dis_Dataset(args.data_root, logger, mode='val', 
                            test_area_idx=fold_idx, 
                            grid_sample_cfg=cfg['data'].get('grid_sample'), 
                            transform_cfg=cfg['data'].get('val_transform'))
    
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                              collate_fn=concat_collate_fn, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, sampler=val_sampler,
                            collate_fn=concat_collate_fn, num_workers=4)

    backbone = SPANetwork(cfg['model']).to(device)
    if args.backbone_ckpt:
        if rank == 0: logger.info(f"Loading backbone from {args.backbone_ckpt}")
        ckpt = torch.load(args.backbone_ckpt, map_location=device)
        state_dict = ckpt['ema'] if 'ema' in ckpt else ckpt['model']
        backbone.load_state_dict(state_dict, strict=False)
    
    spg_head = SPG_GCN(
        in_channels=10, 
        hidden_channels=args.spg_hidden, 
        num_classes=13, 
        iterations=args.spg_iter
    ).to(device)
    
    spg_head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(spg_head)
    spg_head = DDP(spg_head, device_ids=[rank], output_device=rank)
    
    optimizer = optim.Adam(spg_head.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[275, 320], gamma=0.7)
    
    class_weights = torch.tensor(S3DIS_CLASS_WEIGHTS, device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    
    evaluator = PointCloudEvaluator(num_classes=13, device=device)

    best_miou = 0.0
    best_path = os.path.join(args.output_dir, f"best_spg_fold{fold_idx}.pth")
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_fold{fold_idx}.pth")

    start_epoch = 0
    if os.path.isfile(checkpoint_path):
        if rank == 0: logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        spg_head.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0.0)
        if rank == 0: logger.info(f"Resumed at Epoch {start_epoch} with Best mIoU: {best_miou:.4f}")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        loss = train_one_epoch(backbone, spg_head, train_loader, optimizer, criterion, device, rank)
        
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            logger.info(f"Fold {fold_idx} Epoch {epoch+1}/{args.epochs} | Train Loss: {loss:.4f} | LR: {current_lr:.6f}")

        if (epoch + 1) % 10 == 0 or epoch > 300:
            metrics = evaluate(backbone, spg_head, val_loader, device, evaluator, criterion, rank)
            
            if rank == 0:
                current_miou = metrics['mIoU']
                logger.info(f"Fold {fold_idx} Epoch {epoch+1} Val | "
                            f"Loss: {metrics['loss']:.4f} | mIoU: {current_miou*100:.2f}% | "
                            f"OA: {metrics['OA']*100:.2f}% | mAcc: {metrics['mAcc']*100:.2f}%")
                
                if current_miou > best_miou:
                    best_miou = current_miou
                    torch.save(spg_head.module.state_dict(), best_path)
                    logger.info(f"Saved Best Model (mIoU: {best_miou*100:.2f}%)")
        
        if rank == 0:
            checkpoint_state = {
                'epoch': epoch,
                'model': spg_head.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_miou': best_miou
            }
            torch.save(checkpoint_state, checkpoint_path)
    
    if rank == 0: logger.info(f"Reloading best model for Fold {fold_idx} evaluation...")
    dist.barrier()
    
    if os.path.exists(best_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        spg_head.module.load_state_dict(torch.load(best_path, map_location=map_location))
    
    final_metrics = evaluate(backbone, spg_head, val_loader, device, evaluator, criterion, rank)
    
    if rank == 0:
        logger.info(f"Fold {fold_idx} Final Result | mIoU: {final_metrics['mIoU']*100:.2f}%")
    
    return final_metrics 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/s3dis.yaml')
    parser.add_argument('--data_root', type=str, default='/path/to/data/S3DIS')
    parser.add_argument('--output_dir', type=str, default='./exp/spg_cv')
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=4) 
    parser.add_argument('--spg_hidden', type=int, default=128)
    parser.add_argument('--spg_iter', type=int, default=10)
    parser.add_argument('--backbone_ckpt', type=str, default='/path/to/checkpoints/epoch1400.pth')
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        
    logger = setup_logger(args.output_dir, rank)
    torch.set_float32_matmul_precision("high")

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    all_fold_metrics = []
    
    for fold in range(1, 7):
        metrics = run_fold(fold, args, cfg, device, rank, world_size, logger)
        if rank == 0:
            all_fold_metrics.append(metrics)
        dist.barrier()
    
    if rank == 0:
        avg_miou = np.mean([m['mIoU'] for m in all_fold_metrics])
        avg_oa = np.mean([m['OA'] for m in all_fold_metrics])
        avg_macc = np.mean([m['mAcc'] for m in all_fold_metrics])
        
        miou_strs = [f"{m['mIoU']*100:.2f}%" for m in all_fold_metrics]
        
        logger.info("========================================")
        logger.info(f"6-Fold CV Completed.")
        logger.info(f"Per Fold mIoU: {miou_strs}")
        logger.info(f"Average mIoU: {avg_miou*100:.2f}%")
        logger.info(f"Average OA:   {avg_oa*100:.2f}%")
        logger.info(f"Average mAcc: {avg_macc*100:.2f}%")
        logger.info("========================================")
        
        json_results = []
        for m in all_fold_metrics:
            json_results.append({
                "mIoU": float(m['mIoU']),
                "OA": float(m['OA']),
                "mAcc": float(m['mAcc']),
                "Loss": float(m['loss']),
            })
            
        with open(os.path.join(args.output_dir, "cv_summary.json"), "w") as f:
            json.dump({
                "folds": json_results, 
                "avg_miou": float(avg_miou),
                "avg_oa": float(avg_oa),
                "avg_macc": float(avg_macc)
            }, f, indent=4)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
