import numpy as np
import random
import torch


class AdaptiveBatchSampler:
    """Dynamic batch sampler that groups samples by size"""
    def __init__(self, dataset, max_points_per_batch=350000, shuffle=True, seed=1003):
        self.dataset = dataset
        self.max_points_per_batch = max_points_per_batch
        self.shuffle = shuffle
        self.base_seed = int(seed)

        self.sizes = [s['num_points'] for s in dataset.samples]
        self.indices = list(range(len(dataset)))
        self.num_batches = 0

        self.rng = random.Random(self.base_seed)

    def set_epoch(self, epoch):
        self.rng.seed(self.base_seed + epoch)

    def __iter__(self):
        # Optionally shuffle while maintaining size ordering
        if self.shuffle:
            # Shuffle in size-similar groups
            indices = self._shuffle_by_groups()
        else:
            indices = self.indices.copy()

        # Create batches
        batches = []
        current_batch = []
        current_points = 0

        for idx in indices:
            sample_points = self.sizes[idx]

            # Single sample exceeds limit - put in own batch
            if sample_points > self.max_points_per_batch:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_points = 0
                batches.append([idx])
                continue

            # Check if adding would exceed limit
            if current_points + sample_points > self.max_points_per_batch:
                batches.append(current_batch)
                current_batch = [idx]
                current_points = sample_points
            else:
                current_batch.append(idx)
                current_points += sample_points

        # Add remaining batch
        if current_batch:
            batches.append(current_batch)

        # Shuffle batch order
        if self.shuffle:
            self.rng.shuffle(batches)
        
        self.num_batches = len(batches)

        return iter(batches)

    def _shuffle_by_groups(self):
        """Shuffle within size-similar groups"""
        indices = self.indices.copy()
        n = len(indices)
        group_size = max(10, n // 100)  # ~100 groups

        shuffled = []
        for i in range(0, n, group_size):
            group = indices[i:i+group_size]
            self.rng.shuffle(group)
            shuffled.extend(group)

        return shuffled

    def __len__(self):
        if self.num_batches > 0:
            return self.num_batches
        # Estimate number of batches
        total_points = sum(self.sizes)
        return total_points // self.max_points_per_batch + 1


class ShardedBatchSampler:
    def __init__(self, base_sampler, num_replicas, rank, seed=1003):
        self.base = base_sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.all_batches = []
        self.num_samples_per_replica = 0

    def set_epoch(self, epoch: int):
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

        batches = list(iter(self.base))
        n = len(batches)
        pad = (-n) % self.num_replicas

        if pad:
            rng = random.Random(self.seed + epoch)
            batches.extend(rng.choices(batches, k=pad))

        self.all_batches = batches
        self.num_samples_per_replica = len(batches) // self.num_replicas

    def __iter__(self):
        for i in range(self.rank, len(self.all_batches), self.num_replicas):
            yield self.all_batches[i]

    def __len__(self):
        if self.num_samples_per_replica > 0:
            return self.num_samples_per_replica
        base_len = len(self.base)
        return (base_len + self.num_replicas - 1) // self.num_replicas


def concat_collate_fn(batch):
    """
    'batch' 是一个列表, 列表中的 's' 格式如下:
    {
        'pointcloud': {
            'coord': np.ndarray (Ni, 3),
            'feat': np.ndarray (Ni, 9),
        },
        'labels': torch.tensor (Ni,),
        'scene_name': str,
        ...
    }
    
    此函数负责合并(concatenate)返回最终的批处理字典。
    """
    coords, feats, labels = [], [], []
    offsets = [0]

    for s in batch:
        data_dict = s['pointcloud']
        coord = data_dict.get('coord')
        num_points = coord.shape[0]

        coords.append(coord)
        feats.append(data_dict.get('feat'))
        labels.append(s['labels'])
        
        offsets.append(offsets[-1] + num_points)

    coord_cat = np.concatenate(coords, axis=0).astype(np.float32)
    feat_cat = np.concatenate(feats, axis=0).astype(np.float32)
    label_cat = torch.cat(labels, dim=0).to(torch.int64)
    offsets_np = np.asarray(offsets, dtype=np.int32)
    final_points_dict = {
        'coord': coord_cat,
        'feat': feat_cat,
        'offset': offsets_np[1:]
    }

    return {
        'points_dict': final_points_dict,
        'labels': label_cat,
        'offsets_np': offsets_np
    }
