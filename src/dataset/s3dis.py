import os
import glob
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from .transform import GridSample, Compose


class S3dis_Dataset(Dataset):
    def __init__(self, data_root: str, logger: logging.Logger, 
                 mode: str = 'train', 
                 grid_sample_cfg: dict = None, 
                 transform_cfg: dict = None,
                 test_area_idx: int = 5):
        self.data_root = data_root
        self.mode = mode
        self.test_area_idx = test_area_idx
        self.samples = [] 
        self.logger = logger

        if transform_cfg is None:
            self.logger.warning(f"[S3dis_Dataset] 'transform_cfg' is None. Using raw data.")
            self.transform = None
        else:
            self.transform = Compose(transform_cfg)

        if grid_sample_cfg is None:
            self.logger.error("[S3dis_Dataset] 'grid_sample_cfg' is missing.")
            raise ValueError("S3dis_Dataset requires grid_sample_cfg.")

        gs_cfg_copy = grid_sample_cfg.copy()
        try:
            self.grid_sampler = GridSample(**gs_cfg_copy)
        except Exception as e:
            self.logger.error(f"[S3dis_Dataset] Failed to init GridSample: {e}")
            raise

        self.logger.info(f"[S3dis_Dataset] Init: mode={mode}, test_area=Area_{test_area_idx}")
        all_npy_files = glob.glob(os.path.join(self.data_root, "*.npy"))
        scene_files_to_load = []
        
        test_area_str = f"Area_{test_area_idx}"
        
        for f_path in all_npy_files:
            scene_filename = os.path.basename(f_path)
            is_test_scene = scene_filename.startswith(test_area_str)
            
            if self.mode == 'train':
                if not is_test_scene:
                    scene_files_to_load.append(f_path)
            elif self.mode == 'val':
                if is_test_scene:
                    scene_files_to_load.append(f_path)
        
        self.logger.info(f"[S3dis_Dataset] Found {len(scene_files_to_load)} scenes for {mode}.")

        for pc_path in scene_files_to_load:
            scene_name = os.path.splitext(os.path.basename(pc_path))[0]
            try:
                mmap_data = np.load(pc_path, mmap_mode='r')
                if mmap_data.shape[0] == 0:
                    self.logger.warning(f"[S3dis_Dataset] 场景 {scene_name} (train) 为空。跳过。")
                    continue
                if mmap_data.shape[1] < 3:
                    self.logger.error(f"场景 {scene_name} (train) 数据列数 < 3。跳过。")
                    continue

                if self.mode == 'val':
                    num_voxels, _, _ = self.grid_sampler._count_voxels(mmap_data[:, :3])
                else:
                    num_voxels = 0 

                self.samples.append({
                    'scene_name': scene_name,
                    'chunk_name': scene_name,
                    'pc_path': pc_path,
                    'num_points': num_voxels,
                    'tile_indices': None
                })
            except Exception as e:
                self.logger.error(f"[S3dis_Dataset] Failed to load {pc_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        try:
            scene_data_mmap = np.load(sample['pc_path'], mmap_mode='r')

            if self.mode == 'train':
                sampled_data_dict = {
                    'coord': scene_data_mmap[:, :3].copy(),
                    'color': scene_data_mmap[:, 3:6].copy(),
                    'normal': scene_data_mmap[:, 6:9].copy(),
                    'label': scene_data_mmap[:, -1].astype(np.int64).copy()}
            else: # self.mode == 'val'
                num_voxels, inverse, count = self.grid_sampler._count_voxels(scene_data_mmap[:, :3])

                original_labels = scene_data_mmap[:, -1].astype(np.int64)
                num_classes = original_labels.max() + 1 
                voxel_label_counts = np.zeros((num_voxels, num_classes), dtype=np.int64)
                np.add.at(voxel_label_counts, (inverse, original_labels), 1)
                voxel_modes = np.argmax(voxel_label_counts, axis=1).astype(np.int64)

                features_to_average = scene_data_mmap[:, :9]
                voxel_sums = np.zeros((num_voxels, 9), dtype=np.float32)
                np.add.at(voxel_sums, inverse, features_to_average)
                count_broadcast = count[:, np.newaxis]
                count_broadcast[count_broadcast == 0] = 1
                voxel_means = voxel_sums / count_broadcast

                sampled_data_dict = {
                    'coord': voxel_means[:, :3].astype(np.float32),
                    'color': voxel_means[:, 3:6].astype(np.float32),
                    'normal': voxel_means[:, 6:9].astype(np.float32),
                    'label': voxel_modes
                }

            if self.transform is not None:
                data_dict = self.transform(sampled_data_dict)

            labels_t = torch.from_numpy(data_dict['label'])
            
            return {
                'pointcloud': data_dict,
                'labels':     labels_t,
                'scene_name': sample['scene_name'],
                'chunk_name': sample['chunk_name'],
                'num_points': int(labels_t.shape[0])
            }
        except Exception as e:
            self.logger.error(f"__getitem__ 失败, idx: {idx}, sample: {sample.get('chunk_name')}", exc_info=True)
            return {
                'pointcloud': {},
                'labels':     torch.tensor([], dtype=torch.int64),
                'scene_name': "ERROR",
                'chunk_name': "ERROR",
                'num_points': 0
            }
