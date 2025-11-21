import os
import sys
import yaml
import numpy as np
import logging
from tqdm import tqdm
import argparse
import random
import math
from datetime import timedelta
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext
import torch.profiler

from src.dataset.s3dis import S3dis_Dataset
from src.dataset.custom_sampler import AdaptiveBatchSampler, ShardedBatchSampler, concat_collate_fn
from src.dataset.transform import GridSample, ToTensor
from src.network import SPANetwork
from src.total_loss import TotalSemanticLoss
from src.metrics import compute_metrics


DATASET_REGISTRY = {
    's3dis': S3dis_Dataset,
    'vkitti': Vkitti_Dataset
}


def apply_epoch_schedules(model, criterion, cfg, epoch, writer=None):
    def _set_attr(root, path_list, value):
        obj = root
        for p in path_list[:-1]:
            if not hasattr(obj, p):
                return False
            obj = getattr(obj, p)
        if hasattr(obj, path_list[-1]):
            setattr(obj, path_list[-1], float(value) if isinstance(getattr(obj, path_list[-1]), (int, float)) else value)
            return True
        return False

    def _apply_param(path, value, apply_mode='set'):
        # path prefix: 'criterion.' or 'model.'
        if path.startswith('criterion.'):
            ok = _set_attr(criterion, path.split('.')[1:], value)
        elif path.startswith('model.'):
            ok = _set_attr(model,     path.split('.')[1:], value)
        else:
            ok = False
        if not ok:
            print(f"Warning: schedule path '{path}' not found in model or criterion")

    def _eval_schedule(spec, t):
        mode = str(spec.get('mode')).lower()
        if mode == 'const':
            return float(spec['value']), 'set'
        if mode == 'linear':
            a, b = float(spec['start']), float(spec['end'])
            return a + (b - a) * t, 'set'
        if mode == 'cosine':
            a, b = float(spec['start']), float(spec['end'])
            return b + 0.5*(a - b)*(1 + math.cos(math.pi * t)), 'set'
        if mode == 'exp':
            a, b = float(spec['start']), float(spec['end'])
            if a <= 0 or b <= 0:
                return a + (b - a) * t, 'set'
            ratio = b / a
            return a * (ratio ** t), 'set'
        if mode == 'poly':
            a, b = float(spec['start']), float(spec['end'])
            pwr = float(spec.get('power', 2.0))
            return a + (b - a) * (t ** pwr), 'set'
        if mode == 'step':
            bounds = spec.get('boundaries', [])
            vals   = spec.get('values', [])
            assert len(vals) == len(bounds) + 1, "step: len(values)=len(boundaries)+1"
            idx = 0
            while idx < len(bounds) and t >= float(bounds[idx]):
                idx += 1
            return float(vals[idx]), 'set'
        if mode == 'scale':
            return float(spec.get('value')), 'mul'
        
    def log_schedule(name, value):
        # Only log if the writer exists (i.e., on the main process)
        if writer is not None:
            writer.add_scalar(f'schedules/{name}', value, epoch)

    E = int(cfg['optimizer']['epochs'])
    ratio = epoch / max(1, E - 1)

    stage_cfg = cfg.get('stage_schedules')
    if stage_cfg and 'stages' in stage_cfg:
        stages = stage_cfg['stages']
        for st in stages:
            s, e = st['span']
            s, e = float(s), float(e)
            if s <= ratio <= e + 1e-12:
                dur = max(1e-8, e - s)
                t_in = (ratio - s) / dur
                for path, spec in st.get('params').items():
                    val, apply_mode = _eval_schedule(spec, t_in)
                    if apply_mode == 'mul':
                        def _get_attr(root, path_list):
                            obj = root
                            for p in path_list[:-1]:
                                if not hasattr(obj, p): return None
                                obj = getattr(obj, p)
                            return getattr(obj, path_list[-1], None)
                        if path.startswith('criterion.'):
                            cur = _get_attr(criterion, path.split('.')[1:])
                        if path.startswith('model.'):
                            cur = _get_attr(model, path.split('.')[1:])
                        if cur is not None and isinstance(cur, (int, float)):
                            _apply_param(path, float(cur) * float(val))
                            log_schedule(path, float(cur) * float(val))
                    else:
                        _apply_param(path, val)
                        log_schedule(path, val)
                break
        return


class UncertaintyWeighter(torch.nn.Module):
    def __init__(
        self,
        loss_names,
        weight_attrs,
        init_value=0.0,
        zero_shuts_reg=True
    ):
        super().__init__()
        assert len(loss_names) == len(weight_attrs), "loss_names 与 weight_attrs 数量需一致"
        self.loss_names   = list(loss_names)
        self.weight_attrs = list(weight_attrs)
        self.zero_shuts_reg = bool(zero_shuts_reg)

        for nm in self.loss_names:
            pname = f'log_sigma_{nm}'
            self.register_parameter(pname, torch.nn.Parameter(torch.tensor(float(init_value))))

    def _get_logparam(self, name):
        return getattr(self, f'log_sigma_{name}')

    def forward(self, loss_dict, criterion):
        total = 0.0
        details = {}

        for nm, wattr in zip(self.loss_names, self.weight_attrs):
            if nm not in loss_dict:
                continue
            Li = loss_dict[nm]
            gamma_val = float(getattr(criterion, wattr))
            gamma = torch.as_tensor(gamma_val, dtype=Li.dtype, device=Li.device)

            logp = self._get_logparam(nm)
            scale = 0.5 * torch.exp(-2.0 * logp)   # 1/(2 σ^2)
            reg   = logp                           # + log σ

            if self.zero_shuts_reg and gamma_val == 0.0:
                total = total + 0.0 * Li
            else:
                total = total + gamma * scale * Li + reg

            details[nm] = dict(
                raw=float(Li.detach().cpu()),
                gamma=gamma_val,
                log_param=float(logp.detach().cpu()),
                scale=float(scale.detach().cpu()),
                weight=float(gamma_val * scale.detach().cpu())
            )

        return total, details


class Trainer:
    """Main trainer class for SPANetwork"""

    def __init__(self, cfg):
        self.setup_distributed()

        self.cfg = cfg
        self.output_dir = self.cfg['experiment']['output_dir']

        if self.is_main():
            self.setup_directories()
        if self.distributed:
            dist.barrier()

        if self.is_main():
            tb_log_dir = os.path.join(self.output_dir, 'tensorboard_logs')
            self.writer = SummaryWriter(tb_log_dir)
            print(f"TensorBoard logs will be saved to: {tb_log_dir}")
        else:
            self.writer = None

        # Setup logging
        self.setup_logging()

        # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        # Setup data
        self.setup_data()

        # Setup model
        self.setup_model()

        # Setup optimization
        self.setup_optimization()

        # Setup training state
        self.start_epoch = 0
        self.best_val = float('inf')
        self.global_step = 0

        self._profile_sync_steps = 0
        self._profile_sync_ctr = 0

        # Load checkpoint if specified
        resume = self.cfg['checkpoint'].get('resume', None)
        if resume:
            self._load_checkpoint(resume)

    def setup_distributed(self):
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self.distributed = self.local_rank != -1
        if not self.distributed:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.distributed:
            try:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")
                dist.init_process_group(backend='nccl', timeout=timedelta(seconds=600))
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()
                print(f"Initialized distributed process local_rank={self.local_rank} on device cuda:{self.device}")
            except Exception as e:
                print(f"Failed to initialize distributed: {e}")
                raise

    def is_main(self):
        return (not self.distributed) or self.rank == 0

    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)

        # Save config
        with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump(self.cfg, f)

    def setup_logging(self):
        """A robust, DDP-aware logging setup."""
        logger = logging.getLogger("SPLearningTrainer")
        logger.propagate = False
        logger.handlers = []

        log_level = logging.INFO if self.is_main() else logging.WARNING
        logger.setLevel(log_level)

        formatter = logging.Formatter(
            f'[RANK {self.rank if self.distributed else 0}] %(asctime)s - %(levelname)s - %(message)s'
        )

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        log_file = os.path.join(self.output_dir, f'train_rank{self.rank if self.distributed else 0}.log')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        self.logger = logger

    def setup_data(self):
        """Setup datasets and dataloaders"""
        data_cfg = self.cfg['data']

        dataset_name = data_cfg.get('dataset_name')
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset_name: '{dataset_name}'. "
                f"Available options are: {list(DATASET_REGISTRY.keys())}"
            )
        DatasetClass = DATASET_REGISTRY[dataset_name]
        
        max_points_per_batch = data_cfg.get('max_points_per_batch')

        self.train_ds = DatasetClass(
            data_cfg['root'],
            self.logger,
            'train',
            grid_sample_cfg=data_cfg.get('grid_sample'),
            transform_cfg=data_cfg.get('train_transform')
        )
        self.val_ds = DatasetClass(
            data_cfg['root'],
            self.logger,
            'val',
            grid_sample_cfg=data_cfg.get('grid_sample'),
            transform_cfg=data_cfg.get('val_transform')
        )

        if self.distributed:
            # Ensure dataset can be evenly divided
            min_samples_per_gpu = len(self.train_ds) // self.world_size
            if min_samples_per_gpu < 1:
                raise ValueError(f"Not enough samples ({len(self.train_ds)}) for {self.world_size} GPUs")
        
        train_sampler = None
        val_sampler = AdaptiveBatchSampler(
            self.val_ds,
            max_points_per_batch=max_points_per_batch,
            shuffle=False,
            seed=self.cfg['experiment'].get('seed', 1003)
        )

        if self.distributed:
            train_sampler = DistributedSampler(
                self.train_ds,
                shuffle=True,
                drop_last=False # DDP标准做法。如果数据量不能被GPU整除，它会填充样本
            )
            val_sampler = ShardedBatchSampler(val_sampler, self.world_size, self.rank, seed=self.cfg['experiment'].get('seed', 1003))

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=data_cfg.get('train_batch_size'),
            sampler=train_sampler,
            num_workers=data_cfg['num_workers'],
            pin_memory=data_cfg.get('pin_memory', True),
            persistent_workers=data_cfg.get('persistent_workers', True),
            collate_fn=concat_collate_fn
        )

        self.val_loader = DataLoader(
            self.val_ds,
            batch_sampler=val_sampler,
            num_workers=data_cfg['num_workers'],
            pin_memory=data_cfg.get('pin_memory', True),
            persistent_workers=data_cfg.get('persistent_workers', True),
            collate_fn=concat_collate_fn
        )

        self.grid_sampler_val = GridSample(
            grid_size=data_cfg['grid_sample'].get('grid_size'),
            mode='test',
            return_inverse=True,
        )

        self.logger.info(f"Dataset name: {dataset_name}")
        self.logger.info(f"Train samples: {len(self.train_ds)}, Est. batches: {len(train_sampler)}")
        self.logger.info(f"Val samples: {len(self.val_ds)}, Est. batches: {len(val_sampler)}")
        self.logger.info(f"Max points per batch: {max_points_per_batch}")

    def setup_model(self):
        """Setup model and loss"""

        # Create model
        model = SPANetwork(self.cfg['model']).to(self.device)
        if self.distributed:
            self.model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            self.model = model
        self.unwrapped_model = self.model.module if self.distributed else self.model

        # Log model size
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {n_params:,}")

        # Create loss criterion
        loss_cfg = self.cfg['loss']
        self.criterion = TotalSemanticLoss(
            num_classes=loss_cfg['num_classes'],
            ignore_label=loss_cfg['ignore_label'],
            w_ce=loss_cfg['w_ce'],
            w_cbl=loss_cfg['w_cbl'],
            w_sp_disc=loss_cfg['w_sp_disc'],
            sp_margin_pull=loss_cfg['sp_margin_pull'],
            sp_margin_push=loss_cfg['sp_margin_push'],
            sp_weight_pull=loss_cfg['sp_weight_pull'],
            sp_weight_push=loss_cfg['sp_weight_push'],
            cbl_temperature=loss_cfg['cbl_temperature']
        ).to(self.device)

    def setup_optimization(self):
        """Setup optimizer, scheduler, and AMP"""
        opt_cfg = self.cfg['optimizer']

        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg['base_lr'],
            betas=tuple(opt_cfg['betas']),
            weight_decay=opt_cfg['weight_decay']
        )

        self.grad_clip = opt_cfg.get('grad_clip_norm', None)

        # Setup EMA if enabled
        self.ema = None
        if opt_cfg.get('ema', {}).get('enable', False):
            import copy
            self.ema = copy.deepcopy(self.unwrapped_model).to(self.device).eval()
            for p in self.ema.parameters():
                p.requires_grad = False
            self.ema_decay = float(opt_cfg['ema']['decay'])

        # Setup AMP
        amp_cfg = self.cfg['amp']
        self.use_amp = bool(amp_cfg.get('enable', True))
        self.amp_dtype = str(amp_cfg.get('dtype', 'bfloat16'))

        # Use GradScaler only for float16
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp and self.amp_dtype == 'float16'
        )
        self.use_bf16 = self.use_amp and self.amp_dtype == 'bfloat16'

        if self.use_bf16:
            self.ctx = autocast(dtype=torch.bfloat16)
        else:
            self.ctx = autocast(enabled=self.use_amp and self.amp_dtype == 'float16')

        # Setup learning rate schedule parameters
        self.total_epochs = int(opt_cfg['epochs'])
        self.warmup_epochs = int(opt_cfg['warmup_epochs'])
        self.maintain_epochs = int(opt_cfg['maintain_epochs'])
        self.min_lr = float(opt_cfg['min_lr'])
        self.base_lr = float(opt_cfg['base_lr'])

        uncfg = self.cfg.get('uncertainty')
        self.use_uncertainty = bool(uncfg.get('enable'))
        if self.use_uncertainty:
            self.unc_loss_names = uncfg.get('loss_names')
            self.unc_weight_attrs = uncfg.get('weight_attrs')

            self.unc_weighter = UncertaintyWeighter(
                self.unc_loss_names,
                self.unc_weight_attrs,
                init_value=uncfg.get('init_value', 0.0),
                zero_shuts_reg=uncfg.get('zero_shuts_reg', True)
            ).to(self.device)

            self.optimizer.add_param_group({
                'params': self.unc_weighter.parameters(),
                'weight_decay': 0.0,
                'lr': self.optimizer.param_groups[0]['lr'],
            })

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        if self.is_main():
            ckpt_dir = os.path.join(self.output_dir, 'ckpts')

            ckpt = {
                'epoch': epoch,
                'model': self.unwrapped_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_val': self.best_val,
                'config': self.cfg,
                'global_step': self.global_step
            }

            if self.ema is not None:
                ckpt['ema'] = self.ema.state_dict()

            if is_best:
                path = os.path.join(ckpt_dir, 'best.pth')
                torch.save(ckpt, path)
                self.logger.info(f"Saved best model (val_loss={self.best_val:.6f})")

            if (epoch + 1) % self.cfg['checkpoint']['save_every_epochs'] == 0:
                path = os.path.join(ckpt_dir, f'epoch{(epoch+1):03d}.pth')
                torch.save(ckpt, path)
                self.logger.info(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str):
        """Load model checkpoint"""
        self.logger.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, map_location=self.device)

        self.unwrapped_model.load_state_dict(ckpt['model'], strict=False)
        # self.optimizer.load_state_dict(ckpt['optimizer'])
        # self.start_epoch = int(ckpt.get('epoch', 0)) + 1
        # self.best_val = float(ckpt.get('best_val', float('inf')))
        # self.global_step = int(ckpt.get('global_step', 0))

        if 'ema' in ckpt and self.ema is not None:
            self.ema.load_state_dict(ckpt['ema'], strict=False)

        self.logger.info(f"Loading checkpoint from {path}")

    def _lr_for_epoch(self, epoch: int):
        if epoch < self.warmup_epochs:
            return self.base_lr * float(epoch + 1) / float(max(1, self.warmup_epochs))
        elif 0 <= epoch - self.warmup_epochs < self.maintain_epochs:
            return self.base_lr
        else:
            t = (epoch - self.warmup_epochs - self.maintain_epochs) / float(self.total_epochs - self.warmup_epochs - self.maintain_epochs - 1)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))

    def _set_epoch_lr(self, epoch: int):
        """Set learning rate for current epoch"""
        lr = self._lr_for_epoch(epoch)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr
    
    def _make_profiler(self, epoch):
        want_profile = epoch in self.cfg['logging']['epochs_to_profile']
        if want_profile:
            self._profile_sync_steps = 2 + 2 + 2 + 1   # wait + warmup + active + post
            self._profile_sync_ctr = 0
        else:
            self._profile_sync_steps = 0
            self._profile_sync_ctr = 0

        if want_profile and self.is_main():
            self.logger.info(f"Setting up profiler for epoch {epoch}.")
            schedule = torch.profiler.schedule(wait=2, warmup=2, active=2, repeat=1)
            handler = torch.profiler.tensorboard_trace_handler(
                os.path.join(self.output_dir, 'tensorboard_logs'),
                worker_name=f"rank_{self.rank}",
            )
            return torch.profiler.profile(
                schedule=schedule,
                on_trace_ready=handler,
                record_shapes=True,
                profile_memory=True,
                with_stack=False
            )
        else:
            return nullcontext()

    def train_epoch(self, epoch: int, prof=None):
        """Train for one epoch"""
        self.model.train()

        # Set learning rate
        lr = self._set_epoch_lr(epoch)

        # Apply scheduled parameter changes
        apply_epoch_schedules(self.unwrapped_model, self.criterion, self.cfg, epoch, self.writer)

        # Training loop
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} LR {lr:.2e}", disable=not self.is_main())
        for batch_idx, batch in enumerate(pbar):
            try:
                pc_dict = batch['points_dict']
                offsets_np = batch['offsets_np']
                labels = batch['labels'].to(self.device, non_blocking=True)

                pc_dict.update({'grid_size': self.cfg['data']['grid_sample']['grid_size']})

                fwd_oom_local = 0
                try:
                    with self.ctx:
                        outputs = self.model(pc_dict, offsets_np)
                        loss_dict = self.criterion(outputs, labels)

                        if self.use_uncertainty:
                            loss, unc_details = self.unc_weighter(loss_dict, criterion=self.criterion)
                            if self.is_main() and (self.global_step % 50 == 0):
                                for nm, d in unc_details.items():
                                    self.writer.add_scalar(f'uncertainty/log_param/{nm}', d['log_param'], self.global_step)
                                    self.writer.add_scalar(f'uncertainty/gamma/{nm}',     d['gamma'],     self.global_step)
                                    self.writer.add_scalar(f'uncertainty/scale/{nm}',     d['scale'],     self.global_step)
                                    self.writer.add_scalar(f'uncertainty/weight/{nm}',    d['weight'],    self.global_step)
                        else:
                            loss = loss_dict['total_loss']
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        fwd_oom_local = 1
                        torch.cuda.empty_cache()
                    else:
                        raise

                skip_forward = False
                if self.distributed and dist.is_initialized():
                    fwd_flag = torch.tensor([fwd_oom_local], device=self.device, dtype=torch.int)
                    dist.all_reduce(fwd_flag, op=dist.ReduceOp.SUM)
                    skip_forward = int(fwd_flag.item()) > 0
                else:
                    skip_forward = (fwd_oom_local == 1)

                if skip_forward:
                    if self.is_main():
                        self.logger.warning("OOM in forward/loss; skipping batch (synced across ranks).")
                    continue

                self.optimizer.zero_grad(set_to_none=True)

                try:
                    if self.scaler.is_enabled():
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        raise RuntimeError("Backward OOM (abort run)")
                    else:
                        raise

                if self.scaler.is_enabled():
                    if self.grad_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                if self.ema is not None:
                    with torch.no_grad():
                        for p_ema, p in zip(self.ema.parameters(), self.unwrapped_model.parameters()):
                            p_ema.mul_(self.ema_decay).add_(p, alpha=1.0 - self.ema_decay)

                if self.is_main():
                    for k, v in loss_dict.items():
                        if "loss" in k:
                            if isinstance(v, torch.Tensor):
                                v = v.detach().item()
                            self.writer.add_scalar(f"train/{k}", float(v), self.global_step)
                    self.writer.add_scalar("train/learning_rate", lr, self.global_step)

                self.global_step += 1
                epoch_loss += float(loss.detach().item())
                n_batches += 1

                if batch_idx % self.cfg['logging']['log_interval'] == 0:
                    pbar.set_postfix(loss=epoch_loss / max(n_batches, 1))

                if (batch_idx % 10 == 0) and torch.cuda.is_available() and self.is_main():
                    s = torch.cuda.memory_stats()
                    alloc = s["allocated_bytes.all.current"]
                    reserv= s["reserved_bytes.all.current"]
                    inact = s["inactive_split_bytes.all.current"]
                    self.logger.info(f"Mem (MB) Allocated: {alloc/1e6:.1f}, Reserved: {reserv/1e6:.1f}, Inactive: {inact/1e6:.1f}")

            finally:
                if prof:
                    prof.step()
                if self.distributed and self._profile_sync_ctr < self._profile_sync_steps:
                    dist.barrier()
                    self._profile_sync_ctr += 1

        return epoch_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate on all GPUs and aggregate results"""
        model_to_eval = self.ema if self.ema is not None else self.unwrapped_model
        model_to_eval.eval()
        model_to_eval.sp_1.target_sp_size = self.cfg['validation']['target_sp_size']
        model_to_eval.sp_1.target_max_sp_cc = self.cfg['validation']['target_max_sp_cc']
        model_to_eval.sp_1.edge_threshold = self.cfg['validation']['edge_threshold']
        model_to_eval.k_p = self.cfg['validation']['k_p']
        model_to_eval.sp_1.k_seed = self.cfg['validation']['k_seed']
        model_to_eval.sp_1.group_points_cap = self.cfg['validation']['group_points_cap']
        model_to_eval.strong_k_min = self.cfg['validation']['strong_k_min']
        model_to_eval.strong_k_max = self.cfg['validation']['strong_k_max']
        model_to_eval.sp_1.w_cos = self.cfg['validation']['w_cos']
        model_to_eval.sp_1.w_geo = self.cfg['validation']['w_geo']

        model_to_eval.use_dgl_for_scores = bool(self.cfg['validation']['use_dgl_for_scores'])
        model_to_eval.edge_chunk = int(self.cfg['validation']['edge_chunk'])
        model_to_eval.channel_chunk = int(self.cfg['validation']['channel_chunk'])
        model_to_eval.scores_use_fp16 = bool(self.cfg['validation']['scores_use_fp16'])

        total_w  = 0.0
        total_wo = 0.0
        total_ce_loss = 0.0
        total_cbl_loss = 0.0
        total_spd_loss = 0.0
        n_batches = 0
        n_samples = 0
        total_ooa = 0.0
        total_br = 0.0
        total_bp = 0.0
        total_spcounts = 0
        total_sp_sizes = 0

        pbar = tqdm(self.val_loader, desc="Validation", disable=not self.is_main())
        for batch in pbar:
            try:
                points_dict = batch['points_dict']
                labels = batch['labels'].to(self.device, non_blocking=True)
                offsets_np = batch['offsets_np']
                coord = torch.from_numpy(points_dict['coord']).to(self.device, non_blocking=True)
                points_dict.update({'grid_size': self.cfg['data']['grid_sample']['grid_size']})

                with self.ctx:
                    outputs = model_to_eval(points_dict, offsets_np)
                    loss_dict = self.criterion(outputs, labels)
                    num_superpoints = outputs['num_superpoints']
                    cluster_sizes = outputs['cluster_sizes']
                    samples_in_batch = offsets_np.shape[0] - 1

                    if getattr(self, "use_uncertainty", False):
                        loss_w, _ = self.unc_weighter(loss_dict, criterion=self.criterion)
                        val_loss_w = float(loss_w.item())
                        total_w  += val_loss_w

                    val_loss_wo = float(loss_dict['total_loss'].item())
                    total_ce_loss += float(loss_dict['ce_loss'].item())
                    total_cbl_loss += float(loss_dict['cbl_loss'].item())
                    total_spd_loss += float(loss_dict['sp_disc_loss'].item())

                total_wo += val_loss_wo
                total_spcounts += int(num_superpoints)
                total_sp_sizes += int(cluster_sizes.sum().item())
                n_batches += 1
                n_samples += samples_in_batch

                metrics = compute_metrics(outputs, labels, coord, offsets_np, sp_offsets=None, k=8, tolerance=1, device=self.device)
                OOA = metrics['OOA']
                BR = metrics['BR']
                BP = metrics['BP']
                total_ooa += float(OOA)
                total_br += float(BR)
                total_bp += float(BP)

            except Exception as e:
                msg = str(e)
                if "out of memory" in msg.lower():
                    self.logger.warning("OOM in validation, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        if self.distributed and torch.distributed.is_initialized():
            pair = torch.tensor([total_w, total_wo, total_ce_loss, total_cbl_loss, total_spd_loss, total_ooa, total_br, total_bp, n_batches, n_samples, total_spcounts, total_sp_sizes], device=self.device)
            torch.distributed.all_reduce(pair, op=torch.distributed.ReduceOp.SUM)
            total_w, total_wo, total_ce_loss, total_cbl_loss, total_spd_loss, total_ooa, total_br, total_bp, n_batches, n_samples, total_spcounts, total_sp_sizes = pair[0].item(), pair[1].item(), pair[2].item(), pair[3].item(), pair[4].item(), pair[5].item(), pair[6].item(), pair[7].item(), int(pair[8].item()), int(pair[9].item()), int(pair[10].item()), int(pair[11].item())

        val_loss_w  = total_w  / max(n_batches, 1)
        val_loss_wo = total_wo / max(n_batches, 1)
        val_loss_ce = total_ce_loss / max(n_batches, 1)
        val_loss_cbl= total_cbl_loss / max(n_batches, 1)
        val_loss_spd= total_spd_loss / max(n_batches, 1)
        val_ooa = total_ooa / max(n_batches, 1)
        val_br = total_br / max(n_batches, 1)
        val_bp = total_bp / max(n_batches, 1)
        val_avg_spcount = total_spcounts / max(n_samples, 1)
        val_avg_spsize = total_sp_sizes / max(total_spcounts, 1)

        self.logger.info(f"[Val] Epoch {epoch} loss_uncertainty={val_loss_w:.6f}  loss_wonly={val_loss_wo:.6f}  OOA={val_ooa:.6f}  BR={val_br:.6f}  BP={val_bp:.6f}  AvgSPCount={val_avg_spcount:.2f}  AvgSPSize={val_avg_spsize:.2f}")
        if self.is_main():
            self.writer.add_scalar('val/total_loss_uncertainty', val_loss_w, epoch)
            self.writer.add_scalar('val/total_loss_wonly', val_loss_wo, epoch)
            self.writer.add_scalar('val/ce_loss', val_loss_ce, epoch)
            self.writer.add_scalar('val/cbl_loss', val_loss_cbl, epoch)
            self.writer.add_scalar('val/spd_loss', val_loss_spd, epoch)
            self.writer.add_scalar('val/OOA', val_ooa, epoch)
            self.writer.add_scalar('val/BR', val_br, epoch)
            self.writer.add_scalar('val/BP', val_bp, epoch)

        return val_loss_wo

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")

        for epoch in range(self.start_epoch, self.total_epochs):
            self.train_loader.sampler.set_epoch(epoch)
            
            # Train
            with self._make_profiler(epoch) as prof:
                train_loss = self.train_epoch(epoch, prof=prof)

            self.logger.info(f"[Train] Epoch {epoch} loss={train_loss:.6f}")

            # Validate
            if (epoch + 1) % self.cfg['logging']['val_every_epochs'] == 0:
                if self.distributed:
                    dist.barrier()
                self.val_loader.batch_sampler.set_epoch(epoch)
                val_loss = self.validate(epoch)

                # Check if best
                is_best = val_loss < self.best_val
                if is_best:
                    self.best_val = val_loss

                # Save checkpoint
                self._save_checkpoint(epoch, is_best=is_best)
            else:
                # Save checkpoint without validation
                self._save_checkpoint(epoch, is_best=False)
                # pass

        self.logger.info("Training completed!")

    def _shutdown_dataloader(self, loader):
        try:
            it = getattr(loader, "_iterator", None)
            if it is not None:
                it._shutdown_workers()
        except Exception:
            pass

    def cleanup(self):
        if self.is_main():
            self.logger.info(f"Cleanup resources...")

        for loader in (getattr(self, "train_loader", None), getattr(self, "val_loader", None)):
            if loader is not None:
                self._shutdown_dataloader(loader)

        try:
            if getattr(self, "writer", None):
                self.writer.flush()
                self.writer.close()
        except Exception:
            pass

        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='Train SPA model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    rank = int(os.environ.get('RANK', 0))
    seed = int(cfg.get('experiment', {}).get('seed', 1003)) + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    trainer = Trainer(cfg)

    try:
        trainer.train()
    except BaseException as e:
        trainer.logger.error(f"An error occurred during training: {e}", exc_info=True)
        try:
            trainer.cleanup()
        finally:
            os._exit(1)
    else:
        trainer.cleanup()
        sys.exit(0)


if __name__ == '__main__':
    main()
