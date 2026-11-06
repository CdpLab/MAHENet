"""Single-factor rare-class CE DDP experiment with real checkpoints."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import yaml
from .branch_training import ROOT, TASK, PROTOCOL_TASK, build, combined_loss, evaluate37, digest, gradient_groups
from .train_runtime import validate_config, verify_frozen
from .train_runtime import validate_resume
from .data_base import DMS37Dataset
from .branch_training import make_dataset, move_input, batch_size

def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)

def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)

def restore_rng(value, rank):
    r = value[rank]
    random.setstate(r['python'])
    np.random.set_state(r['numpy'])
    torch.set_rng_state(r['torch'])
    torch.cuda.set_rng_state(r['cuda'])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--initial-validation-only', action='store_true')
    a = p.parse_args()
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    rank = dist.get_rank()
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(4)
    c = __import__('mahenet.train_runtime', fromlist=['read']).read(a.config)
    validate_config(c)
    verify_frozen()
    t = c['training']
    data = c['data']
    epochs = t['epochs']
    assert dist.get_world_size() == 2 and epochs == 20 and (data['label_protocol'] == 'dms37')
    assert digest(data['manifest']) == c['manifest_sha256'] and digest(c['model']['pretrained_checkpoint']) == c['pretrained_sha256']
    random.seed(c['seed'] + rank)
    np.random.seed(c['seed'] + rank)
    torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'] + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(a.output)
    checkpoint = torch.load(a.resume, map_location='cpu') if a.resume else None
    if checkpoint is not None:
        validate_resume(checkpoint, c)
    if rank == 0:
        if not checkpoint and out.exists() and any(out.iterdir()):
            raise RuntimeError('Refusing to overwrite ' + str(out))
        out.mkdir(parents=True, exist_ok=True)
        (out / 'config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    dist.barrier()
    model = build(c)
    initial_audit = {'kind': 'fresh author ImageNet initialization', 'source_sha256': c['pretrained_sha256']}
    if checkpoint:
        model.load_state_dict(checkpoint['model'], strict=True)
    random.seed(c['seed'] + rank)
    np.random.seed(c['seed'] + rank)
    torch.manual_seed(c['seed'] + rank)
    torch.cuda.manual_seed_all(c['seed'] + rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=True)
    trainset = make_dataset(c, 'train')
    valset = make_dataset(c, 'validation')
    sampler = DistributedSampler(trainset, num_replicas=2, rank=rank, seed=c['seed'], shuffle=True)
    if rank == 0:
        valgen = torch.Generator().manual_seed(c['seed'] + 90000)
        valloader = DataLoader(valset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, generator=valgen)
    groups = {}
    for (name, param) in model.module.named_parameters():
        part = 'backbone' if name.startswith(('semantic_encoder.', 'backbone.')) else 'head'
        no_decay = param.ndim == 1 or 'norm' in name or getattr(param, '_no_weight_decay', False)
        groups.setdefault((part, no_decay), []).append(param)
    optimizer = torch.optim.AdamW([{'params': params, 'lr': t[part + '_lr'], 'weight_decay': 0.0 if nd else t['weight_decay']} for ((part, nd), params) in groups.items()])
    physical = t['per_gpu_batch_size']
    accum = t['accumulation']
    assert physical * accum == 12
    updates = math.ceil(math.ceil(len(sampler) / physical) / accum)
    assert updates == t['expected_updates_per_epoch']
    total = updates * epochs
    warmup = t['warmup_steps']

    def factor(step):
        return (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    history = []
    best = -1.0
    global_step = 0
    start_epoch = 1
    elapsed = 0.0
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best = checkpoint['best_miou']
        history = json.loads((out / 'history.json').read_text())
        assert [r['epoch'] for r in history] == list(range(1, start_epoch))
        assert global_step == (start_epoch - 1) * updates == scheduler.last_epoch
        elapsed = history[-1]['elapsed_seconds']
        restore_rng(checkpoint['rng_by_rank'], rank)
    if rank == 0 and (not checkpoint):
        atomic_json(out / 'provenance.json', {'initialization': initial_audit, 'planned_training_epochs': epochs, 'run_kind': 'smoke_only_zero_completed_epochs' if a.smoke else 'formal_training', 'protocol_sha256': c['protocol_sha256'], 'selection': 'validation mIoU only', 'loss': c['loss'], 'architecture': c['architecture'], 'paired_data_note': 'Published staged training; validation-mIoU checkpoint selection.', 'experiment': c['experiment'], 'runtime': t, 'parameters': sum((p.numel() for p in model.module.parameters())), 'sources': {'source_snapshot_sha256': digest(TASK / 'source_snapshot.json')}, 'resume_note': 'RNG states saved per rank at epoch boundaries; CUDA/NCCL kernels are not guaranteed bit deterministic.'})
    if a.initial_validation_only:
        dist.barrier()
        if rank == 0:
            v = evaluate37(model.module, valloader, device, include_confusion=True)
            atomic_json(out / 'initial_validation.json', v)
            print(json.dumps({'initial_validation_miou': v['miou']}), flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if hasattr(trainset, 'set_epoch'):
            trainset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(c['seed'] + 100000 * rank + 1000 * epoch)
        loader = DataLoader(trainset, batch_size=physical, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=2, generator=generator)
        loss_sum = 0.0
        sample_count = 0
        checks = []
        torch.cuda.reset_peak_memory_stats()
        for (index, (images, labels, names)) in enumerate(loader):
            gs = index // accum * accum
            count = min(accum, len(loader) - gs)
            update = index - gs + 1 == count
            group_n = sum((min(physical, len(sampler) - j * physical) for j in range(gs, gs + count)))
            images = move_input(images, device)
            labels = labels.to(device, non_blocking=True)
            actual_batch = batch_size(images)
            with nullcontext() if update else model.no_sync():
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=t['precision'] == 'bf16'):
                    outputs = model(images)
                (logits, aux) = outputs[:2]
                (loss, loss_parts) = combined_loss(outputs, labels, c['loss'], global_step)
                assert logits.shape == (actual_batch, 37, 512, 512)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss ' + str(names))
                (loss * (actual_batch / group_n)).backward()
            loss_sum += float(loss) * actual_batch
            sample_count += actual_batch
            if update:
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad):
                    raise FloatingPointError('Nonfinite gradient')
                gradients = {}
                if a.smoke:
                    for prefix in gradient_groups(c['model']['name']):
                        gradients[prefix] = sum((float(p.grad.abs().sum()) for (n, p) in model.module.named_parameters() if n.startswith(prefix) and p.grad is not None))
                    assert all((torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    if global_step >= 1:
                        assert all((v > 0 for v in gradients.values())), gradients
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and (global_step % 25 == 0 or index < accum):
                    print(json.dumps({'epoch': epoch, 'optimizer_step': global_step, 'microbatch': index + 1, 'loss': float(loss), 'grad_norm': float(grad), 'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2 ** 20, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'epoch_elapsed_seconds': time.perf_counter() - epoch_start, **{k: float(v) for (k, v) in loss_parts.items()}}), flush=True)
                if a.smoke:
                    assert all((torch.isfinite(p).all() for p in model.parameters()))
                    checks.append({'step': global_step, 'loss': float(loss), 'grad_norm': float(grad), 'new_module_gradient_l1': gradients, 'epoch_elapsed_seconds': time.perf_counter() - epoch_start, **{k: float(v) for (k, v) in loss_parts.items()}})
                    if len(checks) == 100:
                        break
            del logits, aux, loss, outputs
        if a.smoke:
            torch.cuda.synchronize()
            own = {'rank': rank, 'checks': checks, 'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2 ** 20, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2 ** 20}
            gathered = [None] * 2
            dist.all_gather_object(gathered, own)
            import hashlib
            state_hash = hashlib.sha256()
            for (name, p) in model.module.named_parameters():
                state_hash.update(p.detach().cpu().contiguous().numpy().tobytes())
            own['parameter_sha256'] = state_hash.hexdigest()
            dist.all_gather_object(gathered, own)
            assert len({r['parameter_sha256'] for r in gathered}) == 1
            if rank == 0:
                atomic_json(out / 'gpu_check.json', {'real_updates': len(checks), 'training_epochs_completed': 0, 'model': c['model']['name'], 'physical_batch_per_gpu': physical, 'accumulation': accum, 'ranks': gathered, 'frozen_plan_sha256': digest(TASK / 'frozen_plan.json')})
            dist.barrier()
            dist.destroy_process_group()
            return
        del loader
        reduced = torch.tensor([loss_sum, sample_count], device=device)
        dist.all_reduce(reduced)
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}
        rngs = [None] * 2
        dist.all_gather_object(rngs, rng)
        dist.barrier()
        if rank == 0:
            validation = evaluate37(model.module, valloader, device)
            row = {'epoch': epoch, 'global_step': global_step, 'train_loss': float(reduced[0] / reduced[1]), 'epoch_seconds': time.perf_counter() - epoch_start, 'elapsed_seconds': elapsed + time.perf_counter() - started, **validation}
            history.append(row)
            is_best = row['miou'] > best
            best = max(best, row['miou'])
            value = {'epoch': epoch, 'global_step': global_step, 'config': c, 'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'best_miou': best, 'rng_by_rank': rngs}
            save(out / 'last.pt', value)
            if is_best:
                save(out / 'best.pt', value)
            atomic_json(out / 'history.json', history)
            print(json.dumps({k: row[k] for k in ['epoch', 'miou', 'macc', 'pacc', 'bf1', 'epoch_seconds']}), flush=True)
        dist.barrier()
    dist.destroy_process_group()
if __name__ == '__main__':
    main()
