"""Full20 adaptation with a tested head-only to joint DDP transition."""
import argparse
from datetime import timedelta
import hashlib
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from .train_runtime import TASK, read, atomic, digest, validate, verify
from .model import MAHENet
from .data import PairedDataset, move
from .objective import loss_for
from .validation import shard_indices, evaluate_distributed
from .resume import validate_resume, restore_rng

def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)

def parameter_hash(model):
    h = hashlib.sha256()
    for (_, p) in model.named_parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def wrap(module, local):
    return DDP(module, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume')
    args = parser.parse_args()
    c = read(args.config)
    validate(c)
    verify()
    t = c['training']
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    rank = dist.get_rank()
    local = int(os.environ['LOCAL_RANK'])
    assert dist.get_world_size() == 2
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(4)
    random.seed(3407 + rank)
    np.random.seed(3407 + rank)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407 + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.output)
    saved = torch.load(args.resume, map_location='cpu') if args.resume else None
    history = read(out / 'history.json') if saved else []
    (first, step) = validate_resume(saved, history, c) if saved else (1, 0)
    assert not (saved and args.smoke)
    if rank == 0:
        if saved is None and out.exists() and any(out.iterdir()):
            raise RuntimeError('Refusing overwrite ' + str(out))
        out.mkdir(parents=True, exist_ok=True)
        if saved:
            assert read(out / 'config.json') == c
        else:
            atomic(out / 'config.json', c)
    dist.barrier()
    net = MAHENet(c)
    if saved:
        net.load_state_dict(saved['model'], strict=True)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
    groups = {}
    for (key, p) in net.named_parameters():
        part = 'attention' if key.startswith('new_attention.') else 'backbone' if key.startswith(('global_net.semantic_encoder.', 'local_net.semantic_encoder.')) else 'head'
        nd = p.ndim == 1 or 'norm' in key or getattr(p, '_no_weight_decay', False)
        groups.setdefault((part, nd), []).append(p)
    optimizer = torch.optim.AdamW([{'params': ps, 'lr': t[part + '_lr'], 'weight_decay': 0.0 if nd else t['weight_decay'], 'part': part} for ((part, nd), ps) in groups.items()])
    warm_steps = 2 if args.smoke else 2 * 707
    total = 20 * 707

    def schedule(part):

        def factor(step):
            offset = 0 if part == 'attention' else warm_steps
            if step < offset:
                return 0.0
            s = step - offset
            duration = total - offset
            return (s + 1) / 250 if s < 250 else 0.5 * (1 + math.cos(math.pi * (s - 250) / (duration - 250)))
        return factor
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [schedule(g['part']) for g in optimizer.param_groups])
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        assert scheduler.last_epoch == step
    net.set_phase(saved['head_only'] if saved else True)
    net = wrap(net, local)
    random.seed(3407 + rank)
    np.random.seed(3407 + rank)
    torch.manual_seed(3407 + rank)
    torch.cuda.manual_seed_all(3407 + rank)
    train = PairedDataset(c, 'train')
    val = PairedDataset(c, 'validation')
    assert len(train) == 16968 and len(val) == 4044
    sampler = DistributedSampler(train, num_replicas=2, rank=rank, seed=3407, shuffle=True)
    assert t['per_gpu_batch_size'] == 12 and t['accumulation'] == 1 and (len(sampler) == 8484)
    val_loader = DataLoader(Subset(val, shard_indices(len(val), rank, 2)), batch_size=2, shuffle=False, num_workers=4, pin_memory=True, generator=torch.Generator().manual_seed(93407 + rank), persistent_workers=True, prefetch_factor=2)
    if rank == 0 and saved is None:
        atomic(out / 'provenance.json', {'kind': 'native20update_check' if args.smoke else 'formal20_adaptation', 'segmentation_checkpoints_loaded': True, 'fresh_ImageNet': False, 'config': c, 'parameters': sum((p.numel() for p in net.module.parameters())), 'frozen_plan_sha256': digest(TASK / 'frozen_plan.json')})
    if rank == 0 and saved:
        path = out / 'execution_segments.json'
        segments = read(path) if path.exists() else []
        segments.append({'resumed_checkpoint': str(args.resume), 'saved_epoch': saved['epoch'], 'global_step': step, 'frozen_plan_sha256': digest(TASK / 'frozen_plan.json'), 'time_unix': time.time(), 'validation_workers': 2, 'optimizer_scheduler_rng_restored': True})
        atomic(path, segments)
    best = saved['best_miou'] if saved else -1.0
    elapsed_before = history[-1]['elapsed_seconds'] if saved else 0.0
    if saved:
        restore_rng(saved, rank)
    del saved
    checks = []
    started = time.perf_counter()
    for epoch in range(first, 21):
        epoch_started = time.perf_counter()
        sampler.set_epoch(epoch)
        net.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        loader = DataLoader(train, batch_size=12, sampler=sampler, num_workers=4, pin_memory=True, prefetch_factor=2, generator=torch.Generator().manual_seed(3407 + 100000 * rank + 1000 * epoch))
        loss_sum = 0.0
        samples = 0
        for (index, (inputs, labels, local_targets, names)) in enumerate(loader):
            if step == warm_steps:
                dist.barrier()
                module = net.module
                del net
                module.set_phase(False)
                net = wrap(module, local)
                net.train()
            inputs = move(inputs, device)
            labels = labels.to(device, non_blocking=True)
            inspect_bn = args.smoke and step in (0, warm_steps)
            bn_before = {k: int(v) for (k, v) in net.module.state_dict().items() if k.endswith('num_batches_tracked')} if inspect_bn else None
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = net(inputs)
                local_target = torch.stack([local_targets[i, r] for (i, r) in enumerate(net.module.last_regions)]).to(device, non_blocking=True)
                (loss, parts) = loss_for(outputs, labels, local_target, c, step)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            if inspect_bn:
                bn_forward = {k: int(v) for (k, v) in net.module.state_dict().items() if k.endswith('num_batches_tracked')}
                expected = 0 if step < warm_steps else 1
                assert bn_forward and all((bn_forward[k] - v == expected for (k, v) in bn_before.items())), 'BN forward count'
            loss.backward()
            if inspect_bn:
                bn_after = {k: int(v) for (k, v) in net.module.state_dict().items() if k.endswith('num_batches_tracked')}
                assert bn_after == bn_forward, 'Checkpoint recomputation changed BN'
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            if args.smoke:
                prefixes = ['new_attention.']
                if step >= warm_steps:
                    prefixes += ['global_net.semantic_encoder.', 'local_net.semantic_encoder.', 'global_net.decoder.', 'local_net.decoder.']
                gradients = {p: sum((float(v.grad.abs().sum()) for (k, v) in net.module.named_parameters() if k.startswith(p) and v.grad is not None)) for p in prefixes}
                if step >= warm_steps + 2:
                    assert all((v > 0 for v in gradients.values())), gradients
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            peak = torch.cuda.max_memory_reserved() / 2 ** 20
            if peak > 23000:
                raise RuntimeError('Declared memory limit exceeded: ' + str(peak))
            loss_sum += float(loss.detach()) * len(labels)
            samples += len(labels)
            row = {'epoch': epoch, 'optimizer_step': step, 'loss': float(loss.detach()), 'grad_norm': float(norm), 'head_only': net.module.head_only, 'peak_reserved_mib': peak, 'epoch_elapsed_seconds': time.perf_counter() - epoch_started, 'parts': {k: float(v) for (k, v) in parts.items()}}
            if rank == 0 and (step <= 3 or step % 25 == 0):
                print(__import__('json').dumps(row), flush=True)
            if args.smoke:
                checks.append({**row, 'gradients': gradients})
            del outputs, loss, inputs, labels, local_target, local_targets
            if args.smoke and step == 20:
                break
        if args.smoke:
            own = {'rank': rank, 'checks': checks, 'parameter_sha256': parameter_hash(net.module), 'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2 ** 20}
            gathered = [None] * 2
            dist.all_gather_object(gathered, own)
            assert len({r['parameter_sha256'] for r in gathered}) == 1
            if rank == 0:
                atomic(out / 'gpu_check.json', {'real_updates': 20, 'head_only_updates': 2, 'joint_updates': 18, 'ranks': gathered, 'physical_batch_per_gpu': 12, 'accumulation': 1, 'precision': 'bf16', 'real_bn_check_passed': True, 'ddp_phase_transition_passed': True, 'frozen_plan_sha256': digest(TASK / 'frozen_plan.json')})
            dist.barrier()
            dist.destroy_process_group()
            return
        assert step == 707 * epoch
        del loader
        train_seconds = time.perf_counter() - epoch_started
        reduced = torch.tensor([loss_sum, samples], device=device)
        dist.all_reduce(reduced)
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}
        rngs = [None] * 2
        dist.all_gather_object(rngs, rng)
        dist.barrier()
        validation_started = time.perf_counter()
        (validation, _) = evaluate_distributed(net.module, val_loader, device)
        validation_seconds = time.perf_counter() - validation_started
        if rank == 0:
            row = {'epoch': epoch, 'global_step': step, 'train_loss': float(reduced[0] / reduced[1]), 'epoch_seconds': time.perf_counter() - epoch_started, 'elapsed_seconds': elapsed_before + time.perf_counter() - started, 'train_seconds': train_seconds, 'validation_seconds': validation_seconds, 'head_only': net.module.head_only, **validation}
            history.append(row)
            improved = row['miou'] > best
            best = max(best, row['miou'])
            saved = {'epoch': epoch, 'global_step': step, 'config': c, 'model': net.module.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'best_miou': best, 'rng_by_rank': rngs, 'head_only': net.module.head_only, 'execution_manifest_sha256': digest(TASK / 'frozen_plan.json')}
            save(out / 'last.pt', saved)
            if improved:
                save(out / 'best.pt', saved)
            atomic(out / 'history.json', history)
            print(__import__('json').dumps({k: row[k] for k in ['epoch', 'global_step', 'miou', 'macc', 'pacc', 'bf1', 'epoch_seconds']}), flush=True)
        dist.barrier()
    dist.destroy_process_group()
if __name__ == '__main__':
    main()
