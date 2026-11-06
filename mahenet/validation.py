import time,json
import torch
import torch.distributed as dist
from .metrics import SegmentationMetrics
def shard_indices(length,rank,world_size):
    if length<0 or world_size<1 or not 0<=rank<world_size:raise ValueError('Invalid shard')
    return list(range(length*rank//world_size,length*(rank+1)//world_size))

def pack_metrics(metrics,counts):
    return torch.cat((metrics.confusion.reshape(-1).cpu().long(),metrics.gt_pixels.cpu().long(),
        metrics.predicted_pixels.cpu().long(),torch.tensor([
            metrics.boundary_pred_matched,metrics.boundary_gt_matched,
            metrics.boundary_pred,metrics.boundary_gt,*counts],dtype=torch.long)))

def restore_metrics(metrics,flat):
    n=metrics.num_classes;flat=flat.cpu().long();offset=n*n
    metrics.confusion=flat[:offset].reshape(n,n).clone()
    metrics.gt_pixels=flat[offset:offset+n].clone();offset+=n
    metrics.predicted_pixels=flat[offset:offset+n].clone();offset+=n
    (metrics.boundary_pred_matched,metrics.boundary_gt_matched,
     metrics.boundary_pred,metrics.boundary_gt)=map(int,flat[offset:offset+4]);offset+=4
    counts=flat[offset:].tolist();assert len(counts)==6
    return counts

def reduce_metrics(metrics,counts,device):
    # Communication tensors are ordinary tensors even when caller uses inference_mode.
    with torch.inference_mode(False),torch.no_grad():
        flat=pack_metrics(metrics,counts).to(device)
        if dist.is_initialized():dist.all_reduce(flat,op=dist.ReduceOp.SUM)
        return restore_metrics(metrics,flat.cpu())

@torch.inference_mode()
def evaluate_segment(model,loader,device,progress=False):
    from .data import move
    model.eval();metrics=SegmentationMetrics(37,255,2)
    counts=[0]*6
    for inputs,labels,local_targets,names in loader:
        outputs=model(move(inputs,device));prediction=outputs['final'].argmax(1).cpu()
        reference=outputs['global_pred'].cpu();base=outputs['base_pred'].cpu();valid=labels!=255
        counts[0]+=int(((reference!=labels)&(prediction==labels)&valid).sum())
        counts[1]+=int(((reference==labels)&(prediction!=labels)&valid).sum())
        counts[2]+=int((outputs['accepted'].cpu()&valid).sum())
        counts[3]+=int(((base!=labels)&(prediction==labels)&valid).sum())
        counts[4]+=int(((base==labels)&(prediction!=labels)&valid).sum())
        counts[5]+=len(labels)
        metrics.update(prediction,labels)
        if progress and (counts[5]%256==0 or counts[5]==len(loader.dataset)):
            print(json.dumps({'validation_rank':dist.get_rank() if dist.is_initialized() else 0,
                'validation_images_local':counts[5],'validation_total_local':len(loader.dataset)}),flush=True)
    return metrics,counts

def evaluate_distributed(model,loader,device,expected_images=4044,expected_valid_pixels=601564010,progress=True):
    from .branch import finish_metrics
    model.eval()
    if dist.is_initialized():
        with torch.no_grad():
            for buffer in model.buffers():
                if buffer.numel():dist.broadcast(buffer,src=0)
    metrics,counts=evaluate_segment(model,loader,device,progress)
    counts=reduce_metrics(metrics,counts,device)
    assert counts[5]==expected_images,(counts[5],expected_images)
    if expected_valid_pixels is not None:assert int(metrics.confusion.sum())==expected_valid_pixels
    value=finish_metrics(metrics,True)
    value.update(corrected_pixels=counts[0],damaged_pixels=counts[1],accepted_pixels=counts[2],
        attention_corrected_vs_current_writeback=counts[3],attention_damaged_vs_current_writeback=counts[4],
        reference='same_model_global_branch',validation_precision='fp32_tf32',
        validation_images=counts[5],validation_workers=dist.get_world_size() if dist.is_initialized() else 1,
        boundary_raw_counts={k:int(getattr(metrics,k)) for k in (
            'boundary_pred_matched','boundary_gt_matched','boundary_pred','boundary_gt')})
    return value,pack_metrics(metrics,counts)
