import torch
from torch import nn
from torch.nn import functional as F
from .alignment import unpad
from .geometry import canvas_box,canvas_content,pixel_box
def unpad(x,source_size):
    left,top,w,h=canvas_content(source_size,x.shape[-1])
    return x[...,top:top+h,left:left+w]

def align_local_logits(local_logits,source_size,index):
    a,b,c,d=pixel_box(source_size,index)
    local=unpad(local_logits,(c-a,d-b))
    x0,y0,x1,y1=canvas_box(source_size,index)
    return F.interpolate(local[None].float(),size=(y1-y0,x1-x0),mode='bilinear',align_corners=False)[0]

def writeback(pred,confidence,local_logits,source_size,index,margin=.05,guard=4):
    """Both selectors use the identical acceptance rule and one crop."""
    pred=pred.to(local_logits.device);confidence=confidence.to(local_logits.device)
    x0,y0,x1,y1=canvas_box(source_size,index)
    logits=align_local_logits(local_logits,source_size,index)
    conf,label=logits.softmax(0).max(0)
    accept=conf>confidence[y0:y1,x0:x1]+margin
    accept[:guard]=False;accept[-guard:]=False;accept[:,:guard]=False;accept[:,-guard:]=False
    result=pred.clone();region=result[y0:y1,x0:x1]
    region[accept]=label[accept]
    mask=torch.zeros_like(pred,dtype=torch.bool);mask[y0:y1,x0:x1]=accept
    return result,mask
def finish_metrics(metrics,include_confusion=False):
    result=metrics.compute();m=metrics.confusion.double();gt=m.sum(1);pred=m.sum(0)
    iou=m.diag()/(gt+pred-m.diag()).clamp_min(1)
    result.update(miou=float(iou.mean()),macc=float((m.diag()/gt.clamp_min(1)).mean()),
        pacc=float(m.diag().sum()/m.sum().clamp_min(1)),per_class_iou=iou.tolist(),
        per_class_acc=(m.diag()/gt.clamp_min(1)).tolist(),gt_pixels=gt.long().tolist(),predicted_pixels=pred.long().tolist(),evaluated_classes=37)
    if include_confusion:result['confusion_matrix']=m.long().tolist()
    return result

def evaluate37(model,loader,device,include_confusion=False):
    from .metrics import SegmentationMetrics
    from .local_data import LocalDataset,cached
    model.eval();metrics=SegmentationMetrics(37,255,2);corrected=damaged=accepted=0
    local=isinstance(loader.dataset,LocalDataset)
    for images,labels,names in loader:
        logits=model(images.to(device,non_blocking=True))[0]
        predictions=[]
        for i,name in enumerate(names):
            if local:
                z=cached(name);region=int(z['entropy_index']) if loader.dataset.selected is None else loader.dataset.selected[name]
                base=torch.from_numpy(z['pred']).long().to(device);conf=torch.from_numpy(z['confidence'].astype('float32')).to(device)
                changed,mask=writeback(base,conf,logits[i],tuple(z['source_size']),region)
                target=labels[i].to(device);valid=target!=255
                corrected+=int(((base!=target)&(changed==target)&valid).sum())
                damaged+=int(((base==target)&(changed!=target)&valid).sum());accepted+=int((mask&valid).sum())
                predictions.append(changed.cpu())
            else:predictions.append(logits[i].argmax(0).cpu())
        metrics.update(torch.stack(predictions),labels)
    result=finish_metrics(metrics,include_confusion)
    if local:result.update(corrected_pixels=corrected,damaged_pixels=damaged,accepted_pixels=accepted,selection='entropy' if loader.dataset.selected is None else 'learned_gain')
    return result
