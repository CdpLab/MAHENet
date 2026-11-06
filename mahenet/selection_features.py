import torch
from .geometry import canvas_box
@torch.no_grad()
def region_features(logits,image,source_size):
    prob=logits.float().softmax(0);confidence=prob.max(0).values
    entropy=-(prob*prob.clamp_min(1e-8).log()).sum(0)
    top2=prob.topk(2,dim=0).values;margin=top2[0]-top2[1]
    features=[]
    for i in range(4):
        x0,y0,x1,y1=canvas_box(source_size,i);p=prob[:,y0:y1,x0:x1]
        rgb=image[:,y0:y1,x0:x1]
        scalars=torch.stack([entropy[y0:y1,x0:x1].mean(),confidence[y0:y1,x0:x1].mean(),
            confidence[y0:y1,x0:x1].std(unbiased=False),margin[y0:y1,x0:x1].mean()])
        feature=torch.cat([p.mean((1,2)),scalars,rgb.mean((1,2)),rgb.flatten(1).std(1,unbiased=False),
            logits.new_tensor([x0/512,y0/512,x1/512,y1/512])])
        features.append(feature)
    return torch.stack(features)
