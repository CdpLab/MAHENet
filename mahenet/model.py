from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .segmenter import Segmenter
from .memory import preserve_batchnorm
from .ops import align_global,baseline_logits,aligned_local,selected_regions,local_content,fp32_frequency,enable_block_checkpointing
from .attention import GlobalGuide

class MAHENet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.head_only = False
        self.config = config
        self.global_size = config['experiment']['global_size']
        pretrained = config.get('model', {}).get('pretrained_checkpoint')
        self.global_net = Segmenter(pretrained)
        self.local_net = Segmenter(pretrained)
        for (net, key) in ((self.global_net, 'global_checkpoint'), (self.local_net, 'expert_checkpoint')):
            path = config.get('architecture', {}).get(key)
            if path:
                saved = torch.load(path, map_location='cpu')
                net.load_state_dict(saved.get('model', saved), strict=True)
        self.checkpointed_vss_blocks = sum((enable_block_checkpointing(net.semantic_encoder) for net in (self.global_net, self.local_net)))
        assert self.checkpointed_vss_blocks == 28
        self.new_attention = nn.ModuleList([GlobalGuide(), GlobalGuide()])

    def set_phase(self, head_only):
        self.head_only = head_only
        self.global_net.requires_grad_(not head_only)
        self.local_net.requires_grad_(not head_only)
        self.new_attention.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.head_only:
            self.global_net.eval()
            self.local_net.eval()
        return self

    def call(self, module, function, *args):
        if self.training and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False, context_fn=lambda : (nullcontext(), preserve_batchnorm(module)))
        return function(*args)

    def feature_path(self, net, image, global_features=None, sizes=None, regions=None):
        (s2, s3, s4, s5) = net.semantic_encoder(image)
        lateral = net.fusion.lateral[0](s2)
        p5 = net.fusion.deep_aspp(s5)
        p4 = net.fusion.refine[2](F.interpolate(p5, size=s4.shape[-2:], mode='bilinear', align_corners=False) + net.fusion.lateral[2](s4))
        high = net.fusion.lateral[1](s3)
        p3 = fp32_frequency(net.frequency_fusion_p3, high, p4, net.training)
        p3 = net.fusion.refine[1](p3)
        (p3, coarse) = net.material_context(p3)
        if global_features is not None and True:
            p3 = self.new_attention[0](p3, global_features[1], sizes, regions)
        p2 = fp32_frequency(net.frequency_fusion, lateral, p3, net.training)
        p2 = net.fusion.refine[0](p2)
        if global_features is not None and True:
            p2 = self.new_attention[1](p2, global_features[0], sizes, regions)
        return (p2, p3, coarse)

    def forward(self, inputs):
        sizes = [tuple(map(int, s)) for s in inputs['size'].tolist()]
        (g2, g3, gcoarse) = self.call(self.global_net, lambda x: self.feature_path(self.global_net, x), inputs['global'])

        def decode_global(feature):
            (logits, edge) = self.global_net.decoder(feature, (self.global_size, self.global_size))
            return (align_global(logits, sizes), edge)
        (global_logits, global_edge) = self.call(self.global_net.decoder, decode_global, g2)
        regions = selected_regions(global_logits, sizes)
        local = torch.stack([inputs['local'][i, r] for (i, r) in enumerate(regions)]).to(inputs['global'].device, non_blocking=True)
        (l2, l3, lcoarse) = self.call(self.local_net, lambda x: self.feature_path(self.local_net, x, (g2, g3), sizes, regions), local)
        (local_logits, local_edge) = self.call(self.local_net.decoder, lambda x: self.local_net.decoder(x, (512, 512)), l2)
        (base, accepted) = baseline_logits(global_logits, local_logits, sizes, regions)
        final = base
        self.last_regions = regions
        (aligned, support) = aligned_local(local_logits.detach(), sizes, regions)
        return {'final': final, 'global': (global_logits, global_edge, gcoarse), 'local': (local_logits, local_edge, lcoarse), 'global_pred': global_logits.detach().argmax(1), 'local_pred': aligned.argmax(1), 'local_support': support, 'base_pred': base.detach().argmax(1), 'accepted': accepted.detach()}
