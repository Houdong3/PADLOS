import torch
import torch.nn as nn
import torch.nn.functional as F
from .dpt import DepthAnythingV2
from .util.blocks import FeatureFusionBlock, _make_scratch
#深度信息和p
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFuseBlock(nn.Module):
    """Fuse RGB feature and Depth feature by concatenation + conv + BN + ReLU."""
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(True))
        self.fuse = nn.Sequential(*layers)

    def forward(self, rgb_feat, depth_feat):
        # align spatial size
        if rgb_feat.shape[-2:] != depth_feat.shape[-2:]:
            depth_feat = F.interpolate(depth_feat, size=rgb_feat.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([rgb_feat, depth_feat], dim=1)
        return self.fuse(x)


class DPTHead(nn.Module):
    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()

        # Project token features to CNN features
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        # Resize each scale
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        # build scratch layers
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.stem_transpose = None

        # refinenet
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # --- 🔹 Add RGB–Depth fusion blocks here ---
        self.fuse_blocks = nn.ModuleList([
            ConvFuseBlock(features * 2, features, use_bn=use_bn) for _ in range(4)
        ])

        # output head
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, rgb_features, dpt_features, patch_h, patch_w):
        # project tokens -> feature maps
        out = []
        for i, x in enumerate(rgb_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            #多尺度的实现
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        # apply refinenet
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        fuse_4 = self.fuse_blocks[3](path_4, dpt_features[3])
        path_3 = self.scratch.refinenet3(fuse_4, layer_3_rn, size=layer_2_rn.shape[2:])
        fuse_3 = self.fuse_blocks[2](path_3, dpt_features[2])
        path_2 = self.scratch.refinenet2(fuse_3, layer_2_rn, size=layer_1_rn.shape[2:])
        fuse_2 = self.fuse_blocks[1](path_2, dpt_features[1])
        path_1 = self.scratch.refinenet1(fuse_2, layer_1_rn)
        fuse_1 = self.fuse_blocks[0](path_1, dpt_features[0])
        

        # optional: refine again or select one to output
        # usually path_1 has the highest resolution -> output
        out = self.scratch.output_conv(fuse_1)

        return out


class PDDPT(nn.Module):
    def __init__(
        self, 
        encoder='vits', 
        num_classes=2,
        pretrained: str = None
    ):
        super(PDDPT, self).__init__()

        config = MODEL_CONFIGS[encoder]
        self.depth_anything = DepthAnythingV2(
            encoder=config['encoder'],
            features=config['features'],
            out_channels=config['out_channels']
        )

        if pretrained is not None:
            state_dict = torch.load(pretrained, map_location='cpu')
            self.depth_anything.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained}")

        # for p in self.depth_anything.parameters():
        #     p.requires_grad = False
        # self.depth_anything.eval()

        self.encoder_type = config['encoder']
        self.features=config['features']
        self.out_channels=config['out_channels']
        self.intermediate_idx = self.depth_anything.intermediate_layer_idx[self.encoder_type]
        vit_channels = self.depth_anything.pretrained.embed_dim
        self.patch_size = 14  
        
        self.head = DPTHead(num_classes, vit_channels , self.features, False, out_channels=self.out_channels)
        

    def forward(self, x):
        B, _, H, W = x.shape
        patch_h, patch_w =H // 14, W // 14

        dino_feats = self.depth_anything.pretrained.get_intermediate_layers(
            x,
            self.intermediate_idx,
            return_class_token=False
        )
        depth_map, depth_feats = self.depth_anything(x)


        seg_logits = self.head(dino_feats, depth_feats,patch_h, patch_w)
        seg_logits = F.interpolate(seg_logits, size=(H, W), mode='bilinear', align_corners=False)
        
        return seg_logits



