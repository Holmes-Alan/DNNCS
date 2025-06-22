from einops import rearrange
import torch.nn.functional as F
import math
from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np

class temporal_sr(nn.Module):
    def __init__(self):
        super(temporal_sr, self).__init__()

    def forward(self, x, y):  
        b, c, h, w = x.shape  
        x = x.permute(0, 2, 3, 1) # [b,h,w,c]
        x = x.contiguous().view(b*h*w, c, 1)
        y = y.permute(0, 2, 3, 1) # [b,h,w,c]
        y = y.contiguous().view(b*h*w, c, 1)
        out = torch.cat((x, y), dim=-1)

        out = F.interpolate(out, scale_factor=1.5, mode='linear', align_corners=True)   
        
        out = out.contiguous().view(b, h, w, c, 3)
        out = out.permute(0, 3, 4, 1, 2) # [b,c,t,h,w]
        x, y, z = out.chunk(3, dim=2)
        x = x.squeeze(2)
        y = y.squeeze(2)
        z = z.squeeze(2)

        out = torch.cat((x, y, z), dim=0)

        return out
    
def ForwardMacPI(a, b, frames, vertical=True):
    if vertical:
        # x = rearrange(x, 'b n c h w -> b c (h n) w')
        x = torch.cat((a, b), dim=3)
        x = x.squeeze(1)
        b, c, hn, w = x.shape
        h = hn // frames
        tempU = []
        for i in range(h):
            tempU.append(x[:, :, i::h, :])
        out = torch.cat(tempU, dim=2)
        return out
    else:
        # x = rearrange(x, 'b n c h w -> b c (h n) w')
        x = torch.cat((a, b), dim=4)
        x = x.squeeze(1)
        b, c, h, wn = x.shape
        w = wn // frames
        tempV = []
        for i in range(w):
            tempV.append(x[:, :, :, i::w])
        out = torch.cat(tempV, dim=3)
        return out

def BackwardMacPI(x, frames, vertical=True):
    out = []
    if vertical:
        b, c, hn, w = x.shape
        for i in range(frames):
            out.append(x[:, :, i::frames, :])
        out = torch.cat(out, 2)
        a, b = out.chunk(2, dim=2)
        return a, b
    else:
        b, c, h, wn = x.shape
        for i in range(frames):
            out.append(x[:, :, :, i::frames])
        out = torch.cat(out, 3)
        a, b = out.chunk(2, dim=3)
        return a, b


def ForwardMacPI_coord(a, b, coord, frames, vertical=True):
    if vertical:
        # x = rearrange(x, 'b n c h w -> b c (h n) w')
        x = torch.cat((a, b), dim=3)
        y = coord.repeat(1, 1, 1, 2, 1)
        x = x.squeeze(1)
        y = y.squeeze(1)
        b, c, hn, w = x.shape
        h = hn // frames
        tempU = []
        tempV = []
        for i in range(h):
            tempU.append(x[:, :, i::h, :])
            tempV.append(y[:, :, i::h, :])
        out = torch.cat(tempU, dim=2)
        coord = torch.cat(tempV, dim=2)
        return out, coord
    else:
        # x = rearrange(x, 'b n c h w -> b c (h n) w')
        x = torch.cat((a, b), dim=4)
        y = coord.repeat(1, 1, 1, 1, 2)
        x = x.squeeze(1)
        y = y.squeeze(1)
        b, c, h, wn = x.shape
        w = wn // frames
        tempU = []
        tempV = []
        for i in range(w):
            tempU.append(x[:, :, :, i::w])
            tempV.append(y[:, :, :, i::w])
        out = torch.cat(tempU, dim=3)
        coord = torch.cat(tempV, dim=3)
        return out, coord

    
# a = torch.rand(1, 1, 1, 3, 3)
# b = torch.rand(1, 1, 1, 3, 3)

# c = torch.cat((a, b), dim=1)

# d = ForwardMacPI(a, b, 2, vertical=True)
# a1, b1 = BackwardMacPI(d, 2, vertical=True)

# print('a==========', a)
# print('b==========', b)
# print('d==========', d.shape)
# print('a1==========', a1)
# print('b1==========', b1)




## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, 4, 1, padding=0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(4, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
            if i == 0: modules_body.append(nn.LeakyReLU(0.1, inplace=True))
        modules_body.append(CALayer(n_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(n_feat) \
            for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    

class RCAN_encoder(nn.Module):
    def __init__(self, dim):
        super(RCAN_encoder, self).__init__()
        self.init_feature = nn.Conv2d(3, dim, 3, 1, 1)
        self.RG1 = ResidualGroup(n_feat=dim, n_resblocks=20)
        self.RG2 = ResidualGroup(n_feat=dim, n_resblocks=20)
        self.RG3 = ResidualGroup(n_feat=dim, n_resblocks=20)
        self.RG4 = ResidualGroup(n_feat=dim, n_resblocks=20)
        self.RG5 = ResidualGroup(n_feat=dim, n_resblocks=20)
        # self.RG6 = ResidualGroup(n_feat=dim, n_resblocks=20)
        # self.RG7 = ResidualGroup(n_feat=dim, n_resblocks=20)
        # self.RG8 = ResidualGroup(n_feat=dim, n_resblocks=20)
        # self.RG9 = ResidualGroup(n_feat=dim, n_resblocks=20)
        # self.RG10 = ResidualGroup(n_feat=dim, n_resblocks=20)
        

    def forward(self, lr):
        buffer_00 = self.init_feature(lr)
        buffer_01 = self.RG1(buffer_00)
        buffer_02 = self.RG2(buffer_01)
        buffer_03 = self.RG3(buffer_02)
        buffer_04 = self.RG4(buffer_03)
        buffer_05 = self.RG5(buffer_04)
        # buffer_06 = self.RG6(buffer_05)
        # buffer_07 = self.RG7(buffer_06)
        # buffer_08 = self.RG8(buffer_07)
        # buffer_09 = self.RG9(buffer_08)
        # buffer_10 = self.RG10(buffer_09)

        return buffer_00, buffer_05
    

class RCAN_encoder_v2(nn.Module):
    def __init__(self, dim):
        super(RCAN_encoder_v2, self).__init__()
        self.init_feature = nn.Conv2d(3, dim, 3, 1, 1)
        self.RG1 = ResidualGroup(n_feat=dim, n_resblocks=20)
        self.RG2 = ResidualGroup(n_feat=dim, n_resblocks=20)
        self.RG3 = ResidualGroup(n_feat=dim, n_resblocks=20)

    def forward(self, lr):
        buffer_00 = self.init_feature(lr)
        buffer_01 = self.RG1(buffer_00)
        buffer_02 = self.RG2(buffer_01)
        buffer_03 = self.RG3(buffer_02)

        return buffer_00, buffer_03
    

class BasicTrans(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8):
        super(BasicTrans, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, bias=False)
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim, bias=False),
            nn.ReLU(True),
            nn.Linear(spa_dim, spa_dim, bias=False),
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)

    def forward(self, buffer):
        [b, c, h, w] = buffer.size() # b x 64 x 2h x w

        epi_token = rearrange(buffer, 'b c h w -> w (b h) c')
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, 'w (b h) c -> b c h w', b=b, c=c)

        return buffer
    

class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine activation function with w0 scaling support.
        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
        
class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim

        self.heads = heads
        self.scale = dim ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x, coord):
        norm_x = self.norm(x)

        q, k, v = self.to_qkv(norm_x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        q = q + coord
        k = k + coord
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class BasicTrans_pos(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8):
        super(BasicTrans_pos, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.linear_poe = nn.Sequential(
            nn.Linear(spa_dim, spa_dim),
            Sine(w0=1),
            nn.Linear(spa_dim, spa_dim)
        )
        self.norm = nn.LayerNorm(spa_dim)
        self.attn = nn.MultiheadAttention(spa_dim, num_heads, bias=False)
        self.attn.out_proj.bias = None
        self.attn.in_proj_bias = None
        # self.attention = SelfAttention(spa_dim, num_heads)
        # self.attn_out = nn.Linear(2*spa_dim, spa_dim)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim, bias=False),
            nn.ReLU(True),
            nn.Linear(spa_dim, spa_dim, bias=False),
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)

    def forward(self, buffer, coord):
        # coord: b x 2 x 2h x w
        [b, c, h, w] = buffer.size() # b x 64 x 2h x w
        epi_token = rearrange(buffer, 'b c h w -> w (b h) c')
        epi_token = self.linear_in(epi_token)
        coord = rearrange(coord, 'b c h w -> w (b h) c')
        coord = self.linear_poe(coord)
        # first try
        epi_token_norm = self.norm(epi_token + coord)
        # a, attn_map = self.attn(query=epi_token_norm,
        #                            key=epi_token_norm,
        #                            value=epi_token,
        #                            need_weights=True)
        # epi_token += a
        epi_token = self.attn(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, 'w (b h) c -> b c h w', b=b, c=c)

        # second try
        # epi_token = epi_token.permute(1, 0, 2)
        # coord = coord.permute(1, 0, 2)
        # epi_token = self.attention(epi_token, coord) + epi_token

        # epi_token = self.feed_forward(epi_token) + epi_token
        # epi_token = self.linear_out(epi_token)
        # buffer = rearrange(epi_token, '(b h) w c -> b c h w', b=b, c=c)

        return buffer


class BasicTrans_pos_vis(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8):
        super(BasicTrans_pos_vis, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.linear_poe = nn.Sequential(
            nn.Linear(3, spa_dim),
            Sine(w0=1),
            nn.Linear(spa_dim, spa_dim)
        )
        self.norm = nn.LayerNorm(spa_dim)
        self.attn = nn.MultiheadAttention(spa_dim, num_heads, bias=False)
        self.attn.out_proj.bias = None
        self.attn.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim, bias=False),
            nn.ReLU(True),
            nn.Linear(spa_dim, spa_dim, bias=False),
        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)

    def forward(self, buffer, coord):
        # coord: b x 2 x 2h x w
        [b, c, h, w] = buffer.size() # b x 64 x 2h x w
        epi_token = rearrange(buffer, 'b c h w -> w (b h) c')
        epi_token = self.linear_in(epi_token)
        coord = rearrange(coord, 'b c h w -> w (b h) c')
        coord = self.linear_poe(coord)
        # first try
        epi_token_norm = self.norm(epi_token + coord)
        epi_token = self.attn(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, 'w (b h) c -> b c h w', b=b, c=c)

        return buffer
    

class RCAN(nn.Module):
    def __init__(self, dim):
        super(RCAN, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        
        # Shallow feature extraction net
        self.feat_head = RCAN_encoder(dim)
        self.bp = BasicTrans(dim, dim)
        # self.bp_v = BasicTrans(dim, dim*2)

        self.UPNet = nn.Sequential(
            nn.Conv2d(2*dim, dim*3, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(dim*3, dim*3, 3, 1, 1),
            nn.ReLU()
        )

        self.output = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, 3, 3, 1, 1),
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=0)
        self.mean = self.mean.type_as(x)
        x = x - self.mean

        res, feat = self.feat_head(x)
        feat_x, feat_y = feat.chunk(2, dim=0)
        feat_x = feat_x.unsqueeze(1)
        feat_y = feat_y.unsqueeze(1)
        # horizontal
        z = ForwardMacPI(feat_x, feat_y, 2, vertical=False) # b x 64 x 2h x w
        z = z + self.bp(z)
        feat_x, feat_y = BackwardMacPI(z, 2, vertical=False)
        # vertical
        feat_x = feat_x.unsqueeze(1)
        feat_y = feat_y.unsqueeze(1)
        z = ForwardMacPI(feat_x, feat_y, 2, vertical=True) # b x 64 x h x 2w
        z = z + self.bp(z)
        feat_x, feat_y = BackwardMacPI(z, 2, vertical=True)
        # output
        z = torch.cat((feat_x, feat_y), dim=1)
        res_x, res_y = res.chunk(2, dim=0)
        res = torch.cat((res_x, res_y), dim=1)
        z = z + res
        z = self.UPNet(z)
        feat_x, feat_y, feat_z = z.chunk(3, dim=1)
        z = torch.cat((feat_x, feat_y, feat_z), dim=0)
        z = self.output(z)
        z = z + self.mean
        x1, x2, x3 = z.chunk(3, dim=0)

        return x1, x2, x3
    

class DNNCS(nn.Module):
    def __init__(self, dim):
        super(DNNCS, self).__init__()

        self.tempSR = temporal_sr()
        self.dim = dim
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        # Shallow feature extraction net
        self.feat_head = RCAN_encoder(dim)
        self.bp = BasicTrans_pos(dim, dim)

        self.pos_embed = nn.Parameter(torch.randn(3, 64))
        # self.pos_extend = nn.Conv2d(3, dim, 1, 1, 0)
        # self.pos_encode = nn.Conv2d(3*dim, dim, 1, 1, 0)
        self.UPNet = nn.Sequential(
            nn.Conv2d(2*dim, dim*3, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(dim*3, dim*3, 3, 1, 1),
            nn.ReLU()
        )

        self.output = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, 3, 3, 1, 1),
        )

    def forward(self, x1, x2, coord=None, mask=None):
        bs, h, w = x1.shape[0], x1.shape[2], x1.shape[3]
        x = torch.cat((x1, x2), dim=0)
        self.mean = self.mean.type_as(x)
        x = x - self.mean

        res, feat = self.feat_head(x)
        feat_x, feat_y = feat.chunk(2, dim=0)
        feat_x = feat_x.unsqueeze(1)
        feat_y = feat_y.unsqueeze(1)
        coord = torch.cat((coord, mask[:, 0:1, :, :]), dim=1)
        # coord = self.pos_extend(coord)
        coord = coord.reshape(bs, 3, -1)
        coord = torch.matmul(coord.transpose(-1, -2), self.pos_embed)
        coord = coord.transpose(-1, -2)
        coord = coord.reshape(bs, self.dim, h, w)
        coord_ = coord.unsqueeze(1)
        # horizontal
        z, coord_z = ForwardMacPI_coord(feat_x, feat_y, coord_, 2, vertical=False) # b x 64 x 2h x w
        z = z + self.bp(z, coord_z)
        feat_x, feat_y = BackwardMacPI(z, 2, vertical=False)
        # vertical
        feat_x = feat_x.unsqueeze(1)
        feat_y = feat_y.unsqueeze(1)
        z, coord_z = ForwardMacPI_coord(feat_x, feat_y, coord_, 2, vertical=True) # b x 64 x h x 2w
        z = z + self.bp(z, coord_z)
        feat_x, feat_y = BackwardMacPI(z, 2, vertical=True)

        feat_x = feat_x.unsqueeze(1).permute(0, 1, 3, 2, 4)
        feat_y = feat_y.unsqueeze(1).permute(0, 1, 3, 2, 4)
        coord_ = coord_.permute(0, 1, 3, 2, 4)
        z, coord_z = ForwardMacPI_coord(feat_x, feat_y, coord_, 2, vertical=False) # b x h x (2x64) x w
        z = z + self.bp(z, coord_z)

        feat_x, feat_y = BackwardMacPI(z, 2, vertical=False)
        feat_x = feat_x.permute(0, 2, 1, 3)
        feat_y = feat_y.permute(0, 2, 1, 3)
        # output
        z = torch.cat((feat_x, feat_y), dim=1)
        # frequency encoding
        z_coord = feat_x * coord + feat_y * coord
        z_coord = torch.cat((torch.cos(np.pi * z_coord), torch.sin(np.pi * z_coord)), dim=1)
        z = self.UPNet(z + z * z_coord)
        res_x, res_y = res.chunk(2, dim=0)
        temSR = self.tempSR(res_x, res_y)
        res_x, res_y, res_z = temSR.chunk(3, dim=0)
        res = torch.cat((res_x, res_y, res_z), dim=1)
        z = z + res
        feat_x, feat_y, feat_z = z.chunk(3, dim=1)
        z = torch.cat((feat_x, feat_y, feat_z), dim=0)
        z = self.output(z)
        z = z + self.mean
        x1, x2, x3 = z.chunk(3, dim=0)

        return x1, x2, x3
    



class ConvReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)

        return out
    
class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.bic_model = temporal_sr()
        # Input layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(True),
        )

        # Features trunk blocks
        trunk = []
        for _ in range(18):
            trunk.append(ConvReLU(64))
        self.trunk = nn.Sequential(*trunk)

        # Output layer
        self.conv2 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1), bias=False)

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x, y):
        out = self.bic_model(x, y)
        return self._forward_impl(out)

    # Support torch.script function
    def _forward_impl(self, x):
        identity = x

        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)

        out = torch.add(out, identity)
        x1, x2, x3 = out.chunk(3, dim=0)

        return x1, x2, x3

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, math.sqrt(2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))



def patchify_tensor(features, patch_size, overlap=10):
    batch_size, channels, height, width = features.size()

    effective_patch_size = patch_size - overlap
    n_patches_height = (height // effective_patch_size)
    n_patches_width = (width // effective_patch_size)

    if n_patches_height * effective_patch_size < height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < width:
        n_patches_width += 1

    patches = []
    for b in range(batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, height - patch_size)
                patch_start_width = min(w * effective_patch_size, width - patch_size)
                patches.append(features[b:b+1, :,
                               patch_start_height: patch_start_height + patch_size,
                               patch_start_width: patch_start_width + patch_size])
    return torch.cat(patches, 0)



def recompose_tensor(patches1, patches2, patches3, full_height, full_width, overlap=10):

    batch_size, channels, patch_size, _ = patches1.size()
    effective_patch_size = patch_size - overlap
    n_patches_height = (full_height // effective_patch_size)
    n_patches_width = (full_width // effective_patch_size)

    if n_patches_height * effective_patch_size < full_height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < full_width:
        n_patches_width += 1

    n_patches = n_patches_height * n_patches_width
    if batch_size % n_patches != 0:
        print("Error: The number of patches provided to the recompose function does not match the number of patches in each image.")
    final_batch_size = batch_size // n_patches

    blending_in = torch.linspace(0.1, 1.0, overlap)
    blending_out = torch.linspace(1.0, 0.1, overlap)
    middle_part = torch.ones(patch_size - 2 * overlap)
    blending_profile = torch.cat([blending_in, middle_part, blending_out], 0)

    horizontal_blending = blending_profile[None].repeat(patch_size, 1)
    vertical_blending = blending_profile[:, None].repeat(1, patch_size)
    blending_patch = horizontal_blending * vertical_blending

    blending_image = torch.zeros(1, channels, full_height, full_width)
    for h in range(n_patches_height):
        for w in range(n_patches_width):
            patch_start_height = min(h * effective_patch_size, full_height - patch_size)
            patch_start_width = min(w * effective_patch_size, full_width - patch_size)
            blending_image[0, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += blending_patch[None]

    recomposed_tensor1 = torch.zeros(final_batch_size, channels, full_height, full_width)
    recomposed_tensor2 = torch.zeros(final_batch_size, channels, full_height, full_width)
    recomposed_tensor3 = torch.zeros(final_batch_size, channels, full_height, full_width)
    if patches1.is_cuda:
        blending_patch = blending_patch.cuda()
        blending_image = blending_image.cuda()
        recomposed_tensor1 = recomposed_tensor1.cuda()
        recomposed_tensor2 = recomposed_tensor2.cuda()
        recomposed_tensor3 = recomposed_tensor3.cuda()
    patch_index = 0
    for b in range(final_batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, full_height - patch_size)
                patch_start_width = min(w * effective_patch_size, full_width - patch_size)
                recomposed_tensor1[b, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += patches1[patch_index] * blending_patch
                recomposed_tensor2[b, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += patches2[patch_index] * blending_patch
                recomposed_tensor3[b, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += patches3[patch_index] * blending_patch
                patch_index += 1
    recomposed_tensor1 /= blending_image
    recomposed_tensor2 /= blending_image
    recomposed_tensor3 /= blending_image

    return recomposed_tensor1, recomposed_tensor2, recomposed_tensor3