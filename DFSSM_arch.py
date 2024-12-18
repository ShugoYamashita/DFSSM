import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange
from typing import Callable
from functools import partial

from basic_arch import SS2D, ChannelAttention, OverlapPatchEmbed, Downsample, Upsample

NEG_INF = -1000000


class MGCB(nn.Module):
    def __init__(self, dim, bias=False, **kwargs):
        super().__init__()

        self.hidden_features = int(dim*2)
        self.project_in = nn.Conv2d(dim, self.hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(int(self.hidden_features*3//2), int(self.hidden_features*3//2), kernel_size=3, stride=1, padding=1, groups=int(self.hidden_features*3//2), bias=bias)
        self.dwconv5x5 = nn.Conv2d(int(self.hidden_features//2), int(self.hidden_features//2), kernel_size=5, stride=1, padding=2, groups=int(self.hidden_features//2), bias=bias)
        self.act1 = nn.GELU()
        self.project_out = nn.Conv2d(self.hidden_features, dim, kernel_size=1, bias=bias)
        self.ca = ChannelAttention(dim, squeeze_factor=30)

    def forward(self, x):
        x1, x2 = self.project_in(x).split([int(self.hidden_features*3//2), int(self.hidden_features//2)], dim=1)
        x1_1, x1_2 = self.dwconv3x3(x1).split([int(self.hidden_features), int(self.hidden_features//2)], dim=1)
        x2 = self.dwconv5x5(x2)
        x = self.act1(x1_1) * torch.cat([x1_2, x2], dim=1)
        x = self.project_out(x)
        x = self.ca(x)
        return x


class SSB(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = MGCB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class FFTM(nn.Module):
    def __init__(self, embed_dim, fft_norm='ortho', squeeze_factor=2, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(embed_dim, embed_dim // squeeze_factor, 1, padding=0)
        self.act1 = nn.SiLU()
        self.conv_layer = nn.Conv2d(embed_dim * 2 // squeeze_factor, embed_dim * 2 // squeeze_factor, 1, 1, 0)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(embed_dim // squeeze_factor, embed_dim, 1, padding=0)
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        x = self.act1(self.conv1(x))
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act2(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        output = self.conv2(output)

        return output


class FSSB(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            fft_squeeze_factor = 2,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.fft_block = FFTM(hidden_dim, squeeze_factor=fft_squeeze_factor)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = MGCB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x)) + self.fft_block(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class OneStage(nn.Module):
    def __init__(self, hidden_dim, drop_path, norm_layer, attn_drop_rate, expand, d_state, num_blocks, fft_squeeze_factor=2, num_fft_block=1, **kwargs):
        super().__init__()
        self.vss_blocks = nn.ModuleList([
            SSB(
                hidden_dim=hidden_dim,
                drop_path=drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                expand=expand,
                d_state=d_state,
            )
            for i in range(num_blocks - num_fft_block)])
        self.fftvss_block = nn.ModuleList([
            FSSB(
                hidden_dim=hidden_dim,
                drop_path=drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                expand=expand,
                d_state=d_state,
                fft_squeeze_factor=fft_squeeze_factor,
            )
            for i in range(num_fft_block)])

    def forward(self, x, x_size):
        for layer in self.vss_blocks:
            x = layer(x, x_size)
        for layer in self.fftvss_block:
            x = layer(x, x_size)
        return x


class DFSSM(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 4, 4, 4, 4, 4, 4],
                 num_refinement_blocks=4,
                 mlp_ratio=1.5,
                 drop_path_rate=0.,
                 num_fft_blocks=[1, 1, 1, 1, 1, 1, 1, 1],
                 fft_squeeze_factor=2,
                 bias=False,
                 **kwargs,
                 ):

        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        base_d_state = 4

        self.encoder_level1 = nn.ModuleList([
            OneStage(
                hidden_dim=dim,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=base_d_state * 2 ** 2,
                num_fft_block=num_fft_blocks[0],
                fft_squeeze_factor=fft_squeeze_factor,
                num_blocks=num_blocks[0],
            )])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([
            OneStage(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
                num_fft_block=num_fft_blocks[1],
                fft_squeeze_factor=fft_squeeze_factor,
                num_blocks=num_blocks[1],
            )])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([
            OneStage(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
                num_fft_block=num_fft_blocks[2],
                fft_squeeze_factor=fft_squeeze_factor,
                num_blocks=num_blocks[2],
            )])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.ModuleList([
            OneStage(
                hidden_dim=int(dim * 2 ** 3),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
                num_fft_block=num_fft_blocks[3],
                fft_squeeze_factor=fft_squeeze_factor,
                num_blocks=num_blocks[3],
            )])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            OneStage(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
                num_fft_block=num_fft_blocks[4],
                fft_squeeze_factor=fft_squeeze_factor,
                num_blocks=num_blocks[4],
            )])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            OneStage(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
                num_fft_block=num_fft_blocks[5],
                fft_squeeze_factor=fft_squeeze_factor,
                num_blocks=num_blocks[5],
            )])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([
            OneStage(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
                num_fft_block=num_fft_blocks[6],
                fft_squeeze_factor=fft_squeeze_factor,
                num_blocks=num_blocks[6],
            )])

        self.refinement = nn.ModuleList([
            OneStage(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
                num_fft_block=num_fft_blocks[7],
                fft_squeeze_factor=fft_squeeze_factor,
                num_blocks=num_refinement_blocks,
            )])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        _, _, H, W = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)  # b,hw,c
        out_enc_level1 = inp_enc_level1
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [H, W])

        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)  # b, hw//4, 2c
        out_enc_level2 = inp_enc_level2
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H // 2, W // 2])

        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2)  # b, hw//16, 4c
        out_enc_level3 = inp_enc_level3
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, [H // 4, W // 4])

        inp_enc_level4 = self.down3_4(out_enc_level3, H // 4, W // 4)  # b, hw//64, 8c
        latent = inp_enc_level4
        for layer in self.latent:
            latent = layer(latent, [H // 8, W // 8])

        inp_dec_level3 = self.up4_3(latent, H // 8, W // 8)  # b, hw//16, 4c
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c
        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, [H // 4, W // 4])

        inp_dec_level2 = self.up3_2(out_dec_level3, H // 4, W // 4)  # b, hw//4, 2c
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c
        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, [H // 2, W // 2])

        inp_dec_level1 = self.up2_1(out_dec_level2, H // 2, W // 2)  # b, hw, c
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, [H, W])

        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, [H, W])

        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


if __name__ == '__main__':
    height = 128
    width = 128
    model = DFSSM(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 4, 4, 4, 4, 4, 4],
        num_refinement_blocks=4,
        mlp_ratio=1.5,
        num_fft_blocks=[3, 3, 3, 3, 3, 3, 3, 3],
        fft_squeeze_factor=2,
        bias=False,
    ).cuda()
