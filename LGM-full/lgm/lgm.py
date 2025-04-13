import os
import warnings
from functools import partial
from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fast_gauss import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from diffusers import ConfigMixin, ModelMixin
from torch import Tensor, nn


def look_at(campos):
    forward_vector = -campos / np.linalg.norm(campos, axis=-1)
    up_vector = np.array([0, 1, 0], dtype=np.float32)
    right_vector = np.cross(up_vector, forward_vector)
    up_vector = np.cross(forward_vector, right_vector)
    R = np.stack([right_vector, up_vector, forward_vector], axis=-1)
    return R


def orbit_camera(elevation, azimuth, radius=1):
    elevation = np.deg2rad(elevation)
    azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = -radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    campos = np.array([x, y, z])
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos)
    T[:3, 3] = campos
    return T


def get_rays(pose, h, w, fovy, opengl=True):
    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d)

    rays_o = rays_o.view(h, w, 3)
    rays_d = F.normalize(rays_d, dim=-1).view(h, w, 3)

    return rays_o, rays_d


class GaussianRenderer:
    def __init__(self, fovy, output_size):
        self.output_size = output_size

        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

        zfar = 2.5
        znear = 0.1
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (zfar + znear) / (zfar - znear)
        self.proj_matrix[3, 2] = -(zfar * znear) / (zfar - znear)
        self.proj_matrix[2, 3] = 1

    def render(
        self,
        gaussians,
        cam_view,
        cam_view_proj,
        cam_pos,
        bg_color=None,
        scale_modifier=1,
    ):
        device = gaussians.device
        B, V = cam_view.shape[:2]

        images = []
        alphas = []
        for b in range(B):

            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float()

            for v in range(V):
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                raster_settings = GaussianRasterizationSettings(
                    image_height=self.output_size,
                    image_width=self.output_size,
                    tanfovx=self.tan_half_fov,
                    tanfovy=self.tan_half_fov,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                rendered_image, _, _, rendered_alpha = rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(
                        means3D, dtype=torch.float32, device=device
                    ),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )

                rendered_image = rendered_image.clamp(0, 1)

                images.append(rendered_image)
                alphas.append(rendered_alpha)

        images = torch.stack(images, dim=0).view(
            B, V, 3, self.output_size, self.output_size
        )
        alphas = torch.stack(alphas, dim=0).view(
            B, V, 1, self.output_size, self.output_size
        )

        return {"image": images, "alpha": alphas}

    def save_ply(self, gaussians, path):
        assert gaussians.shape[0] == 1, "only support batch size 1"

        from plyfile import PlyData, PlyElement

        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float()

        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        opacity = opacity.clamp(1e-6, 1 - 1e-6)
        opacity = torch.log(opacity / (1 - opacity))
        scales = torch.log(scales + 1e-8)
        shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = (
            shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        )
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        h = ["x", "y", "z"]
        for i in range(f_dc.shape[1]):
            h.append("f_dc_{}".format(i))
        h.append("opacity")
        for i in range(scales.shape[1]):
            h.append("scale_{}".format(i))
        for i in range(rotations.shape[1]):
            h.append("rot_{}".format(i))

        dtype_full = [(attribute, "f4") for attribute in h]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")

        PlyData([el]).write(path)


class LGM(ModelMixin, ConfigMixin):
    def __init__(self):
        super().__init__()

        self.input_size = 256
        self.splat_size = 128
        self.output_size = 512
        self.radius = 1.5
        self.fovy = 49.1

        self.unet = UNet(
            9,
            14,
            down_channels=(64, 128, 256, 512, 1024, 1024),
            down_attention=(False, False, False, True, True, True),
            mid_attention=True,
            up_channels=(1024, 1024, 512, 256, 128),
            up_attention=(True, True, True, False, False),
        )

        self.conv = nn.Conv2d(14, 14, kernel_size=1)
        self.gs = GaussianRenderer(self.fovy, self.output_size)

        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5

    def prepare_default_rays(self, device, elevation=0):
        cam_poses = np.stack(
            [
                orbit_camera(elevation, 0, radius=self.radius),
                orbit_camera(elevation, 90, radius=self.radius),
                orbit_camera(elevation, 180, radius=self.radius),
                orbit_camera(elevation, 270, radius=self.radius),
            ],
            axis=0,
        )
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(
                cam_poses[i], self.input_size, self.input_size, self.fovy
            )
            rays_plucker = torch.cat(
                [torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1
            )
            rays_embeddings.append(rays_plucker)

        rays_embeddings = (
            torch.stack(rays_embeddings, dim=0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(device)
        )

        return rays_embeddings

    def forward(self, images):
        B, V, C, H, W = images.shape
        images = images.view(B * V, C, H, W)

        x = self.unet(images)
        x = self.conv(x)

        x = x.reshape(B, 4, 14, self.splat_size, self.splat_size)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)

        pos = self.pos_act(x[..., 0:3])
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        q = torch.tensor([0, 0, 1, 0], dtype=pos.dtype, device=pos.device)
        R = torch.tensor(
            [
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
            ],
            dtype=pos.dtype,
            device=pos.device,
        )

        pos = torch.matmul(pos, R.T)

        def multiply_quat(q1, q2):
            w1, x1, y1, z1 = q1.unbind(-1)
            w2, x2, y2, z2 = q2.unbind(-1)
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
            z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
            return torch.stack([w, x, y, z], dim=-1)

        for i in range(B):
            rotation[i, :] = multiply_quat(q, rotation[i, :])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)

        return gaussians


# =============================================================================
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py
# =============================================================================
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim_q, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim_k, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim_v, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, N, _ = q.shape
        M = k.shape[1]

        q = self.scale * self.to_q(q).reshape(
            B, N, self.num_heads, self.dim // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.to_k(k)
            .reshape(B, M, self.num_heads, self.dim // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.to_v(v)
            .reshape(B, M, self.num_heads, self.dim // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttention(CrossAttention):
    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(q, k, v)

        B, N, _ = q.shape
        M = k.shape[1]

        q = self.scale * self.to_q(q).reshape(
            B, N, self.num_heads, self.dim // self.num_heads
        )
        k = self.to_k(k).reshape(B, M, self.num_heads, self.dim // self.num_heads)
        v = self.to_v(v).reshape(B, M, self.num_heads, self.dim // self.num_heads)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# End of xFormers


class MVAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_frames: int = 4,
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(
            num_groups=groups, num_channels=dim, eps=eps, affine=True
        )
        self.attn = MemEffAttention(
            dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop
        )

    def forward(self, x):
        BV, C, H, W = x.shape
        B = BV // self.num_frames

        res = x
        x = self.norm(x)

        x = (
            x.reshape(B, self.num_frames, C, H, W)
            .permute(0, 1, 3, 4, 2)
            .reshape(B, -1, C)
        )
        x = self.attn(x)
        x = (
            x.reshape(B, self.num_frames, H, W, C)
            .permute(0, 1, 4, 2, 3)
            .reshape(BV, C, H, W)
        )

        if self.residual:
            x = (x + res) * self.skip_scale
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal["default", "up", "down"] = "default",
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.norm2 = nn.GroupNorm(
            num_groups=groups, num_channels=out_channels, eps=eps, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.act = F.silu

        self.resample = None
        if resample == "up":
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == "down":
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=True
            )

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.act(x)
        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        x = (x + self.shortcut(res)) * self.skip_scale
        return x


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(in_channels, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(
                    MVAttention(out_channels, attention_heads, skip_scale=skip_scale)
                )
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )

    def forward(self, x):
        xs = []
        for attn, net in zip(self.attns, self.nets):
            x = net(x)
            if attn:
                x = attn(x)
            xs.append(x)
        if self.downsample:
            x = self.downsample(x)
            xs.append(x)
        return x, xs


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
        for _ in range(num_layers):
            nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
            if attention:
                attns.append(
                    MVAttention(in_channels, attention_heads, skip_scale=skip_scale)
                )
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

    def forward(self, x):
        x = self.nets[0](x)
        for attn, net in zip(self.attns, self.nets[1:]):
            if attn:
                x = attn(x)
            x = net(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) else out_channels

            nets.append(ResnetBlock(cin + cskip, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(
                    MVAttention(out_channels, attention_heads, skip_scale=skip_scale)
                )
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.upsample = None
        if upsample:
            self.upsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x, xs):
        for attn, net in zip(self.attns, self.nets):
            res_x = xs[-1]
            xs = xs[:-1]
            x = torch.cat([x, res_x], dim=1)
            x = net(x)
            if attn:
                x = attn(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = self.upsample(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 14,
        down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024),
        down_attention: Tuple[bool, ...] = (False, False, False, True, True, True),
        mid_attention: bool = True,
        up_channels: Tuple[int, ...] = (1024, 1024, 512, 256, 128),
        up_attention: Tuple[bool, ...] = (True, True, True, False, False),
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels, down_channels[0], kernel_size=3, stride=1, padding=1
        )

        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(
                DownBlock(
                    cin,
                    cout,
                    num_layers=layers_per_block,
                    downsample=(i != len(down_channels) - 1),
                    attention=down_attention[i],
                    skip_scale=skip_scale,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        self.mid_block = MidBlock(
            down_channels[-1], attention=mid_attention, skip_scale=skip_scale
        )

        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))]

            up_blocks.append(
                UpBlock(
                    cin,
                    cskip,
                    cout,
                    num_layers=layers_per_block + 1,
                    upsample=(i != len(up_channels) - 1),
                    attention=up_attention[i],
                    skip_scale=skip_scale,
                )
            )
        self.up_blocks = nn.ModuleList(up_blocks)
        self.norm_out = nn.GroupNorm(
            num_channels=up_channels[-1], num_groups=32, eps=1e-5
        )
        self.conv_out = nn.Conv2d(
            up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.conv_in(x)
        xss = [x]
        for block in self.down_blocks:
            x, xs = block(x)
            xss.extend(xs)
        x = self.mid_block(x)
        for block in self.up_blocks:
            xs = xss[-len(block.nets) :]
            xss = xss[: -len(block.nets)]
            x = block(x, xs)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x
