"""Defines the epsilon prediction U-Net.

U-Net espilon prediction network from the paper "Denoising Diffusion Probabilistic Models"
(https://arxiv.org/abs/2006.11239)
"""

import torch

"""Utility layers used in defining a Denoising Diffusion Probabilistic Model."""

import math
import torch


class SinusoidalPositionEmbedding(torch.nn.Module):
    """Implementation of Sinusoidal Position Embedding.

    Originally introduced in the paper "Attention Is All You Need",
    the original tensorflow implementation is here:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L408
    """

    def __init__(self, embedding_dim, theta=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.embedding_dim // 2
        embedding = math.log(self.theta) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=device) * -embedding)
        embedding = x[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return embedding


class Block(torch.nn.Module):
    """
    A convolutional block which makes up the two convolutional
    layers in the ResnetBlock.
    """

    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = torch.nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=dim)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        # The original paper implementation uses norm->swish->projection
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)
        return x


class ResnetBlock(torch.nn.Module):
    """ResNet block based on WideResNet architecture.

    From DDPM, uses GroupNorm instead of weight normalization and Swish activation.
    """

    def __init__(self, dim, dim_out, time_emb_dim=None):
        super().__init__()
        self.timestep_proj = (
            torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.residual_proj = torch.nn.Linear(dim, dim_out)
        self.dim_out = dim_out

    def forward(self, x, time_emb=None):
        B, C, H, W = x.shape

        h = self.block1(x)

        # Add in the timstep embedding between blocks 1 and 2
        if time_emb is not None and self.timestep_proj is not None:
            h += self.timestep_proj(time_emb)[:, :, None, None]

        h = self.block2(h)

        # Project the residual channel to the output dimensions
        if C != self.dim_out:
            x = self.residual_proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return h + x


class SelfAttention(torch.nn.Module):
    """One head of self-attention"""

    def __init__(self, input_channels):
        super().__init__()
        self.key = torch.nn.Linear(input_channels, input_channels, bias=False)
        self.query = torch.nn.Linear(input_channels, input_channels, bias=False)
        self.value = torch.nn.Linear(input_channels, input_channels, bias=False)
        self.proj = torch.nn.Linear(input_channels, input_channels, bias=False)
        self.normalize = torch.nn.GroupNorm(num_groups=32, num_channels=input_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.normalize(x)

        # Move channels to the end
        h = torch.permute(h, (0, 2, 3, 1))
        k = self.key(h)  # (B,H,W,C)
        q = self.query(h)  # (B,H,W,C)
        v = self.value(h)  # (B,H,W,C)

        # compute attention scores ("affinities")
        w = torch.einsum("bhwc,bHWc->bhwHW", q, k) * (int(C) ** (-0.5))  # (B,H,W,H,W)
        w = torch.reshape(w, [B, H, W, H * W])  # (B, H, W, H*W)
        w = torch.nn.functional.softmax(w, dim=-1)
        w = torch.reshape(w, [B, H, W, H, W])

        h = torch.einsum("bhwHW,bHWc->bhwc", w, v)
        h = self.proj(h)
        h = torch.permute(h, (0, 3, 1, 2))
        return x + h


class MNistUNet(torch.nn.Module):
    """A time-dependent score-based model built upon U-Net architecture.

    This UNet is designed to predict the score of each pixel, given an input image.
    That score could be the mean of the noise added to the original image, the mean
    and the variance of the added noise, the original
    image itself, the noise at a previsous timestep, or the re-parameterized mean
    of the added noise. Other than predicting the mean and variance (which requires
    additional output channels) the model is the same for each.

    The model here predicts epsilon, the re-parameterized mean of the added noise.

    The model is a general form for all types of data. However, it is sized for the
    MNIST dataset, and contains ~37m parameters. This matches the CIFAR-10 UNet model
    from the DDPM paper and implementation.
    """

    def __init__(self):
        super().__init__()

        # Original paper had channel multipliers of [1,2,2,2] and input_channels = 128
        channel_mults = [1, 2, 2, 2]
        input_channels = 128
        channels = list(map(lambda x: input_channels * x, channel_mults))

        # The time embedding dimension was 4*input_channels
        time_emb_dim = input_channels * 4

        # Timestep embedding projection
        self.time_proj = torch.nn.Sequential(
            SinusoidalPositionEmbedding(input_channels),
            torch.nn.Linear(input_channels, time_emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Original paper implementation had kernel size = 3, stride = 1
        self.initial_convolution = torch.nn.Conv2d(
            in_channels=1,
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # The down/up sampling layers have 4 feature map resolutions for 32x32 input.
        # Note that we are padding the MNist dataset to 32x32 from 28x28 to make the math
        # works out easier. All resolution levels have two convolutional residual blocks,
        # and self-attention layers at the 16 level between the convolutions.

        self.downs = torch.nn.ModuleList(
            [
                # Input (B, 128, 32, 32) Output (B, 128, 16, 16)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=input_channels,
                            dim_out=channels[0],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        ResnetBlock(
                            dim=channels[0],
                            dim_out=channels[0],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        torch.nn.Conv2d(
                            channels[0], channels[0], 3, padding=1, stride=2
                        ),
                    ]
                ),
                # Input (B, 128, 16 , 16) Output (B, 256, 8, 8)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[0],
                            dim_out=channels[1],
                            time_emb_dim=time_emb_dim,
                        ),
                        SelfAttention(channels[1]),
                        ResnetBlock(
                            dim=channels[1],
                            dim_out=channels[1],
                            time_emb_dim=time_emb_dim,
                        ),
                        SelfAttention(channels[1]),
                        torch.nn.Conv2d(
                            channels[1], channels[1], 3, padding=1, stride=2
                        ),
                    ]
                ),
                # Input (B, 256, 8, 8), Output (B, 256, 4, 4)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[1],
                            dim_out=channels[2],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        ResnetBlock(
                            dim=channels[2],
                            dim_out=channels[2],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        torch.nn.Conv2d(
                            channels[2], channels[2], 3, padding=1, stride=2
                        ),
                    ]
                ),
                # Input (B, 256, 4, 4), Output (B, 256, 4, 4)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[2],
                            dim_out=channels[3],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        ResnetBlock(
                            dim=channels[3],
                            dim_out=channels[3],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        torch.nn.Identity(),
                    ]
                ),
            ]
        )

        # Middle layers
        self.middle = torch.nn.ModuleList(
            [
                # Input (B, 256, 4, 4), Output (B, 256, 4, 4)
                ResnetBlock(
                    dim=channels[3], dim_out=channels[3], time_emb_dim=time_emb_dim
                ),
                SelfAttention(channels[3]),
                # Input (B, 256, 4, 4), Output (B, 256, 4, 4)
                ResnetBlock(
                    dim=channels[3], dim_out=channels[3], time_emb_dim=time_emb_dim
                ),
            ]
        )

        # Upsampling layers
        self.ups = torch.nn.ModuleList(
            [
                # Input (B, 256, 4, 4), Output (B, 256, 8, 8)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[3] + channels[3],
                            dim_out=channels[3],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        ResnetBlock(
                            dim=channels[3] + channels[3],
                            dim_out=channels[3],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        ResnetBlock(
                            dim=channels[3] + channels[2],
                            dim_out=channels[2],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        torch.nn.Sequential(
                            torch.nn.Upsample(scale_factor=2, mode="nearest"),
                            torch.nn.Conv2d(channels[2], channels[2], 3, padding=1),
                        ),
                    ]
                ),
                # Input (B, 256, 8, 8), Output (B, 256, 16, 16)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[2] + channels[2],
                            dim_out=channels[2],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        ResnetBlock(
                            dim=channels[2] + channels[2],
                            dim_out=channels[2],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        ResnetBlock(
                            dim=channels[2] + channels[1],
                            dim_out=channels[1],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        torch.nn.Sequential(
                            torch.nn.Upsample(scale_factor=2, mode="nearest"),
                            torch.nn.Conv2d(channels[1], channels[1], 3, padding=1),
                        ),
                    ]
                ),
                # Input (B, 256, 16, 16), Output (B, 256, 32, 32)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[1] + channels[1],
                            dim_out=channels[1],
                            time_emb_dim=time_emb_dim,
                        ),
                        SelfAttention(channels[1]),
                        ResnetBlock(
                            dim=channels[1] + channels[1],
                            dim_out=channels[1],
                            time_emb_dim=time_emb_dim,
                        ),
                        SelfAttention(channels[1]),
                        ResnetBlock(
                            dim=channels[1] + channels[0],
                            dim_out=channels[0],
                            time_emb_dim=time_emb_dim,
                        ),
                        SelfAttention(channels[0]),
                        torch.nn.Sequential(
                            torch.nn.Upsample(scale_factor=2, mode="nearest"),
                            torch.nn.Conv2d(channels[0], channels[0], 3, padding=1),
                        ),
                    ]
                ),
                # Input (B, 128, 32, 32), Output (B, 128, 32, 32)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[0] + channels[0],
                            dim_out=channels[0],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        ResnetBlock(
                            dim=channels[0] + channels[0],
                            dim_out=channels[0],
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        ResnetBlock(
                            dim=channels[0] + input_channels,
                            dim_out=input_channels,
                            time_emb_dim=time_emb_dim,
                        ),
                        torch.nn.Identity(),
                        torch.nn.Identity(),
                    ]
                ),
            ]
        )

        # Final projection
        self.final_projection = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=input_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x, t, y=None):
        # Convert the timestep t to an embedding
        timestep_embedding = self.time_proj(t)

        # Initial convolution
        h = self.initial_convolution(x)  # B,C=1,H,W -> B,C=32,H,W

        # Downsampling blocks
        skips = [h]
        for i, layer in enumerate(self.downs):
            block1, attn1, block2, attn2, downsample = layer
            h = block1(h, time_emb=timestep_embedding)
            h = attn1(h)
            skips.append(h)
            h = block2(h, time_emb=timestep_embedding)
            h = attn2(h)
            skips.append(h)
            h = downsample(h)

            if i != len(self.downs) - 1:
                skips.append(h)

        # Middle layers
        middle_block1, middle_attn, middle_block2 = self.middle
        h = middle_block1(h, time_emb=timestep_embedding)
        h = middle_attn(h)
        h = middle_block2(h, time_emb=timestep_embedding)

        for i, layer in enumerate(self.ups):
            block1, attn1, block2, attn2, block3, attn3, upsample = layer

            h = block1(torch.cat([h, skips.pop()], dim=1), time_emb=timestep_embedding)
            h = attn1(h)
            h = block2(torch.cat([h, skips.pop()], dim=1), time_emb=timestep_embedding)
            h = attn2(h)
            h = block3(torch.cat([h, skips.pop()], dim=1), time_emb=timestep_embedding)
            h = attn3(h)
            h = upsample(h)

        h = self.final_projection(h)
        return h