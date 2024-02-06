"""
CODE FROM https://github.com/lucidrains/DALLE-pytorch/blob/58c1e1a4fef10725a79bd45cdb5581c03e3e59e7/dalle_pytorch/vae.py
"""
from math import log2, sqrt

import torch
import torch.nn.functional as F
from torch import einsum, nn
from ..persistence import PersistableModel

import pydantic

from einops import rearrange


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def is_empty(t):
    return t.nelement() == 0


def masked_mean(t, mask, dim=1):
    t = t.masked_fill(~mask[:, :, None], 0.)
    return t.sum(dim=dim) / mask.sum(dim=dim)[..., None]


def prob_mask_like(shape, prob, device):
    return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# sampling helpers

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class Config(pydantic.BaseModel):
    image_size: int
    num_tokens: int
    num_classes: int
    codebook_dim: int
    num_layers: int
    num_resnet_blocks: int
    hidden_dim: int
    channels: int
    smooth_l1_loss = False
    temperature = 0.9
    straight_through = False
    reinmax = False
    kl_div_loss_weight = 0.0


class DiscreteVAE(PersistableModel[Config]):
    def __init__(self, c: Config):
        super().__init__()
        assert log2(c.image_size).is_integer(), 'image size must be a power of 2'
        assert c.num_layers >= 1, 'number of layers must be greater than or equal to 1'
        self._config = c
        has_resblocks = c.num_resnet_blocks > 0
        self.channels = c.channels
        self.image_size = c.image_size
        self.num_tokens = c.num_tokens
        self.num_layers = c.num_layers
        self.temperature = c.temperature
        self.straight_through = c.straight_through
        self.reinmax = c.reinmax
        self.codebook = nn.Embedding(c.num_tokens, c.codebook_dim)

        enc_chans = [c.hidden_dim] * c.num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [c.channels, *enc_chans]

        dec_init_chan = c.codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=enc_in, out_channels=enc_out, kernel_size=4, stride=2, padding=1),
                nn.ReLU()))
            dec_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=dec_in, out_channels=dec_out, kernel_size=4, stride=2, padding=1),
                nn.ReLU()))

        for _ in range(c.num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if c.num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(c.codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], c.num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], c.channels, 1))

        enc_layers.insert(0, nn.Sequential(nn.Conv2d(in_channels=c.num_classes, out_channels=c.channels, kernel_size=1), nn.ReLU()))
        dec_layers.append(nn.Sequential(nn.Conv2d(in_channels=c.channels, out_channels=c.num_classes, kernel_size=1), nn.ReLU()))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = nn.CrossEntropyLoss()
        self.kl_div_loss_weight = c.kl_div_loss_weight

    def config(self) -> Config:
        return self._config

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits=True)
        codebook_indices = logits.argmax(dim=1).flatten(1)
        return codebook_indices

    def decode(self, img_seq):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h=h, w=w)
        images = self.decoder(image_embeds)
        return images

    def forward(
            self,
            img,
            return_loss=False,
            return_recons=False,
            return_logits=False,
            temp=None
    ):
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size and img.shape[-2] == image_size, f'input must have the correct image size {image_size}'

        logits = self.encoder(img)

        if return_logits:
            return logits  # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)

        one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=self.straight_through)

        if self.straight_through and self.reinmax:
            # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
            # algorithm 2
            one_hot = one_hot.detach()
            π0 = logits.softmax(dim=1)
            π1 = (one_hot + (logits / temp).softmax(dim=1)) / 2
            π1 = ((log(π1) - logits).detach() + logits).softmax(dim=1)
            π2 = 2 * π1 - 0.5 * π0
            one_hot = π2 - π2.detach() + one_hot

        sampled = einsum('b n h w, n d -> b d h w', one_hot, self.codebook.weight)
        out = self.decoder(sampled)
        if not return_loss:
            return out

        # reconstruction loss
        # out = (b, n_rooms, 32, 32) -> (b*32*32, n_rooms)
        # img = (b, n_rooms, 32, 32) -> (b*32*32, as)
        recon_loss = self.loss_fn(
            input=rearrange(out, 'b n h w -> (b h w) n'),
            target=torch.argmax(
                input=rearrange(img, 'b n h w -> (b h w) n'),
                dim=-1)
        )

        # kl divergence
        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device=device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out
