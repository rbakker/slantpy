"""The SLANT 3D U-Net (python/torchsrc/models/Unet3D.py) for current PyTorch.

Parameter names are unchanged, so the released ``model_epoch_0027.pth`` files
load directly. Two things in the original matter for reproducing its output:

* The encoder blocks use ``BatchNorm2d`` on 5-D tensors (accepted by PyTorch
  0.2). ``BatchNorm3d`` has the same parameters and the same arithmetic.
* ``Trainer.validate()`` calls ``model.train()``, so at test time batch norm
  normalises with the statistics of the tile being segmented, not with the
  stored running averages. ``batch_stats=True`` (default) reproduces that.

``forward``/``labels`` are the literal network. ``forward_lean``/``labels_lean``
compute the same function with a smaller memory footprint (docs/design-notes.md).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, in_channel=1, n_classes=133):
        super().__init__()
        enc, dec = self._encoder, self._decoder
        self.ec0 = enc(in_channel, 32)
        self.ec1 = enc(32, 64)
        self.ec2 = enc(64, 64)
        self.ec3 = enc(64, 128)
        self.ec4 = enc(128, 128)
        self.ec5 = enc(128, 256)
        self.ec6 = enc(256, 256)
        self.ec7 = enc(256, 512)
        self.pool = nn.MaxPool3d(2)
        self.dc9 = dec(512, 512, 2, 2, 0)
        self.dc8 = dec(256 + 512, 256, 3, 1, 1)
        self.dc7 = dec(256, 256, 3, 1, 1)
        self.dc6 = dec(256, 256, 2, 2, 0)
        self.dc5 = dec(128 + 256, 133, 3, 1, 1)
        self.dc4 = dec(133, 133, 3, 1, 1)
        self.dc3 = dec(133, 133, 2, 2, 0)
        self.dc2 = dec(64 + 133, 133, 3, 1, 1)
        self.dc1 = dec(133, 133, 3, 1, 1)
        # present in the checkpoints but never used by forward()
        self.dc0 = dec(64, n_classes, 1, 1, 0)
        self.batch_stats = True

    @staticmethod
    def _encoder(cin, cout):
        return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.BatchNorm3d(cout), nn.ReLU())

    @staticmethod
    def _decoder(cin, cout, k, stride, pad):
        return nn.Sequential(nn.ConvTranspose3d(cin, cout, k, stride=stride, padding=pad), nn.ReLU())

    # ---- literal forward pass -------------------------------------------------
    def forward(self, x):
        syn0 = self.ec1(self.ec0(x))
        syn1 = self.ec3(self.ec2(self.pool(syn0)))
        syn2 = self.ec5(self.ec4(self.pool(syn1)))
        e7 = self.ec7(self.ec6(self.pool(syn2)))
        d7 = self.dc7(self.dc8(torch.cat((self.dc9(e7), syn2), 1)))
        d4 = self.dc4(self.dc5(torch.cat((self.dc6(d7), syn1), 1)))
        return self.dc1(self.dc2(torch.cat((self.dc3(d4), syn0), 1)))

    def train(self, mode=True):
        super().train(mode)
        if self.batch_stats:      # keep the original's test-time behaviour in eval() too
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.training = True
        return self

    # ---- same function, lower memory -------------------------------------------
    def _enc(self, blk, x):
        conv, bn = blk[0], blk[1]
        x = conv(x)
        if self.batch_stats:
            x = F.batch_norm(x, None, None, bn.weight, bn.bias, True, 0.0, bn.eps)
        else:
            x = F.batch_norm(x, bn.running_mean, bn.running_var, bn.weight, bn.bias, False, 0.0, bn.eps)
        return F.relu_(x)

    @staticmethod
    def _dec3(blk, *inputs):
        """Stride-1 3x3x3 transposed convolution written as an ordinary convolution
        (flipped kernel, in/out swapped). Several inputs stand for their channel
        concatenation; they are convolved separately and summed, so the
        concatenated tensor is never built."""
        w = blk[0].weight.transpose(0, 1).flip(2, 3, 4)
        out, c0 = None, 0
        for i, x in enumerate(inputs):
            c1 = c0 + x.shape[1]
            y = F.conv3d(x, w[:, c0:c1], blk[0].bias if i == 0 else None, padding=1)
            out = y if out is None else out.add_(y)
            c0 = c1
            del y
        return F.relu_(out)

    @staticmethod
    def _up(blk, x):
        return F.relu_(blk[0](x))

    def _trunk(self, x):
        """Everything up to the half-resolution decoder output d4, plus the
        full-resolution skip connection syn0."""
        syn0 = self._enc(self.ec1, self._enc(self.ec0, x))
        syn1 = self._enc(self.ec3, self._enc(self.ec2, self.pool(syn0)))
        syn2 = self._enc(self.ec5, self._enc(self.ec4, self.pool(syn1)))
        d = self._enc(self.ec7, self._enc(self.ec6, self.pool(syn2)))
        d = self._dec3(self.dc8, self._up(self.dc9, d), syn2)
        del syn2
        d = self._dec3(self.dc7, d)
        d = self._dec3(self.dc5, self._up(self.dc6, d), syn1)
        del syn1
        return self._dec3(self.dc4, d), syn0

    def forward_lean(self, x):
        """Same output as forward() (133 channels at full resolution)."""
        d, syn0 = self._trunk(x)
        d = self._dec3(self.dc2, self._up(self.dc3, d), syn0)
        del syn0
        return self._dec3(self.dc1, d)

    def labels(self, x):
        """Class index per voxel, literal network: forward(x).argmax(1)."""
        return self.forward(x).argmax(1).to(torch.uint8)

    def labels_lean(self, x, slab=16):
        """Class index per voxel without ever holding a 133-channel full-resolution
        tensor. The last three layers (dc3, dc2, dc1) contain no normalisation, so
        they are evaluated slab by slab along the first spatial axis with a
        2-voxel halo, which gives the same values as evaluating them at once."""
        d4, syn0 = self._trunk(x)
        n = syn0.shape[2]
        out = torch.empty((x.shape[0],) + tuple(syn0.shape[2:]), dtype=torch.uint8, device=x.device)
        for s0 in range(0, n, slab):
            s1 = min(s0 + slab, n)
            lo = max(((s0 - 2) // 2) * 2, 0)                # even, includes the halo
            hi = min(-((-(s1 + 2)) // 2) * 2, n)
            up = self._up(self.dc3, d4[:, :, lo // 2:(hi + 1) // 2])[:, :, :hi - lo]
            h = self._dec3(self.dc2, up, syn0[:, :, lo:hi])
            del up
            h = self._dec3(self.dc1, h)
            out[:, s0:s1] = h[:, :, s0 - lo:s1 - lo].argmax(1)
            del h
        return out


def load_weights(model, path):
    """Load a released checkpoint (.pth written by PyTorch 0.2, or a .npz export)."""
    if path.endswith(".npz"):
        import numpy as np
        with np.load(path) as z:
            state = {k: torch.from_numpy(z[k]) for k in z.files}
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Python-2 era pickle. Only do this for checkpoints you trust
            # (here: the files shipped in the SLANT image).
            state = torch.load(path, map_location="cpu", weights_only=False, encoding="latin1")
    if not isinstance(state, dict):
        state = state.state_dict()
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked") and not k.startswith("dc0.")]
    if missing or unexpected:
        raise RuntimeError("checkpoint %s does not match the network (missing %s, unexpected %s)"
                           % (path, missing, list(unexpected)))
    return model
