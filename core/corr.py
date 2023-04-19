import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        self.corr_map = corr.view(batch, h1 * w1, h2 * w2)
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords, transformations=None):
        r = self.radius

        if transformations is not None:
            assert(transformations.shape[-1] == 5 and transformations.shape[-2] == self.num_levels)
            transformations = transformations.view(-1, 1, transformations.shape[-2], transformations.shape[-1])

        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            centroid_lvl = coords.reshape(batch, h1*w1, 1, 1, 2) / 2**i
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            delta = delta.view(-1, 2)
            delta = delta.repeat((batch, 1, 1))
            if transformations is not None:
                # print(f"TRANSFORMATIONS at {i}", transformations[..., i])
                delta[..., 0] *= transformations[..., i, 0]
                delta[..., 1] *= transformations[..., i, 1]
                delta[..., 0] += torch.sign(delta[..., 0]) * transformations[..., i, 2] * r
                delta[..., 1] += torch.sign(delta[..., 1]) * transformations[..., i, 3] * r
                s = torch.sin(transformations[..., i, 4])
                c = torch.cos(transformations[..., i, 4])
                # print(s.shape, c.shape)
                rotation_mat_t = torch.concat([torch.stack([ c, s], dim=-1),
                                               torch.stack([-s, c], dim=-1)], dim=-2)
                # print(rotation_mat_t.shape, rotation_mat_t, "vs.", delta.shape)
                # print("WOW", delta[:, 0, :])
                delta = torch.matmul(delta, rotation_mat_t)
                # print(delta.shape)
                # print("WOW", delta[:, 0, :])
            delta_lvl = delta.view(batch, 1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            coords_lvl = coords_lvl.reshape(-1, coords_lvl.shape[-3], coords_lvl.shape[-2], coords_lvl.shape[-1])

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
