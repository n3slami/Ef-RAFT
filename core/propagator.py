import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.cpp_extension import load
matching_prop = load(name="matching_prop", sources=["matching_prop/match_propagate.cpp",
                                                    "matching_prop/match_propagate_kernel.cu"])
from utils.utils import bilinear_sampler

class MatchingPropagator(nn.Module):
    UPDATE_MULTIPLIER_THRESHOLD = 1.05
    def __init__(self, search_range=3):
        super(MatchingPropagator, self).__init__()
        self.r = search_range
    

    def __get_scores(self, coords, corr_map):
        H, W = coords.shape[1:3]
        gridify_coords = coords.view(-1, 2).unsqueeze_(dim=-2).unsqueeze_(dim=-2)
        volumify_corr_map = corr_map.view(-1, 1, H, W)
        return bilinear_sampler(volumify_corr_map, gridify_coords).view(-1, H, W)

    
    def __random_search(self, coords, corr_map):
        H, W = coords.shape[1:3]
        new_coords = torch.normal(coords, self.r)
        # Clip the newly calculated coordinates
        new_coords[new_coords < 0] = 0
        new_coords[new_coords[..., 0] >= H] = H - 1
        new_coords[new_coords[..., 1] >= W] = W - 1

        old_scores = self.__get_scores(coords, corr_map)
        new_scores = self.__get_scores(new_coords, corr_map)
        update_ind = new_scores > MatchingPropagator.UPDATE_MULTIPLIER_THRESHOLD * old_scores
        coords[update_ind] = new_coords[update_ind]
        return coords


    def __call__(self, raw_coords, corr_map):
        torch.set_printoptions(profile="full")
        coords = torch.permute(raw_coords, (0, 2, 3, 1)).contiguous()
        # print(coords.shape)
        # print("BEFORE:\n", coords[0,:14,:14,:])
        res = matching_prop.forward(coords, corr_map, torch.Tensor([1, 1]).cuda())
        res = self.__random_search(res, corr_map)
        res = matching_prop.forward(coords, corr_map, torch.Tensor([-1, -1]).cuda())
        res = self.__random_search(res, corr_map)
        res = matching_prop.forward(coords, corr_map, torch.Tensor([-1, 1]).cuda())
        res = self.__random_search(res, corr_map)
        res = matching_prop.forward(coords, corr_map, torch.Tensor([1, -1]).cuda())
        # print("AFTER:\n", res[0,:14,:14,:])
        return torch.permute(res, (0, 3, 1, 2))