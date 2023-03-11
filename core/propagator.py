import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.cpp_extension import load
matching_prop = load(name="matching_prop", sources=["matching_prop/match_propagate.cpp",
                                                    "matching_prop/match_propagate_kernel.cu"])
from utils.utils import bilinear_sampler

class MatchingPropagator(nn.Module):
    UPDATE_MULTIPLIER_THRESHOLD = 1.0
    def __init__(self, search_range=3, forward_backward_error=0.01):
        super(MatchingPropagator, self).__init__()
        self.r = search_range
        self.eps = forward_backward_error
    

    def __get_scores(self, coords, corr_map):
        H, W = coords.shape[1:3]
        gridify_coords = coords.view(-1, 1, 1, 2)
        volumify_corr_map = corr_map.view(-1, 1, H, W)
        return bilinear_sampler(volumify_corr_map, gridify_coords).view(-1, H, W)

    
    def __random_search(self, coords, corr_map, forward):
        H, W = coords.shape[1:3]
        new_coords = torch.normal(coords, self.r)
        # Clip the newly calculated coordinates
        torch.clamp(new_coords[..., 1], min=0, max=H - 1, out=new_coords[..., 1])
        torch.clamp(new_coords[..., 0], min=0, max=W - 1, out=new_coords[..., 0])

        old_scores = matching_prop.get_scores(coords, corr_map, forward)
        new_scores = matching_prop.get_scores(new_coords, corr_map, forward)
        update_ind = (new_scores - MatchingPropagator.UPDATE_MULTIPLIER_THRESHOLD * old_scores > 0) \
                        .unsqueeze_(dim=-1).expand(-1, -1, -1, 2)
        coords[update_ind] = new_coords[update_ind]
        return coords

    
    def __propagate(seld, coords, corr_map, direction):
        return matching_prop.propagate(coords, corr_map, direction)


    def __handle(self, matching, corr_map, forward):
        coords = torch.permute(matching, (0, 2, 3, 1)).contiguous()
        res = self.__propagate(coords, corr_map, torch.Tensor([1, 1]).cuda())    # Propagate down and right
        res = self.__random_search(res, corr_map, forward)
        res = self.__propagate(res, corr_map, torch.Tensor([-1, -1]).cuda())     # Propagate up and left
        res = self.__random_search(res, corr_map, forward)
        res = self.__propagate(res, corr_map, torch.Tensor([-1, 1]).cuda())      # Propagate up and right
        res = self.__random_search(res, corr_map, forward)
        res = self.__propagate(res, corr_map, torch.Tensor([1, -1]).cuda())      # Propagate down and left
        return res


    def __get_invalid_fb_matches(self, res_f, res_b):
        BATCH_SIZE = res_f.shape[0]

        res_f_lookup = res_f.view(BATCH_SIZE, -1, 1, 1, 2)
        volumify_res_b = torch.permute(res_b, (0, 3, 1, 2))
        volumify_res_b = volumify_res_b.unsqueeze_(dim=1).expand(-1, res_f_lookup.shape[1], -1, -1, -1)
        res_f_counter_flow = torch.zeros((BATCH_SIZE, res_f.shape[-1], res_f.shape[1], res_f.shape[2])).cuda()
        for i in range(BATCH_SIZE):
            res_f_counter_flow[i] = bilinear_sampler(volumify_res_b[i], res_f_lookup[i]).view(*res_f_counter_flow.shape[1:])
        res_f_counter_flow = torch.permute(res_f_counter_flow, (0, 2, 3, 1))
        absolute_difference = torch.abs(res_f - res_f_counter_flow).max(dim=-1)[0]
        invalid_matches = (absolute_difference > self.eps).unsqueeze_(dim=-1).expand(-1, -1, -1, 2)
        
        return invalid_matches


    def __call__(self, matching_f, matching_b, corr_map):
        res_f = self.__handle(matching_f, corr_map, True)
        res_b = self.__handle(matching_b, corr_map, False)

        invalid_matches = self.__get_invalid_fb_matches(res_f, res_b)
        invalid_matches = torch.permute(invalid_matches, (0, 3, 1, 2))
        res = torch.permute(res_f, (0, 3, 1, 2))
        
        res[invalid_matches] = matching_f[invalid_matches]
        return res