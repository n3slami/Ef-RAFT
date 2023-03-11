#include <torch/extension.h>

#include <stdio.h>
#include <stdlib.h>

torch::Tensor match_propagate_cuda_propagate(
    torch::Tensor matching,
    torch::Tensor corr,
    torch::Tensor direction);

torch::Tensor match_propagate_cuda_get_scores(
    torch::Tensor matching,
    torch::Tensor corr,
    bool is_forward);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor match_propagate_propagate(
    torch::Tensor matching,
    torch::Tensor corr,
    torch::Tensor direction)
{
    CHECK_INPUT(matching);
    CHECK_CUDA(corr);
    CHECK_INPUT(direction);

    AT_ASSERTM(matching.size(3) == 2, "The matching tensor must have 2D matching information being stored in axis=3");
    AT_ASSERTM(direction.size(0) == 2, "The direction tensor must have shape (2)");
    AT_ASSERTM(abs(direction[0].item<int>()) == 1 && abs(direction[1].item<int>()) == 1,
                "The direction tensor only contain values of 1 and -1");

    return match_propagate_cuda_propagate(matching, corr, direction);
}

torch::Tensor match_propagate_get_scores(
    torch::Tensor matching,
    torch::Tensor corr,
    bool forward)
{
    CHECK_INPUT(matching);
    CHECK_CUDA(corr);
    AT_ASSERTM(matching.size(3) == 2, "The matching tensor must have 2D matching information being stored in axis=3");

    return match_propagate_cuda_get_scores(matching, corr, forward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("propagate", &match_propagate_propagate, "Match propagation algorithm (CUDA)");
    m.def("get_scores", &match_propagate_get_scores, "Match propagation algorithm score calculation (CUDA)");
}