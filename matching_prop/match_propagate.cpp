#include <torch/extension.h>

#include <stdlib.h>

torch::Tensor match_propagate_cuda_forward(
    torch::Tensor matching,
    torch::Tensor corr,
    torch::Tensor direction);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor match_propagate_forward(
    torch::Tensor matching,
    torch::Tensor corr,
    torch::Tensor direction)
{
    CHECK_INPUT(matching);
    CHECK_INPUT(corr);
    CHECK_INPUT(direction);
    AT_ASSERTM(matching.size(1) == 2, "The matching tensor must have 2D matching information");
    AT_ASSERTM(direction.size(0) == 2, "The direction tensor must have shape (2)");
    AT_ASSERTM(abs(direction[0].item<int>()) == 1 && abs(direction[1].item<int>()) == 1,
                "The direction tensor only contain values of 1 and -1");


    return match_propagate_cuda_forward(matching, corr, direction);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &match_propagate_forward, "Match propagation algorithm forward (CUDA)");
}