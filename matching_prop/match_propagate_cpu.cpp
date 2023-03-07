#include <torch/extension.h>

#include <stdlib.h>

int min(int a, int b)
{
    return (a < b ? a : b);
}

int max(int a, int b)
{
    return (a > b ? a : b);
}

inline bool is_valid(int x, int y, int H, int W)
{
    if (0 <= x && x < H)
        return (0 <= y && y < W);
    return false;
}

template <typename scalar_t>
scalar_t bilinear_interpolation(
    const torch::PackedTensorAccessor32<scalar_t, 5> corr,
    int n, int x, int y, scalar_t new_x, scalar_t new_y)
{
    int new_floor_x = static_cast<int>(new_x);
    int new_floor_y = static_cast<int>(new_y);

    // Check for out-of-bounds access
    if ((new_floor_x < 0 || new_floor_x + 1 >= corr.size(3))
        || (new_floor_y < 0 || new_floor_y + 1 >= corr.size(4)))
    {
        new_floor_x = max(min(new_floor_x, static_cast<int>(corr.size(3)) - 2), 0);
        new_floor_y = max(min(new_floor_y, static_cast<int>(corr.size(4)) - 2), 0);
        return corr[n][x][y][new_floor_x][new_floor_y];
    }
    
    scalar_t alpha = new_x - new_floor_x;
    scalar_t beta = new_y - new_floor_y;
    scalar_t a = alpha * corr[n][x][y][new_floor_x][new_floor_y] + (1 - alpha) * corr[n][x][y][new_floor_x + 1][new_floor_y];
    scalar_t b = alpha * corr[n][x][y][new_floor_x][new_floor_y + 1] + (1 - alpha) * corr[n][x][y][new_floor_x + 1][new_floor_y + 1];
    scalar_t res = beta * a + (1 - beta) * b;
    return res;
}

torch::Tensor match_propagate_cpu_forward(
    torch::Tensor matching,
    torch::Tensor corr,
    torch::Tensor direction)
{
    const int batch_size = matching.size(0);
    const int H = matching.size(1), W = matching.size(2);
    const int max_iter = W + H;
    const int delta_x = direction[0].item<int>();
    const int delta_y = direction[1].item<int>();
    matching[0][0][0][0] = 0;
}

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
    AT_ASSERTM(direction.size(0) == 2, "The direction tensor must have shape (2)");
    AT_ASSERTM(abs(direction[0].item<int>()) != 1 || abs(direction[1].item<int>()) != 1,
                "The direction tensor only contain values of 1 and -1");


    return match_propagate_cpu_forward(matching, corr, direction);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &match_propagate_forward, "Match propagation algorithm forward (CPU)");
}