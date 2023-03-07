#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

template <typename scalar_t>
__device__ scalar_t bilinear_interpolation(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> corr,
    int n, int x, int y, scalar_t new_x, scalar_t new_y)
{
    int new_floor_x = static_cast<int>(new_x);
    int new_floor_y = static_cast<int>(new_y);
 
    scalar_t alpha = new_x - new_floor_x;
    scalar_t beta = new_y - new_floor_y;
    scalar_t a = alpha * corr[n][x][y][new_floor_x][new_floor_y] + (1 - alpha) * corr[n][x][y][new_floor_x + 1][new_floor_y];
    scalar_t b = alpha * corr[n][x][y][new_floor_x][new_floor_y + 1] + (1 - alpha) * corr[n][x][y][new_floor_x + 1][new_floor_y + 1];
    scalar_t res = beta * a + (1 - beta) * b;
    return res;
}

template <typename scalar_t>
__global__ void match_propagate_cuda_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> matching,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> corr,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> direction)
{
    // Used for grid-level synchronization
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    // Batch Index
    const int n = blockIdx.y;
    // Track diagonal index for each batch
    extern __shared__ int diagonal_index[];
    // Index in diagonal for all grid threads
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int H = matching.size(3);
    const int W = matching.size(4);
    const int total_diagonals = W + H;
    const int x_change = direction[0];
    const int y_change = direction[1];

    // Setup initial diagonal
    int x, y;
    if (x_change == 1 && y_change == 1)
        x = -c, y = c;
    else if (x_change == -1 && y_change == -1)
        x = c + H - 1, y = W - c - 1;
    else if (x_change == 1 && y_change == -1)
        x = -c, y = W - 1 - c;
    else if (x_change == -1 && y_change == 1)
        x = c + H - 1, y = W - 1 - c;

    if (c == 0)
        diagonal_index[n] = 0;
    grid.sync();
    for ( ; diagonal_index[n] < total_diagonals; )
    {
        if ((0 <= x && x < H) && (0 <= y && y < W))
        {
            int disp_x, disp_y;
            scalar_t new_x, new_y;
            scalar_t value = bilinear_interpolation(corr, n, x, y, matching[n][0][x][y], matching[n][1][x][y]);
            scalar_t new_value = value;

            // Handle X-wise displacement
            disp_x = x - x_change;
            disp_y = y;
            if ((0 <= disp_x && disp_x < H) && (0 <= disp_y && disp_y < W))
            {
                new_x = matching[n][0][disp_x][disp_y];
                new_y = matching[n][1][disp_x][disp_y];

                // Check for out-of-bounds access
                bool should_calc = false;
                if ((0 <= new_x + x_change && new_x + x_change < H - 1) && (0 <= new_y && new_y < W - 1))
                    should_calc = true;
                bool is_valid_lookup = (new_x != (scalar_t) disp_x || new_y != (scalar_t) disp_y);
                if (should_calc && is_valid_lookup)
                {
                    new_value = bilinear_interpolation(corr, n, x, y, new_x + x_change, new_y);
                    if (new_value > value)
                    {
                        value = new_value;
                        matching[n][0][x][y] = new_x;
                        matching[n][1][x][y] = new_y;
                    }
                }
            }

            // Handle Y-wise displacement
            disp_x = x;
            disp_y = y - y_change;
            if ((0 <= disp_x && disp_x < H) && (0 <= disp_y && disp_y < W))
            {
                new_x = matching[n][0][disp_x][disp_y];
                new_y = matching[n][1][disp_x][disp_y];

                // Check for out-of-bounds access
                bool should_calc = false;
                if ((0 <= new_x && new_x < H - 1) && (0 <= new_y + y_change && new_y + y_change < W - 1))
                    should_calc = true;
                bool is_valid_lookup = (new_x != (scalar_t) disp_x || new_y != (scalar_t) disp_y);
                if (should_calc && is_valid_lookup)
                {
                    new_value = bilinear_interpolation(corr, n, x, y, new_x, new_y + y_change);
                    if (new_value > value)
                    {
                        value = new_value;
                        matching[n][0][x][y] = new_x;
                        matching[n][1][x][y] = new_y;
                    }
                }
            }
        }
        x += x_change;
        if (c == 0)
            diagonal_index[n]++;
        grid.sync();
    }
}

torch::Tensor match_propagate_cuda_forward(
    torch::Tensor matching,
    torch::Tensor corr,
    torch::Tensor direction)
{
    const auto batch_size = matching.size(0);
    const auto max_diag_size = max(matching.size(2), matching.size(3));

    const int threads = 1024;
    const dim3 blocks((max_diag_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(corr.type(), "match_propagate_cuda_forward", ([&]
        {
            match_propagate_cuda_forward_kernel<scalar_t><<<blocks, threads, batch_size * sizeof(int)>>>(
                matching.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                corr.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                direction.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
        })
    );

    return matching;
}