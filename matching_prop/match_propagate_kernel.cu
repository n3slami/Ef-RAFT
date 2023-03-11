#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <stdio.h>

__device__ bool is_valid_coord(size_t H, size_t W, int x, int y)
{
    if (0 <= x && x < H)
        return (0 <= y && y < W);
    return false;
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolation(
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> corr,
    int n, int x, int y, scalar_t new_x, scalar_t new_y, const bool forward=true)
{
    const size_t H = corr.size(1);
    const size_t W = corr.size(2);

    int new_floor_x = static_cast<int>(new_x);
    int new_floor_y = static_cast<int>(new_y);

    scalar_t alpha = new_x - new_floor_x;
    scalar_t beta = new_y - new_floor_y;
    scalar_t top_left = 0, top_right = 0, bottom_left = 0, bottom_right = 0;
    
    if (is_valid_coord(H, W, new_floor_x, new_floor_y))
        top_left = forward ? corr[n][x][y][new_floor_x][new_floor_y] : corr[n][new_floor_x][new_floor_y][x][y];
    
    if (is_valid_coord(H, W, new_floor_x + 1, new_floor_y))
        bottom_left = forward ? corr[n][x][y][new_floor_x + 1][new_floor_y] : corr[n][new_floor_x + 1][new_floor_y][x][y];
    
    if (is_valid_coord(H, W, new_floor_x, new_floor_y + 1))
        top_right = forward ? corr[n][x][y][new_floor_x][new_floor_y + 1] : corr[n][new_floor_x][new_floor_y + 1][x][y];
    
    if (is_valid_coord(H, W, new_floor_x + 1, new_floor_y + 1))
        bottom_right = forward ? corr[n][x][y][new_floor_x + 1][new_floor_y + 1] : corr[n][new_floor_x + 1][new_floor_y + 1][x][y];
    
    scalar_t a = (1 - alpha) * top_left + alpha * bottom_left;
    scalar_t b = (1 - alpha) * top_right + alpha * bottom_right;
    scalar_t res = (1 - beta) * a + beta * b;
    return res;
}

template <typename scalar_t>
__global__ void match_propagate_cuda_propagate_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> matching,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> corr,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> direction)
{
    // Used for grid-level synchronization
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const float update_threshold_multiplier = 1.05;
    
    // Batch Index
    const int n = blockIdx.y;
    // Track diagonal index for each batch
    extern __shared__ int diagonal_index[];
    // Index in diagonal for all grid threads
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int H = matching.size(1);
    const int W = matching.size(2);
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
            scalar_t value = bilinear_interpolation<scalar_t>(corr, n, x, y, matching[n][x][y][1], matching[n][x][y][0]);
            scalar_t base_value = (value > 0 ? value : 0);
            scalar_t new_value = value;

            // Handle X-wise displacement
            disp_x = x - x_change;
            disp_y = y;
            if ((0 <= disp_x && disp_x < H) && (0 <= disp_y && disp_y < W))
            {
                new_x = matching[n][disp_x][disp_y][1];
                new_y = matching[n][disp_x][disp_y][0];

                // Check for out-of-bounds access
                bool should_calc = false;
                if ((0 <= new_x + x_change && new_x + x_change <= H - 1) && (0 <= new_y && new_y <= W - 1))
                    should_calc = true;
                bool is_valid_lookup = (new_x != (scalar_t) disp_x || new_y != (scalar_t) disp_y);
                if (should_calc && is_valid_lookup)
                {
                    new_value = bilinear_interpolation<scalar_t>(corr, n, x, y, new_x + x_change, new_y);
                    if (new_value > value && new_value > base_value * update_threshold_multiplier)
                    {
                        value = new_value;
                        // if (new_x + x_change > H - 1 || new_y > W - 1)
                        //     printf("WELL SHIT: %f %f\n", new_x, new_y + y_change);
                        matching[n][x][y][1] = new_x + x_change;
                        matching[n][x][y][0] = new_y;
                    }
                }
            }

            // Handle Y-wise displacement
            disp_x = x;
            disp_y = y - y_change;
            if ((0 <= disp_x && disp_x < H) && (0 <= disp_y && disp_y < W))
            {
                new_x = matching[n][disp_x][disp_y][1];
                new_y = matching[n][disp_x][disp_y][0];

                // Check for out-of-bounds access
                bool should_calc = false;
                if ((0 <= new_x && new_x <= H - 1) && (0 <= new_y + y_change && new_y + y_change <= W - 1))
                    should_calc = true;
                bool is_valid_lookup = (new_x != (scalar_t) disp_x || new_y != (scalar_t) disp_y);
                if (should_calc && is_valid_lookup)
                {
                    new_value = bilinear_interpolation<scalar_t>(corr, n, x, y, new_x, new_y + y_change);
                    if (new_value > value && new_value > base_value * update_threshold_multiplier)
                    {
                        value = new_value;
                        // if (new_x > H - 1 || new_y + y_change > W - 1)
                        //     printf("WELL SHIT: %f %f\n", new_x, new_y + y_change);
                        matching[n][x][y][1] = new_x;
                        matching[n][x][y][0] = new_y + y_change;
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

torch::Tensor match_propagate_cuda_propagate(
    torch::Tensor matching,
    torch::Tensor corr,
    torch::Tensor direction)
{
    const auto batch_size = matching.size(0);
    const auto max_diag_size = max(matching.size(1), matching.size(2));

    const int threads = 1024;
    const int block_count = (max_diag_size + threads - 1) / threads;
    const dim3 blocks(block_count, batch_size);
    

    AT_DISPATCH_FLOATING_TYPES(corr.type(), "match_propagate_cuda_propagate", ([&]() {
            auto matching_pa = matching.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
            auto corr_pa = corr.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>();
            auto direction_pa = direction.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>();
            void *args[4] = {&matching_pa, &corr_pa, &direction_pa, NULL};
            size_t smem_size = batch_size * sizeof(int);
            cudaStream_t stream = 0;
            cudaLaunchCooperativeKernel((void *) match_propagate_cuda_propagate_kernel<scalar_t>,
                                        blocks, threads, (void **) args, smem_size, stream);
            // match_propagate_cuda_propagate_kernel<scalar_t><<<blocks, threads, smem_size>>>(
            //     matching_pa, corr_pa, direction_pa);
        })
    );

    return matching;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void match_propagate_cuda_get_scores_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> matching,
    const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> corr,
    bool forward,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> res)
{
    // Batch Index
    const int n = blockIdx.y;
    // Index in diagonal for all grid threads
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    const int H = matching.size(1);
    const int W = matching.size(2);
    const int total_pixels = H * W;
    const int index_increment = gridDim.x * blockDim.x;

    while (ind < total_pixels)
    {
        int x = ind / W, y = ind % W;
        // if (matching[n][x][y][1] > H - 1 || matching[n][x][y][0] > W - 1)
        //     printf("WELP THIS IS PROBABLY THE REASON (%d: %d, %d): %f %f\n", n, x, y, matching[n][x][y][1], matching[n][x][y][0]);
        res[n][x][y] = bilinear_interpolation<scalar_t>(corr, n, x, y,
                                                        matching[n][x][y][1], matching[n][x][y][0],
                                                        forward);
        ind += index_increment;
    }
}


torch::Tensor match_propagate_cuda_get_scores(
    torch::Tensor matching,
    torch::Tensor corr,
    bool is_forward)
{
    const auto batch_size = matching.size(0);
    const auto image_size = matching.size(1) * matching.size(2);

    const int threads = 1024;
    const int block_count = (image_size + threads - 1) / threads;
    const dim3 blocks(block_count, batch_size);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto res = torch::zeros({matching.size(0), matching.size(1), matching.size(2)}, options);

    
    AT_DISPATCH_FLOATING_TYPES(corr.type(), "match_propagate_cuda_get_scores", ([&]() {
            auto matching_pa = matching.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
            auto corr_pa = corr.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>();
            auto res_pa = res.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
            match_propagate_cuda_get_scores_kernel<scalar_t><<<blocks, threads>>>(
                matching_pa, corr_pa, is_forward, res_pa);
        })
    );
    
    return res;
}