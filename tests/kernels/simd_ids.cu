// Lane and warp ids across block shapes. CUDA numbers the threads of a block x fastest and cuts that order into warps
// of 32; mvcc derives %laneid (linear & 31) and %warpid (linear >> 5) from the thread position the same way, rather
// than from Metal's [[thread_index_in_simdgroup]] / [[simdgroup_index_in_threadgroup]] (msl_emitter.cpp simdIds).
// The shuffle and ballot checks confirm the derived lane is the lane the hardware exchanges through.
// %warpid is volatile on NVIDIA hardware; mvcc's contract, checked here, is linear >> 5.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CK(x)                                                                                              \
  do {                                                                                                     \
    cudaError_t e_ = (x);                                                                                  \
    if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } \
  } while (0)

// out[4 * (block * threads + linear) + i]: laneid, warpid, a rotated shuffle of the linear id, and a ballot.
__global__ void ids_kernel(uint32_t* out) {
  const uint32_t t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const uint32_t b = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  const uint32_t n = blockDim.x * blockDim.y * blockDim.z;
  uint32_t lane, warp;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp));
  uint32_t* o = out + 4 * (b * n + t);
  o[0] = lane;
  o[1] = warp;
  o[2] = __shfl_sync(0xffffffffu, t, (lane + 5) & 31);
  o[3] = __ballot_sync(0xffffffffu, (t & 3) == 1);
}

// A per-warp partial sum through shared memory indexed by warp id, then one thread combines them: a wrong warp id
// makes two warps share a slot (or leaves one unwritten) and the block total comes out wrong.
__global__ void warp_slots_kernel(const uint32_t* in, uint32_t* out) {
  __shared__ uint32_t slots[32];
  const uint32_t t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const uint32_t n = blockDim.x * blockDim.y * blockDim.z;
  const uint32_t b = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  uint32_t lane, warp;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warp));
  uint32_t v = in[b * n + t];
  for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
  if (lane == 0) slots[warp] = v;
  __syncthreads();
  if (t == 0) {
    uint32_t s = 0;
    for (uint32_t w = 0; w < n / 32; ++w) s += slots[w];
    out[b] = s;
  }
}

static int failures = 0;

static void check_shape(dim3 grid, dim3 block) {
  const uint32_t n = block.x * block.y * block.z, nb = grid.x * grid.y * grid.z;
  uint32_t *d, *din, *dsum;
  CK(cudaMalloc(&d, 4ull * n * nb * sizeof(uint32_t)));
  CK(cudaMalloc(&din, (size_t)n * nb * sizeof(uint32_t)));
  CK(cudaMalloc(&dsum, nb * sizeof(uint32_t)));
  std::vector<uint32_t> in((size_t)n * nb);
  for (size_t i = 0; i < in.size(); ++i) in[i] = (uint32_t)(i * 2654435761u >> 7);
  CK(cudaMemcpy(din, in.data(), in.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  ids_kernel<<<grid, block>>>(d);
  warp_slots_kernel<<<grid, block>>>(din, dsum);
  CK(cudaGetLastError());
  std::vector<uint32_t> got(4ull * n * nb), sums(nb);
  CK(cudaMemcpy(got.data(), d, got.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sums.data(), dsum, sums.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (uint32_t b = 0; b < nb; ++b) {
    uint32_t want_sum = 0;
    for (uint32_t t = 0; t < n; ++t) {
      const uint32_t* o = &got[4 * ((size_t)b * n + t)];
      const uint32_t lane = t & 31, warp = t >> 5;
      const uint32_t want[4] = {lane, warp, warp * 32 + ((lane + 5) & 31), 0x22222222u};
      for (int i = 0; i < 4; ++i)
        if (o[i] != want[i] && bad++ < 5)
          printf("  block (%u,%u,%u) block %u thread %u: field %d = %u, expected %u\n", block.x, block.y, block.z, b, t, i, o[i], want[i]);
      want_sum += in[(size_t)b * n + t];
    }
    if (sums[b] != want_sum && bad++ < 5)
      printf("  block (%u,%u,%u) block %u: per-warp slot sum %u, expected %u\n", block.x, block.y, block.z, b, sums[b], want_sum);
  }
  printf("  grid (%u,%u,%u) block (%u,%u,%u): %s\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, bad ? "FAIL" : "ok");
  if (bad) ++failures;
  CK(cudaFree(d)); CK(cudaFree(din)); CK(cudaFree(dsum));
}

int main() {
  // Block sizes are multiples of 32: the pipelines assume whole simdgroups (threadGroupSizeIsMultipleOfThreadExecutionWidth).
  check_shape(dim3(3), dim3(32));
  check_shape(dim3(2), dim3(256));
  check_shape(dim3(2, 2), dim3(8, 4, 3));
  check_shape(dim3(1, 2, 2), dim3(16, 2, 2));
  check_shape(dim3(2), dim3(4, 8, 8));
  check_shape(dim3(1), dim3(1, 32, 32));
  printf(failures ? "FAIL\n" : "PASS\n");
  return failures ? 1 : 0;
}
