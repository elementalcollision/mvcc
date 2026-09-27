# Fork roadmap (elementalcollision/mvcc)

This fork is where we fold what we learned running Metal kernels at scale in TEMPO's lane-parallel co-simulation
("Lanes") back into mvcc, and where we stage contributions for upstream (doximity/mvcc). Items are processed in
order; each lands as its own commit (and, for upstream, its own pull request) with the verification listed next to
it. This file is fork-local and is not meant to go upstream.

## Working rules carried over from Lanes

- **Same results, faster.** A speed-up must leave results identical to the exact path. For integer and layout work
  that means bit-identical; for floating point it means no worse than the existing CPU-reference tolerances in
  `tests/`. Anything else is a separate, explicitly flagged change.
- **Two GPU generations.** Kernel-facing changes are validated on both the M5 Max and the M4 Pro (Mac mini). Lanes
  found a shader-compiler miscompile that only the M4 Pro showed; the M5-only numbers in the README would not have
  caught it.
- **Back-to-back A/B.** Run-to-run drift was about 5% in Lanes, so performance claims come from alternating A/B runs
  on the same machine in the same session, never from comparing against an old number.
- **Adversarial review before merge.** Each change gets a review pass that tries to break it (a wrong-answer case,
  a launch shape outside what was proven, a stale cache) before it is pushed.
- **Provenance.** Anything cached (pipelines, libraries, gem builds) is keyed by the content and toolchain that
  produced it and written atomically, so a stale artifact can never be used silently.

## How to verify on Linux (no GPU)

The compiler half of mvcc builds and runs on Linux. That covers `mvcc-ir2msl`, the pass plugin and the `nvcc`
driver's device path, which produces MSL, the kernel ABI and the SIG/STIR dumps. This is how the compiler-side items
below are checked before they reach a Mac:

1. Take LLVM 21 from the official release tarball (`LLVM-21.1.8-Linux-X64.tar.xz`); apt.llvm.org is not needed.
   Install `libzstd-dev` and `libxml2-dev`.
2. Configure `cpp/mvcc-llvm` with that LLVM's `clang`/`clang++`, and link with its `lld`. The release's static
   libraries are ThinLTO bitcode, so pass `-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld` and
   `-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld`.
3. Run `cargo build -p mvcc-driver --release`. The runtime crate is macOS-only.
4. Make a toolkit directory holding `bin/nvcc -> mvcc`, `bin/mvcc-ir2msl`, `lib64/libmvcc-passes.dylib`, `include`
   and `share/mvcc/mvcc_prelude.metal`. Then compile with `MVCC_ROOT=<toolkit> MVCC_CLANG=<llvm>/bin/clang++ MVCC_KEEP=1`.

Every kernel in `tests/kernels` and `examples/` compiles this way, except `nvptx_oracle_gemm_host.cu`, which is built
with NVIDIA's nvcc. `--engine-self-test` passes.

The checked-in SIG goldens (`tests/stir/*.golden`) do not match a Linux build. They carry LLVM value numbering and
fitted-site counts from the Mac's Homebrew clang. On Linux the check is therefore differential: the MSL, ABI and SIG
output of a change is compared against the same build of the parent commit.

## Status

| #  | Item                                                                        | Kind        | Status                             |
| -- | --------------------------------------------------------------------------- | ----------- | ---------------------------------- |
| 1  | Derive lane / warp ids from the thread position (M4 Pro miscompile)          | correctness | done; needs Mac validation         |
| 2  | Linux compiler CI (build + differential MSL/SIG check)                       | foundation  | next                               |
| 3  | Staging buffers leak into the residency set; `cudaHostRegister` alignment    | correctness |                                    |
| 4  | Launches whose block is not a multiple of 32 run a pipeline that assumes it  | correctness |                                    |
| 5  | Named / partial / predicated barriers lowered as full barriers               | correctness |                                    |
| 6  | `__mvcc_memcpy` 16-byte vector access on 4-byte-aligned pointers             | correctness |                                    |
| 7  | `__syncthreads` fences device memory even when nothing needs it              | performance |                                    |
| 8  | Wide-vector legalization splits by lanes, not bytes (`<16 x i8>` → 4 loads)  | performance |                                    |
| 9  | Tensor-op recovery on GPUs without Neural Accelerators (M1–M4)               | performance |                                    |
| 10 | Library + pipeline cache with provenance; compile off the global lock        | performance |                                    |
| 11 | Stream-ordered `cudaMallocAsync` / `cudaFreeAsync`                           | performance |                                    |
| 12 | Device `printf` through a readout buffer                                     | capability  |                                    |
| 13 | Lanes' netlist kernel as CUDA, built by mvcc and by nvcc                     | capability  |                                    |
| 14 | Device properties per GPU family                                             | capability  |                                    |
| 15 | Explicit graph API, occupancy/3D-copy gaps                                   | capability  |                                    |
| 16 | Hygiene: dead code and stale docs                                            | cleanup     |                                    |

## Items

### 1. Derive lane / warp ids from the thread position — done, needs Mac validation

The problem:
- Every kernel declared `[[thread_index_in_simdgroup]]` and `[[simdgroup_index_in_threadgroup]]`.
- `%warpid` read the second of these.
- The recovered tensor-op kernels also used it to find each simdgroup's conversion scratch
  (`msl_emitter.cpp`, `__mvcc_tp_acc2ptx`/`ptx2acc`).
- In Lanes, the M4 Pro's shader compiler deterministically miscompiled a kernel that read
  `[[simdgroup_index_in_threadgroup]]`. The M5 Max was fine with the same source.
- Deriving the index from the thread position fixed it.

The change:
- mvcc now computes `__lane = linear & 31` and `__sgid = linear >> 5` from `[[thread_position_in_threadgroup]]`.
- It declares them only in kernels that use them.
- `MVCC_SIMD_ATTRS=1` restores the attributes for A/B comparison.

Verified on Linux:
- Every test kernel and example compiles.
- The MSL diff is exactly the signature and prologue change.
- SIG/STIR dumps are byte-identical to the parent commit.
- `MVCC_SIMD_ATTRS=1` reproduces the parent's kernels.

To do on the Macs:
- Run `tests/run.sh` on the M5 Max and the M4 Pro. It now includes `tests/kernels/simd_ids.cu`, which checks
  lane/warp ids, shuffles, ballots and warp-indexed shared memory across 1-D, 2-D and 3-D blocks.
- On the M4 Pro, also run `gemm_ptx` and `stir_ops` with `MVCC_SIMD_ATTRS=1`. That shows whether mvcc's recovered
  kernels hit the miscompile, which is worth reporting upstream either way.
- Run an A/B of `gemm_ptx --bench` and qwen decode with and without the change. Expect no difference.

### 2. Linux compiler CI

What to build:
- A GitHub Actions job on `ubuntu-24.04` that caches the LLVM 21 tarball.
- It builds `mvcc-ir2msl`, the plugin and the driver as described above.
- It compiles `tests/kernels` and `examples`.
- It runs `--engine-self-test`, the `--stir-in` / `--sched-in` / `--sched-diff` checks from `tests/run.sh`,
  `gen_diagnostics.py --check` and `check_provenance.sh`.
- It compares MSL/ABI/SIG output against the merge base, and posts the diff as a job summary. An intended change
  shows up as a readable diff rather than a golden to regenerate.

Portability fixes this needs:
- The plugin links with the macOS-only `-undefined dynamic_lookup` and is forced to a `.dylib` suffix.
- `run.sh` runs `cargo test --workspace`, which cannot build `mvcc-cudart` on Linux.
- Already fixed in this branch: `msl_emitter.cpp` relied on a transitive `ValueTracking.h` include that stock
  LLVM 21 does not provide.

### 3. Staging buffers in the residency set; `cudaHostRegister` alignment

Staging buffers:
- Pageable host-to-device copies on a busy stream, and graph `HostIn`/`HostOut` nodes, allocate a staging buffer.
- They add it to the residency set and commit it: `runtime.rs` `memcpy` and `graph_launch`.
- Nothing ever removes these buffers. Only `memory.rs` calls `res.remove`.
- The set keeps growing, and it most likely keeps every staging buffer alive.
- Blit arguments are made resident by the encoder, so the fix is to not add them. Reusing staging buffers from a
  small pool also removes an allocation and a residency commit per copy.

`cudaHostRegister`:
- It rounds the range to 4 KiB pages (`memory.rs` `register_host`).
- `newBufferWithBytesNoCopy` needs 16 KiB on Apple silicon (`mvcc_metal.mm` `mvccm_buffer_wrap_host`), so most
  registrations fail.
- `cudaHostRegister` still returns success (`api.rs`), and the copies then take the slow path.
- The fix is to round to 16 KiB and report the failure.

Verify with a leak test (RSS and `currentAllocatedSize` flat over 10k busy-stream copies) on both Macs.

### 4. Launches whose block is not a multiple of 32

The problem:
- Every pipeline is created with `threadGroupSizeIsMultipleOfThreadExecutionWidth = YES` (`mvcc_metal.mm`).
- Metal defines the results as undefined when the threadgroup is not a multiple of 32.
- CUDA programs launch `<<<1, 1>>>` and odd block sizes routinely, and so do the tests (`single_thread_order.cu`,
  `device_graph_tail.cu`).

The fix:
- Create a second pipeline without the flag, and use it for launches whose block is not a multiple of 32.
- Key the pipeline cache on that choice.

Verify with odd-block tests (1, 33, 48 and 100 threads) that use shuffles, ballots and reductions, on both Macs.

### 5. Named, partial and predicated barriers

The problem:
- `__syncthreads` lowering matches every `llvm.nvvm.barrier*` and `bar.sync` by prefix.
- So `bar.sync id, n`, `barrier.cta.*.count` and the arrive forms become full-block barriers. Code that syncs
  only a subset of warps then deadlocks.
- Inline-asm `@p bar.sync` also drops its predicate.

The fix is to lower the count forms correctly where Metal can express them. Otherwise fail at build time with a
message that names the construct; a hang is a bug, a clear error is a limitation.

### 6. `__mvcc_memcpy` vector width

The problem:
- The prelude copies with `uint4` once both pointers are 4-byte aligned.
- The emitter's own notes say Metal rounds under-aligned addresses down (`msl_emitter.cpp`, `CudaAlignmentPass`).
- If that holds for 16-byte vectors, a copy between 4-aligned addresses that are not 16-aligned reads the wrong bytes.

The fix is `packed_uint4`, which needs only element alignment. Add a device test that copies at offsets 4, 8 and 12.

### 7. Barrier strength

The problem:
- `__syncthreads` becomes `threadgroup_barrier(mem_threadgroup | mem_device)` in every kernel.
- `__syncwarp` and LLVM `fence` (always seq_cst, device scope, ignoring `syncscope`) are similar.
- The device fence only matters when a device-memory write before the barrier can be read by another thread of the
  block after it.

The change:
- Per barrier, use `mem_threadgroup` alone when no device store or atomic can reach it, or when no device load or
  atomic is reachable from it.
- Treat generic pointers as device.
- Keep `__MVCC_SMEM_FLAGS` so the spill build still fences its device-memory shared memory.
- Honour `syncscope` on fences.

Verification:
- Compile-side: the MSL diff shows only barrier flags changing.
- On the Macs: the full suite, plus an A/B of shared-memory-tiled kernels (exact `gemm_ptx` twins, reductions, qwen).

### 8. Wide-vector legalization by bytes

The problem:
- `LegalizeWideVectorsPass` splits loads and stores of more than 4 lanes into 4-lane pieces whatever the element
  size.
- So `<16 x i8>` becomes four `uchar4` accesses and `<8 x half>` two `half4`, where one 128-bit access would do.
- INT4/INT8 decode kernels are bandwidth-bound on exactly these loads.

The change is to split into ≤128-bit pieces, and reinterpret through `uint4`/`as_type`.

Verification:
- Compile-side: MSL diff.
- On the Macs: an A/B of qwen decode and the roofline read, with bit-identical outputs.

### 9. Tensor-op recovery on M1–M4

The problem:
- The only runtime gate for recovered kernels is `MTLGPUFamilyMetal4` plus the layout check.
- Nothing distinguishes the M5's Neural Accelerators from older Metal 4 GPUs, which lack them.
- Every published number is from an M5 Pro.

The change:
- Measure recovered and exact kernels on the M4 Pro.
- If recovered kernels lose there, choose per GPU family: a runtime policy with `MVCC_TENSOR_EXACT` kept as the
  override.

Also extend the verify kernel to run at least two simdgroups. Today it dispatches one, so it cannot catch a
per-simdgroup scratch error like the one item 1 guards against.

### 10. Library and pipeline cache with provenance

The problem:
- Only pipelines are archived (`~/Library/Caches/mvcc/<fnv64>.metalar`). The MSL-to-library compile runs in every
  process.
- The archive is marked dirty and re-serialized after every pipeline creation, even when the pipeline came from the
  archive. It is flushed only from `__cudaUnregisterFatBinary`.
- The key does not include the OS build or the GPU family.
- All compilation happens synchronously under the global runtime mutex.

The change:
- Cache the compiled library, keyed by SHA-256 of the MSL, the macOS build and the GPU family.
- Write atomically.
- Serialize only when something new was compiled.
- Compile a module's pipelines on a background queue when the module registers.

The Ruby gem cache has related problems:
- Its key omits the per-call `flags:` and the toolkit version.
- It writes the `.so` in place.

### 11. Stream-ordered allocation

`cudaMallocAsync` is `cudaMalloc`, and `cudaFreeAsync` is `cudaFree`, which synchronizes the device when any stream
is busy. A free list released at the stream's fence value turns that into a no-wait operation. Frameworks that
allocate per step depend on this.

### 12. Device `printf`

Device `printf` compiles to `return 0`. Lanes classified results on the GPU and read back one vector instead of
shipping state to the host. The same shape works here:
- `printf` appends its format id and arguments to a per-launch ring buffer (an atomic cursor in a buffer bound like
  the zero page).
- The host formats the records after the command buffer completes, in stream order.

### 13. Lanes' netlist kernel as CUDA

Lanes' Metal netlist kernel cannot run on RunPod's NVIDIA GPUs, and a CUDA port was already planned. Writing it
once in CUDA and building it with mvcc on the Macs and with nvcc on RunPod gives one source for both. Lanes'
bit-identical-to-the-ISS requirement also makes it a demanding end-to-end test for mvcc. Its kernel shape (16
simdgroups per word, `uint4` state quads, thousands of lanes) exercises items 1, 7 and 8 directly.

### 14. Device properties per family

`clockRate`, `l2CacheSize`, `memoryClockRate` and `memoryBusWidth` are hard-coded; the last two are 0, so any
bandwidth estimate a program computes is 0. The shared-memory values also disagree:
- `cudaDevAttrMaxSharedMemoryPerBlock` deliberately returns the 32 KB threadgroup limit (`api.rs`), so that
  launchers pick tiles that fit.
- `cudaGetDeviceProperties` still reports `sharedMemPerBlock` = 49152 (`mvcc_props.cpp`).

The fix is a per-family table (M1–M5, base/Pro/Max) and one shared-memory policy for both paths.

### 15. API gaps

Missing:
- The explicit graph API: `cudaGraphAddKernelNode` and friends return not-supported, so graphs work only through
  stream capture.
- `cudaOccupancyMaxPotentialBlockSize*`.
- `cudaMalloc3D` / `cudaMemcpy3D`.
- `cuDeviceGet` for ordinals above 0 when `MVCC_DEVICES` > 1.

Driver bug: in link mode, two inputs with the same file name (`a/k.cu`, `b/k.cu`) overwrite each other's object
files.

### 16. Hygiene

- `staging_elim.cpp` (3,255 lines, 40% of the tensor-recovery line budget) has no caller: nothing calls
  `emitFromStir`.
- `FmaDotPackPass` and `WarpButterflyReducePass` are never scheduled.
- `MVCC_TP_GROUPS` is documented but never read.
- The `search_variant` ABI field is not consumed by the runtime.
- `sha2` is an unused driver dependency.
- The README says `mvcc-ir2msl` links Z3, which it does not do directly.

Each of these needs a keep-or-remove decision with upstream rather than a unilateral deletion.
