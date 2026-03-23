import tile.language as tl
import torch

@ascend_kernel
def gelu_kernel(input_ptr, output_ptr,
                elements_per_core, tile_size, inner_loops):
    
    pid = tl.program_id(0)
    start = pid * elements_per_core
    
    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    x_ub          = tl.alloc_ub(tile_size, dtype=tl.float32)
    scaled_ub     = tl.alloc_ub(tile_size, dtype=tl.float32)
    erf_ub        = tl.alloc_ub(tile_size, dtype=tl.float32)
    one_plus_erf_ub = tl.alloc_ub(tile_size, dtype=tl.float32)
    half_x_ub     = tl.alloc_ub(tile_size, dtype=tl.float32)
    out_ub        = tl.alloc_ub(tile_size, dtype=tl.float32)
    
    # Constants
    inv_sqrt2 = 0.7071067811865475  # 1 / sqrt(2)
    
    # ------------------------------------------------------------
    # Tile loop
    # ------------------------------------------------------------
    for i in range(inner_loops):
        tile_start = start + i * tile_size
        offsets = tile_start + tl.arange(0, tile_size)
        
        # --------------------------------------------------------
        # COPYIN
        # --------------------------------------------------------
        with tl.copyin():
            tl.load(input_ptr + offsets, x_ub)
        
        # --------------------------------------------------------
        # COMPUTE
        # --------------------------------------------------------
        with tl.compute():
            # scaled = x * inv_sqrt2
            tl.vmul_scalar(scaled_ub, x_ub, inv_sqrt2)
            
            # erf_val = erf(scaled)
            tl.verf(erf_ub, scaled_ub)
            
            # one_plus_erf = 1 + erf_val
            tl.vadd_scalar(one_plus_erf_ub, erf_ub, 1.0)
            
            # half_x = 0.5 * x
            tl.vmul_scalar(half_x_ub, x_ub, 0.5)
            
            # out = half_x * one_plus_erf
            tl.vmul(out_ub, half_x_ub, one_plus_erf_ub)
        
        # --------------------------------------------------------
        # COPYOUT
        # --------------------------------------------------------
        with tl.copyout():
            tl.store(output_ptr + offsets, out_ub)


def gelu_host(x: torch.Tensor, output: torch.Tensor):
    total_elems = x.numel()
    
    # ------------------------------------------------------------
    # Core Partitioning
    # ------------------------------------------------------------
    n_cores = 16
    elements_per_core = total_elems // n_cores
    
    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    tile_size = 2048
    inner_loops = elements_per_core // tile_size
    
    # ------------------------------------------------------------
    # Launch kernel
    # ------------------------------------------------------------
    gelu_kernel[n_cores](
        x, output,
        elements_per_core,
        tile_size,
        inner_loops
    )