

"""
    godunov_upwind_3d(i, j, k, T_src, thick_r_sq, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2, base_h2)

Device-compatible 3D quadratic Godunov upwind Eikonal stencil for Hele-Shaw flow
with speed `F = 4 * thick_r_sq * Δx^2`.
"""
@inline function godunov_upwind_3d(
    i::Int, j::Int, k::Int,
    T_src, thick_r_sq,
    nx::Int, ny::Int, nz::Int,
    inv_dx2::Float32, inv_dy2::Float32, inv_dz2::Float32,
    base_h2::Float32
)
    @inbounds begin
        rsq = Float32(thick_r_sq[i, j, k])
        rsq <= 0.0f0 && return Inf32
        inv_F2 = 1.0f0 / ((base_h2 * rsq)^2)

        tx = min(
            i > 1  ? T_src[i - 1, j, k] : Inf32,
            i < nx ? T_src[i + 1, j, k] : Inf32
        )
        ty = min(
            j > 1  ? T_src[i, j - 1, k] : Inf32,
            j < ny ? T_src[i, j + 1, k] : Inf32
        )
        tz = min(
            k > 1  ? T_src[i, j, k - 1] : Inf32,
            k < nz ? T_src[i, j, k + 1] : Inf32
        )

        # Sort directional upwind minima: t1 <= t2 <= t3 with corresponding 1/h^2 weights
        t1, w1 = tx, inv_dx2
        t2, w2 = ty, inv_dy2
        t3, w3 = tz, inv_dz2

        if t1 > t2; t1, t2 = t2, t1; w1, w2 = w2, w1; end
        if t2 > t3; t2, t3 = t3, t2; w2, w3 = w3, w2; end
        if t1 > t2; t1, t2 = t2, t1; w1, w2 = w2, w1; end

        isfinite(t1) || return Inf32

        # 1D update
        t_sol = t1 + sqrt(inv_F2 / w1)
        t_sol <= t2 && return t_sol

        # 2D quadratic update: w1*(t - t1)^2 + w2*(t - t2)^2 = inv_F2
        a2 = w1 + w2
        b2 = -2.0f0 * (w1 * t1 + w2 * t2)
        c2 = w1 * t1 * t1 + w2 * t2 * t2 - inv_F2
        disc2 = b2 * b2 - 4.0f0 * a2 * c2
        if disc2 >= 0.0f0
            t_sol2 = (-b2 + sqrt(disc2)) / (2.0f0 * a2)
            t_sol2 <= t3 && return t_sol2
        end

        # 3D quadratic update: w1*(t - t1)^2 + w2*(t - t2)^2 + w3*(t - t3)^2 = inv_F2
        a3 = a2 + w3
        b3 = b2 - 2.0f0 * w3 * t3
        c3 = c2 + w3 * t3 * t3
        disc3 = b3 * b3 - 4.0f0 * a3 * c3
        if disc3 >= 0.0f0
            return (-b3 + sqrt(disc3)) / (2.0f0 * a3)
        end
        return t_sol
    end
end

# Kernel 1: Update all currently active voxels and retain unconverged ones
@kernel function fim_update_kernel!(
    T_next, active_next, changed_flag,
    @Const(T_cur), @Const(active_cur), @Const(voxel_grid), @Const(thick_r_sq),
    inv_dx2::Float32, inv_dy2::Float32, inv_dz2::Float32, base_h2::Float32, tol::Float32
)
    i, j, k = @index(Global, NTuple)
    nx, ny, nz = size(voxel_grid)

    @inbounds begin
        if voxel_grid[i, j, k] && active_cur[i, j, k] != 0x00
            t_old = T_cur[i, j, k]
            t_new = godunov_upwind_3d(
                i, j, k, T_cur, thick_r_sq,
                nx, ny, nz, inv_dx2, inv_dy2, inv_dz2, base_h2
            )

            if t_new < t_old
                T_next[i, j, k] = t_new
            else
                t_new = t_old
            end

            if abs(t_old - t_new) > tol
                active_next[i, j, k] = 0x01
                changed_flag[1] = 0x01
            end
        end
    end
end

# Kernel 2: Activate neighboring solid voxels adjacent to the updated front
@kernel function fim_activate_neighbors_kernel!(
    T_cur, active_next, changed_flag,
    @Const(T_next), @Const(active_cur), @Const(voxel_grid), @Const(thick_r_sq),
    inv_dx2::Float32, inv_dy2::Float32, inv_dz2::Float32, base_h2::Float32, tol::Float32
)
    i, j, k = @index(Global, NTuple)
    nx, ny, nz = size(voxel_grid)

    @inbounds begin
        if voxel_grid[i, j, k]
            # Sync T_cur with T_next for all solid voxels on this pass
            t_self = T_next[i, j, k]
            T_cur[i, j, k] = t_self

            if active_next[i, j, k] == 0x00
                has_active_nbr =
                    (i > 1  && active_cur[i - 1, j, k] != 0x00) ||
                    (i < nx && active_cur[i + 1, j, k] != 0x00) ||
                    (j > 1  && active_cur[i, j - 1, k] != 0x00) ||
                    (j < ny && active_cur[i, j + 1, k] != 0x00) ||
                    (k > 1  && active_cur[i, j, k - 1] != 0x00) ||
                    (k < nz && active_cur[i, j, k + 1] != 0x00)

                if has_active_nbr
                    t_cand = godunov_upwind_3d(
                        i, j, k, T_next, thick_r_sq,
                        nx, ny, nz, inv_dx2, inv_dy2, inv_dz2, base_h2
                    )
                    if t_cand + tol < t_self
                        T_cur[i, j, k] = t_cand
                        active_next[i, j, k] = 0x01
                        changed_flag[1] = 0x01
                    end
                end
            end
        end
    end
end

"""
    solve_eikonal_fim_ka(
        voxel_grid::AbstractArray{Bool, 3},
        thick_r_sq::AbstractArray{UInt16, 3},
        gate_voxels::AbstractVector{CartesianIndex{3}},
        voxel_spacing::NTuple{3, Real};
        backend = KernelAbstractions.get_backend(thick_r_sq),
        tol::Float32 = 1f-5,
        max_iters::Int = 10_000,
        workgroupsize = (8, 8, 4)
    )

Portable `KernelAbstractions.jl` Fast Iterative Method (FIM) Eikonal solver.
Pass `backend = CPU()` for multi-core CPU or `CUDABackend()` / `MetalBackend()` / `ROCBackend()`
for GPU execution (automatically inferred if `thick_r_sq` is already a GPU array).
"""
function solve_eikonal_fim_ka(
    voxel_grid::AbstractArray{Bool, 3},
    thick_r_sq::AbstractArray{UInt16, 3},
    gate_voxels::AbstractVector{CartesianIndex{3}},
    voxel_spacing::NTuple{3, Real};
    backend = KernelAbstractions.get_backend(thick_r_sq),
    tol::Float32 = 1f-5,
    max_iters::Int = 10_000,
    workgroupsize = (8, 8, 4)
)
    
    dims = size(voxel_grid)
    nx, ny, nz = dims

    dx = Float32(voxel_spacing[1])
    dy = Float32(voxel_spacing[2])
    dz = Float32(voxel_spacing[3])
    inv_dx2 = 1.0f0 / (dx * dx)
    inv_dy2 = 1.0f0 / (dy * dy)
    inv_dz2 = 1.0f0 / (dz * dz)
    dx_mean = (dx + dy + dz) / 3.0f0
    base_h2 = 4.0f0 * dx_mean * dx_mean

    # Host initialization of arrival times and initial narrow-band active mask
    T_host = fill(Inf32, dims)
    act_host = zeros(UInt8, dims)
    voxel_host = Array(voxel_grid)

    offsets = (
        CartesianIndex(-1, 0, 0), CartesianIndex(1, 0, 0),
        CartesianIndex(0, -1, 0), CartesianIndex(0, 1, 0),
        CartesianIndex(0, 0, -1), CartesianIndex(0, 0, 1)
    )

    for g in gate_voxels
        (1 <= g[1] <= nx && 1 <= g[2] <= ny && 1 <= g[3] <= nz) || continue
        voxel_host[g] || continue
        T_host[g] = 0.0f0
        for off in offsets
            nb = g + off
            (1 <= nb[1] <= nx && 1 <= nb[2] <= ny && 1 <= nb[3] <= nz) || continue
            if voxel_host[nb] && T_host[nb] > 0.0f0
                act_host[nb] = 0x01
            end
        end
    end

    # Allocate/adapt buffers on target backend (no-op if already on target device)
    d_voxel = adapt(backend, voxel_grid)
    d_rsq   = adapt(backend, thick_r_sq)
    T_cur   = adapt(backend, T_host)
    T_next  = adapt(backend, copy(T_host))
    act_cur = adapt(backend, act_host)
    act_nxt = KernelAbstractions.zeros(backend, UInt8, dims)
    changed = KernelAbstractions.zeros(backend, UInt8, 1)

    update_kern!   = fim_update_kernel!(backend, workgroupsize)
    activate_kern! = fim_activate_neighbors_kernel!(backend, workgroupsize)

    for iter in 1:max_iters
        fill!(act_nxt, 0x00)
        fill!(changed, 0x00)

        update_kern!(
            T_next, act_nxt, changed,
            T_cur, act_cur, d_voxel, d_rsq,
            inv_dx2, inv_dy2, inv_dz2, base_h2, tol;
            ndrange = dims
        )

        activate_kern!(
            T_cur, act_nxt, changed,
            T_next, act_cur, d_voxel, d_rsq,
            inv_dx2, inv_dy2, inv_dz2, base_h2, tol;
            ndrange = dims
        )

        KernelAbstractions.synchronize(backend)

        any(!iszero, Array(changed)) || break

        # Keep T_next consistent with any newly activated t_cand values in T_cur
        copyto!(T_next, T_cur)
        act_cur, act_nxt = act_nxt, act_cur
    end

    return T_cur
end


function solve_eikonal_fim_ka(
    voxeldata::VoxelData,
    gates::AbstractVector{ValveGate};
    kwargs...
)

    voxel_grid = voxeldata.grid
    thick_r_sq = voxeldata.thickness
    voxel_spacing = voxeldata.voxel_spacing
    gate_voxels = gates_to_voxels(gates, voxeldata)

    return solve_eikonal_fim_ka(voxel_grid, thick_r_sq, gate_voxels, voxel_spacing; kwargs...)
end
