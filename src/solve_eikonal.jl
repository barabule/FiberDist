



"""
    cartesian_upwind_3d(T_arr, i, j, k, F, dx, dy, dz)

Solves the exact 3D Cartesian upwind quadratic equation:
  max((T - Tx)/dx, 0)^2 + max((T - Ty)/dy, 0)^2 + max((T - Tz)/dz, 0)^2 = 1 / F^2
"""
function cartesian_upwind_3d(
    T_arr::Array{T, 3},
    i::Int, j::Int, k::Int,
    F::T,
    dx::T, dy::T, dz::T
) where {T<:AbstractFloat}
    nx, ny, nz = size(T_arr)
    inf_val = T(Inf)

    tx = min(i > 1 ? T_arr[i-1, j, k] : inf_val, i < nx ? T_arr[i+1, j, k] : inf_val)
    ty = min(j > 1 ? T_arr[i, j-1, k] : inf_val, j < ny ? T_arr[i, j+1, k] : inf_val)
    tz = min(k > 1 ? T_arr[i, j, k-1] : inf_val, k < nz ? T_arr[i, j, k+1] : inf_val)

    # Sort active upwind dimensions by arrival time: t1 <= t2 <= t3
    d1 = (tx, dx)
    d2 = (ty, dy)
    d3 = (tz, dz)
    if d1[1] > d2[1]; d1, d2 = d2, d1; end
    if d2[1] > d3[1]; d2, d3 = d3, d2; end
    if d1[1] > d2[1]; d1, d2 = d2, d1; end

    invF = inv(F)
    t1, h1 = d1
    !isfinite(t1) && return inf_val

    # 1D update
    t_cand = t1 + h1 * invF

    # 2D quadratic update
    t2, h2 = d2
    if isfinite(t2) && t2 < t_cand
        a = inv(h1^2) + inv(h2^2)
        b = -T(2) * (t1 / h1^2 + t2 / h2^2)
        c = (t1^2 / h1^2) + (t2^2 / h2^2) - invF^2
        disc = b^2 - T(4) * a * c
        if disc >= zero(T)
            t_cand = (-b + sqrt(disc)) / (T(2) * a)
        end
    end

    # 3D quadratic update
    t3, h3 = d3
    if isfinite(t3) && t3 < t_cand
        a = inv(h1^2) + inv(h2^2) + inv(h3^2)
        b = -T(2) * (t1 / h1^2 + t2 / h2^2 + t3 / h3^2)
        c = (t1^2 / h1^2) + (t2^2 / h2^2) + (t3^2 / h3^2) - invF^2
        disc = b^2 - T(4) * a * c
        if disc >= zero(T)
            t_cand = (-b + sqrt(disc)) / (T(2) * a)
        end
    end

    return t_cand
end

"""
    solve_eikonal_voxel(
    voxeldata::VoxelData{T},
    tfield::ThicknessField,
    gates::AbstractVector{<:ValveGate}; 
    min_thickness::Real = 0.1, #
    shift::Integer = 0
) where {T<:Real}

Solves the 3D Eikonal equation |∇T| = 1/F directly on interior solid voxels using
the `UInt16` squared-radius thickness volume `thick_r_sq`.

Computes Hele-Shaw speed on the fly without `sqrt` or allocating a Float thickness array:
    h² = 4 * (thick_r_sq << shift) * dx_mean²
    F  = max(h², min_thickness²)
"""
function solve_eikonal_voxel(
    voxeldata::VoxelData{T},
    tfield::ThicknessField,
    gates::AbstractVector{<:ValveGate}; 
    min_thickness::Real = 0.1, #
    shift::Integer = 0
) where {T<:Real}


    voxel_grid, voxel_spacing, corner = voxeldata.grid, voxeldata.voxel_spacing, voxeldata.corner
    thick_r_sq = tfield.thick_r_sq

    @assert size(voxel_grid) == size(thick_r_sq) "voxel_grid and thick_r_sq must have identical dimensions"

    # Infer working floating-point precision from inputs
    T_gate = isempty(gates) ? Float32 : eltype(gates[1].open_time)
    

    nx, ny, nz = size(voxel_grid)
    dx, dy, dz = T(voxel_spacing[1]), T(voxel_spacing[2]), T(voxel_spacing[3])
    ox, oy, oz = T(corner[1]), T(corner[2]), T(corner[3])

    # Precompute constant physical scaling factor: 4 * 2^shift * dx_mean^2
    dx_mean = (dx + dy + dz) / T(3)
    speed_scale = T(4) * T(1 << shift) * (dx_mean * dx_mean)
    min_F = T(min_thickness)^2

    fill_time = fill(T(Inf), nx, ny, nz)
    accepted = falses(nx, ny, nz)
    pq = PriorityQueue{CartesianIndex{3}, T}()

    
    

    # Seed gates (supports cascaded opening times t_open >= 0)
    for gate in gates
        t_open = T(gate.open_time)
        @assert t_open >= zero(T) "Gate opening time must be >= 0"
        idx = find_nearest_interior_voxel(gate.position, voxeldata)
        if t_open < fill_time[idx]
            fill_time[idx] = t_open
            pq[idx] = t_open
        end
    end

    offsets = (
        CartesianIndex(-1, 0, 0), CartesianIndex(1, 0, 0),
        CartesianIndex(0, -1, 0), CartesianIndex(0, 1, 0),
        CartesianIndex(0, 0, -1), CartesianIndex(0, 0, 1)
    )

    @inbounds while !isempty(pq)
        curr, _ = dequeue_pair!(pq)
        accepted[curr] && continue
        accepted[curr] = true

        for off in offsets
            nbr = curr + off
            i, j, k = Tuple(nbr)
            (1 <= i <= nx && 1 <= j <= ny && 1 <= k <= nz) || continue
            (voxel_grid[i, j, k] && !accepted[i, j, k]) || continue

            # On-the-fly Hele-Shaw speed directly from UInt16 squared voxel radius
            F_local = max(T(thick_r_sq[i, j, k]) * speed_scale, min_F)
            t_cand = cartesian_upwind_3d(fill_time, i, j, k, F_local, dx, dy, dz)

            if t_cand < fill_time[i, j, k]
                fill_time[i, j, k] = t_cand
                pq[nbr] = t_cand
            end
        end
    end

    return fill_time
end





"""
    analyze_eikonal_field(
        T_field, voxel_grid, thick_r_sq, voxel_spacing;
        r_sq = nothing,
        weld_dot_thresh = -0.3,
        meld_dot_thresh = 0.4,
        div_thresh = 0.05,
        hesitation_quantile = 0.15,
        core_exponent = 2.0,
        div_scale = 5.0,
        max_shear_tilt = 0.08
    )

Extracts sparse localized defect masks (`weld_lines`, `meld_lines`, `air_traps`) directly
as `SparseVoxelMask` (`Vector{Int32}` linear indices) without intermediate `BitArray`
allocations, dense flow-regime masks (`diverging_flow`, `converging_flow`, `hesitation`),
unit flow directions `û = ∇T / |∇T|`, and the full 6-component symmetric Fiber Orientation
Tensor `(a11, a22, a33, a12, a13, a23)`.
"""
function analyze_eikonal_field(
    T_field::AbstractArray{T1, 3},
    voxel_grid::AbstractArray{Bool, 3},
    thick_r_sq::AbstractArray{UInt16, 3},
    voxel_spacing::Union{NTuple{3, T2}, StaticVector{3, T2}};
    r_sq::Union{Nothing, AbstractArray{UInt16, 3}} = nothing,
    weld_dot_thresh::Real = -0.3,
    meld_dot_thresh::Real = 0.4,
    div_thresh::Real = 0.05,
    hesitation_quantile::Real = 0.15,
    core_exponent::Real = 2.0,
    div_scale::Real = 5.0,
    max_shear_tilt::Real = 0.08
) where {T1<:Real, T2<:Real}

    T = float(promote_type(T1, T2))
    dims = size(voxel_grid)
    nx, ny, nz = dims
    @assert prod(dims) <= typemax(Int32) "Grid voxel count exceeds Int32 indexing limit; use Int64 indices"

    dx, dy, dz = T(voxel_spacing[1]), T(voxel_spacing[2]), T(voxel_spacing[3])
    lin_inds = LinearIndices(dims)

    # 1. Compute interior-only 3D gradient ∇T and unit flow direction û
    flow_dirs = fill(zero(SVector{3, T}), nx, ny, nz)
    front_speed = zeros(T, nx, ny, nz)

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        (voxel_grid[i, j, k] && isfinite(T_field[i, j, k])) || continue
        tc = T(T_field[i, j, k])

        xm = i > 1  && voxel_grid[i-1, j, k] && isfinite(T_field[i-1, j, k])
        xp = i < nx && voxel_grid[i+1, j, k] && isfinite(T_field[i+1, j, k])
        gx = (xm && xp) ? (T(T_field[i+1, j, k]) - T(T_field[i-1, j, k])) / (T(2) * dx) :
             xp         ? (T(T_field[i+1, j, k]) - tc) / dx :
             xm         ? (tc - T(T_field[i-1, j, k])) / dx : zero(T)

        ym = j > 1  && voxel_grid[i, j-1, k] && isfinite(T_field[i, j-1, k])
        yp = j < ny && voxel_grid[i, j+1, k] && isfinite(T_field[i, j+1, k])
        gy = (ym && yp) ? (T(T_field[i, j+1, k]) - T(T_field[i, j-1, k])) / (T(2) * dy) :
             yp         ? (T(T_field[i, j+1, k]) - tc) / dy :
             ym         ? (tc - T(T_field[i, j-1, k])) / dy : zero(T)

        zm = k > 1  && voxel_grid[i, j, k-1] && isfinite(T_field[i, j, k-1])
        zp = k < nz && voxel_grid[i, j, k+1] && isfinite(T_field[i, j, k+1])
        gz = (zm && zp) ? (T(T_field[i, j, k+1]) - T(T_field[i, j, k-1])) / (T(2) * dz) :
             zp         ? (T(T_field[i, j, k+1]) - tc) / dz :
             zm         ? (tc - T(T_field[i, j, k-1])) / dz : zero(T)

        g = SVector{3, T}(gx, gy, gz)
        gn = norm(g)
        if gn > T(1e-9)
            flow_dirs[i, j, k] = g / gn
            front_speed[i, j, k] = inv(gn)
        end
    end

    valid_speeds = [front_speed[idx] for idx in eachindex(voxel_grid) if voxel_grid[idx] && front_speed[idx] > zero(T)]
    speed_cutoff = isempty(valid_speeds) ? zero(T) :
        sort(valid_speeds)[max(1, round(Int, T(hesitation_quantile) * length(valid_speeds)))]

    # 2. Direct sparse index buffers for localized defects (zero intermediate BitArrays)
    #    Iterating `for k in 1:nz, j in 1:ny, i in 1:nx` visits linear indices in strictly
    #    ascending order, so `nz_inds` are guaranteed sorted upon completion.
    weld_inds = Int32[]
    meld_inds = Int32[]
    trap_inds = Int32[]

    # Dense BitArray{3} masks for broad flow regimes
    diverging_flow  = falses(nx, ny, nz)
    converging_flow = falses(nx, ny, nz)
    hesitation      = falses(nx, ny, nz)

    # Full 6-component symmetric Fiber Orientation Tensor
    a11 = zeros(T, nx, ny, nz)
    a22 = zeros(T, nx, ny, nz)
    a33 = zeros(T, nx, ny, nz)
    a12 = zeros(T, nx, ny, nz)
    a13 = zeros(T, nx, ny, nz)
    a23 = zeros(T, nx, ny, nz)

    offsets = (
        CartesianIndex(-1, 0, 0), CartesianIndex(1, 0, 0),
        CartesianIndex(0, -1, 0), CartesianIndex(0, 1, 0),
        CartesianIndex(0, 0, -1), CartesianIndex(0, 0, 1)
    )

    w_thresh = T(weld_dot_thresh)
    m_thresh = T(meld_dot_thresh)
    d_thresh = T(div_thresh)
    p_exp    = T(core_exponent)
    d_scale  = T(div_scale)
    s_tilt   = T(max_shear_tilt)

    # 3. Single-pass extraction of sparse defect indices, dense masks, and 6-component FOT
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        (voxel_grid[i, j, k] && isfinite(T_field[i, j, k])) || continue

        lin_idx = Int32(lin_inds[i, j, k])
        tc = T(T_field[i, j, k])
        u_c = flow_dirs[i, j, k]
        has_flow = norm(u_c) > T(1e-6)

        min_dot = one(T)
        is_local_max = true
        has_valid_nbr = false

        for off in offsets
            ii, jj, kk = i + off[1], j + off[2], k + off[3]
            (1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz) || continue
            (voxel_grid[ii, jj, kk] && isfinite(T_field[ii, jj, kk])) || continue

            has_valid_nbr = true
            if T(T_field[ii, jj, kk]) > tc
                is_local_max = false
            end

            u_n = flow_dirs[ii, jj, kk]
            if has_flow && norm(u_n) > T(1e-6)
                d_un = dot(u_c, u_n)
                if d_un < min_dot
                    min_dot = d_un
                end
            end
        end

        if has_valid_nbr && is_local_max && tc > zero(T)
            push!(trap_inds, lin_idx)
        end

        is_weld = false
        if has_flow
            if min_dot < w_thresh
                push!(weld_inds, lin_idx)
                is_weld = true
            elseif min_dot < m_thresh
                push!(meld_inds, lin_idx)
            end
        end

        if front_speed[i, j, k] > zero(T) && front_speed[i, j, k] <= speed_cutoff
            hesitation[i, j, k] = true
        end

        # Flow Divergence ∇ ⋅ û
        dux_dx = (i > 1 && i < nx && voxel_grid[i-1,j,k] && voxel_grid[i+1,j,k]) ?
                 (flow_dirs[i+1,j,k][1] - flow_dirs[i-1,j,k][1]) / (T(2) * dx) : zero(T)
        duy_dy = (j > 1 && j < ny && voxel_grid[i,j-1,k] && voxel_grid[i,j+1,k]) ?
                 (flow_dirs[i,j+1,k][2] - flow_dirs[i,j-1,k][2]) / (T(2) * dy) : zero(T)
        duz_dz = (k > 1 && k < nz && voxel_grid[i,j,k-1] && voxel_grid[i,j,k+1]) ?
                 (flow_dirs[i,j,k+1][3] - flow_dirs[i,j,k-1][3]) / (T(2) * dz) : zero(T)

        div_u = dux_dx + duy_dy + duz_dz
        if div_u > d_thresh
            diverging_flow[i, j, k] = true
        elseif div_u < -d_thresh
            converging_flow[i, j, k] = true
        end

        if !has_flow
            a11[i, j, k] = T(1) / T(3)
            a22[i, j, k] = T(1) / T(3)
            a33[i, j, k] = T(1) / T(3)
            continue
        end

        # Local orthonormal frame (e1 = flow, e2 = in-plane transverse, e3 = wall normal)
        e1 = u_c
        if !isnothing(r_sq) && i > 1 && i < nx && j > 1 && j < ny && k > 1 && k < nz
            gn_x = T(Int32(r_sq[i+1,j,k]) - Int32(r_sq[i-1,j,k]))
            gn_y = T(Int32(r_sq[i,j+1,k]) - Int32(r_sq[i,j-1,k]))
            gn_z = T(Int32(r_sq[i,j,k+1]) - Int32(r_sq[i,j,k-1]))
            n_raw = SVector{3, T}(gn_x, gn_y, gn_z)
            n_ortho = n_raw - dot(n_raw, e1) * e1
            if norm(n_ortho) > T(1e-5)
                e3 = normalize(n_ortho)
            else
                ref = abs(e1[1]) < T(0.9) ? SVector{3, T}(1, 0, 0) : SVector{3, T}(0, 1, 0)
                e3 = normalize(cross(e1, ref))
            end
        else
            ref = abs(e1[1]) < T(0.9) ? SVector{3, T}(1, 0, 0) : SVector{3, T}(0, 1, 0)
            e3 = normalize(cross(e1, ref))
        end
        e2 = cross(e3, e1)

        # Normalized through-thickness depth ζ ∈ [0 (core), 1 (mold wall skin)]
        ζ = if !isnothing(r_sq) && thick_r_sq[i, j, k] > UInt16(0)
            ratio = clamp(T(r_sq[i, j, k]) / T(thick_r_sq[i, j, k]), zero(T), one(T))
            one(T) - sqrt(ratio)
        else
            T(0.5)
        end

        # Local tensor components (λ_flow, λ_trans, λ_norm, λ_13)
        if is_weld
            λ_flow, λ_trans, λ_norm, λ_13 = T(0.15), T(0.78), T(0.07), zero(T)
        else
            α_div = T(0.5) * (one(T) + tanh(div_u * d_scale))
            c_flow  = (one(T) - α_div) * T(0.72) + α_div * T(0.18)
            c_trans = (one(T) - α_div) * T(0.20) + α_div * T(0.74)
            c_norm  = T(0.08)

            s_flow, s_trans, s_norm = T(0.84), T(0.12), T(0.04)

            w_shear = ζ^p_exp
            λ_flow  = (one(T) - w_shear) * c_flow  + w_shear * s_flow
            λ_trans = (one(T) - w_shear) * c_trans + w_shear * s_trans
            λ_norm  = (one(T) - w_shear) * c_norm  + w_shear * s_norm
            λ_13    = s_tilt * T(4) * ζ * (one(T) - ζ) * w_shear
        end

        # Full symmetric tensor rotation: A_global = R * A_local * Rᵀ
        a11[i, j, k] = λ_flow * e1[1]^2 + λ_trans * e2[1]^2 + λ_norm * e3[1]^2 +
                       T(2) * λ_13 * e1[1] * e3[1]
        a22[i, j, k] = λ_flow * e1[2]^2 + λ_trans * e2[2]^2 + λ_norm * e3[2]^2 +
                       T(2) * λ_13 * e1[2] * e3[2]
        a33[i, j, k] = λ_flow * e1[3]^2 + λ_trans * e2[3]^2 + λ_norm * e3[3]^2 +
                       T(2) * λ_13 * e1[3] * e3[3]

        a12[i, j, k] = λ_flow * e1[1]*e1[2] + λ_trans * e2[1]*e2[2] + λ_norm * e3[1]*e3[2] +
                       λ_13 * (e1[1]*e3[2] + e3[1]*e1[2])
        a13[i, j, k] = λ_flow * e1[1]*e1[3] + λ_trans * e2[1]*e2[3] + λ_norm * e3[1]*e3[3] +
                       λ_13 * (e1[1]*e3[3] + e3[1]*e1[3])
        a23[i, j, k] = λ_flow * e1[2]*e1[3] + λ_trans * e2[2]*e2[3] + λ_norm * e3[2]*e3[3] +
                       λ_13 * (e1[2]*e3[3] + e3[2]*e1[3])
    end

    masks = (;
        weld_lines      = SparseVoxelMask(dims, weld_inds),
        meld_lines      = SparseVoxelMask(dims, meld_inds),
        air_traps       = SparseVoxelMask(dims, trap_inds),
        diverging_flow,
        converging_flow,
        hesitation
    )

    fot = (; a11, a22, a33, a12, a13, a23)

    return (; masks, flow_dirs, front_speed, fot)
end
