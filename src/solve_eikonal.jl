



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