


"""
    sample_flow_trilinear(pos, flow_dirs, voxel_grid, voxel_spacing, corner) -> SVector{3, T}

Trilinearly interpolates the unit flow direction `flow_dirs` at continuous world coordinate `pos`,
weighting only neighboring voxels that lie inside `voxel_grid` (`true`).
Returns a zero vector if `pos` is outside the grid or surrounded solely by void voxels.
"""
function sample_flow_trilinear(
    pos::StaticVector{3, T},
    flow_dirs::AbstractArray{<:StaticVector{3}, 3},
    voxel_grid::AbstractArray{Bool, 3},
    voxel_spacing::StaticVector{3, T},
    corner::StaticVector{3, T}
) where {T<:AbstractFloat}
    nx, ny, nz = size(voxel_grid)

    # Convert world position to continuous 1-based voxel center coordinates
    fx = (pos[1] - corner[1]) / voxel_spacing[1] + T(0.5)
    fy = (pos[2] - corner[2]) / voxel_spacing[2] + T(0.5)
    fz = (pos[3] - corner[3]) / voxel_spacing[3] + T(0.5)

    # Out of grid bounds check
    if fx < one(T) || fx > T(nx) || fy < one(T) || fy > T(ny) || fz < one(T) || fz > T(nz)
        return zero(SVector{3, T})
    end

    i0 = clamp(floor(Int, fx), 1, nx - 1)
    j0 = clamp(floor(Int, fy), 1, ny - 1)
    k0 = clamp(floor(Int, fz), 1, nz - 1)

    tx = clamp(fx - T(i0), zero(T), one(T))
    ty = clamp(fy - T(j0), zero(T), one(T))
    tz = clamp(fz - T(k0), zero(T), one(T))

    wx0, wx1 = one(T) - tx, tx
    wy0, wy1 = one(T) - ty, ty
    wz0, wz1 = one(T) - tz, tz

    v_acc = zero(SVector{3, T})
    w_acc = zero(T)

    @inbounds for dk in 0:1, dj in 0:1, di in 0:1
        ii, jj, kk = i0 + di, j0 + dj, k0 + dk
        voxel_grid[ii, jj, kk] || continue

        u = SVector{3, T}(flow_dirs[ii, jj, kk])
        norm(u) > T(1e-6) || continue

        w = (di == 0 ? wx0 : wx1) * (dj == 0 ? wy0 : wy1) * (dk == 0 ? wz0 : wz1)
        v_acc += w * u
        w_acc += w
    end

    if w_acc <= T(1e-6) || norm(v_acc) <= T(1e-6)
        return zero(SVector{3, T})
    end

    return normalize(v_acc)
end



"""
    smooth_vector_field(vec_field, voxel_grid; passes=3) -> Array{SVector{3, Float32}, 3}

Applies mask-aware 3D Gaussian-weighted smoothing to a vector field inside `voxel_grid`,
eliminating voxel staircase jitter while keeping vectors strictly unit-length.
"""
function smooth_vector_field(
    vec_field::AbstractArray{<:StaticVector{3}, 3},
    voxel_grid::AbstractArray{Bool, 3};
    passes::Int = 3
)
    nx, ny, nz = size(voxel_grid)
    curr = Array{SVector{3, Float32}, 3}(undef, nx, ny, nz)
    next_buf = Array{SVector{3, Float32}, 3}(undef, nx, ny, nz)

    @inbounds for idx in eachindex(voxel_grid)
        v = SVector{3, Float32}(vec_field[idx])
        curr[idx] = (voxel_grid[idx] && norm(v) > 1.0f-6) ? normalize(v) : zero(SVector{3, Float32})
    end

    # 1D Gaussian weights [1, 2, 1] -> 3D separable weights w(di)*w(dj)*w(dk)
    w1d = (1.0f0, 2.0f0, 1.0f0)

    for _ in 1:passes
        @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
            if !voxel_grid[i, j, k]
                next_buf[i, j, k] = zero(SVector{3, Float32})
                continue
            end

            acc = zero(SVector{3, Float32})
            for dk in -1:1
                kk = k + dk
                (1 <= kk <= nz) || continue
                wk = w1d[dk + 2]
                for dj in -1:1
                    jj = j + dj
                    (1 <= jj <= ny) || continue
                    wjk = wk * w1d[dj + 2]
                    for di in -1:1
                        ii = i + di
                        (1 <= ii <= nx && voxel_grid[ii, jj, kk]) || continue
                        v = curr[ii, jj, kk]
                        norm(v) > 1.0f-6 || continue
                        acc += (wjk * w1d[di + 2]) * v
                    end
                end
            end

            next_buf[i, j, k] = norm(acc) > 1.0f-6 ? normalize(acc) : curr[i, j, k]
        end
        curr, next_buf = next_buf, curr
    end

    return curr
end

"""
    compute_smooth_wall_normals(r_sq, voxel_grid, spacing; passes=3)

Computes a smooth, unit-length inward wall-normal field (`∇r_sq`) across the interior voxels
so midplane guidance never suffers from integer `UInt16` staircase jumps.
"""
function compute_smooth_wall_normals(
    r_sq::AbstractArray{UInt16, 3},
    voxel_grid::AbstractArray{Bool, 3},
    spacing::SVector{3, Float32};
    passes::Int = 3
)
    nx, ny, nz = size(voxel_grid)
    raw_normals = fill(zero(SVector{3, Float32}), nx, ny, nz)

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        voxel_grid[i, j, k] || continue
        rc = Float32(r_sq[i, j, k])

        gx = (i > 1 && i < nx && voxel_grid[i-1,j,k] && voxel_grid[i+1,j,k]) ?
             (Float32(r_sq[i+1,j,k]) - Float32(r_sq[i-1,j,k])) / (2.0f0 * spacing[1]) :
             (i < nx && voxel_grid[i+1,j,k]) ? (Float32(r_sq[i+1,j,k]) - rc) / spacing[1] :
             (i > 1  && voxel_grid[i-1,j,k]) ? (rc - Float32(r_sq[i-1,j,k])) / spacing[1] : 0.0f0

        gy = (j > 1 && j < ny && voxel_grid[i,j-1,k] && voxel_grid[i,j+1,k]) ?
             (Float32(r_sq[i,j+1,k]) - Float32(r_sq[i,j-1,k])) / (2.0f0 * spacing[2]) :
             (j < ny && voxel_grid[i,j+1,k]) ? (Float32(r_sq[i,j+1,k]) - rc) / spacing[2] :
             (j > 1  && voxel_grid[i,j-1,k]) ? (rc - Float32(r_sq[i,j-1,k])) / spacing[2] : 0.0f0

        gz = (k > 1 && k < nz && voxel_grid[i,j,k-1] && voxel_grid[i,j,k+1]) ?
             (Float32(r_sq[i,j,k+1]) - Float32(r_sq[i,j,k-1])) / (2.0f0 * spacing[3]) :
             (k < nz && voxel_grid[i,j,k+1]) ? (Float32(r_sq[i,j,k+1]) - rc) / spacing[3] :
             (k > 1  && voxel_grid[i,j,k-1]) ? (rc - Float32(r_sq[i,j,k-1])) / spacing[3] : 0.0f0

        g = SVector{3, Float32}(gx, gy, gz)
        if norm(g) > 1.0f-5
            raw_normals[i, j, k] = normalize(g)
        end
    end

    return smooth_vector_field(raw_normals, voxel_grid; passes = passes)
end

"""
    sample_scalar_trilinear(pos, field, voxel_grid, spacing, corner) -> Float32

Trilinearly interpolates a scalar field (such as `fill_time`) over valid interior voxels.
"""
function sample_scalar_trilinear(
    pos::SVector{3, Float32},
    field::AbstractArray{<:Real, 3},
    voxel_grid::AbstractArray{Bool, 3},
    spacing::SVector{3, Float32},
    corner::SVector{3, Float32}
)
    nx, ny, nz = size(voxel_grid)
    fx = (pos[1] - corner[1]) / spacing[1] + 0.5f0
    fy = (pos[2] - corner[2]) / spacing[2] + 0.5f0
    fz = (pos[3] - corner[3]) / spacing[3] + 0.5f0

    (fx < 1.0f0 || fx > Float32(nx) || fy < 1.0f0 || fy > Float32(ny) || fz < 1.0f0 || fz > Float32(nz)) && return Inf32

    i0 = clamp(floor(Int, fx), 1, nx - 1)
    j0 = clamp(floor(Int, fy), 1, ny - 1)
    k0 = clamp(floor(Int, fz), 1, nz - 1)

    tx = clamp(fx - Float32(i0), 0.0f0, 1.0f0)
    ty = clamp(fy - Float32(j0), 0.0f0, 1.0f0)
    tz = clamp(fz - Float32(k0), 0.0f0, 1.0f0)

    wx0, wx1 = 1.0f0 - tx, tx
    wy0, wy1 = 1.0f0 - ty, ty
    wz0, wz1 = 1.0f0 - tz, tz

    val_acc = 0.0f0
    w_acc   = 0.0f0

    @inbounds for dk in 0:1, dj in 0:1, di in 0:1
        ii, jj, kk = i0 + di, j0 + dj, k0 + dk
        (voxel_grid[ii, jj, kk] && isfinite(field[ii, jj, kk])) || continue
        w = (di == 0 ? wx0 : wx1) * (dj == 0 ? wy0 : wy1) * (dk == 0 ? wz0 : wz1)
        val_acc += w * Float32(field[ii, jj, kk])
        w_acc   += w
    end

    return w_acc > 1.0f-6 ? (val_acc / w_acc) : Inf32
end

"""
    generate_streamlines(
        positions, vector_field, voxeldata;
        fill_time = nothing,
        r_sq = nothing,
        num_streamlines = 16,
        num_steps = 1500,
        stepsize = nothing,
        seed_radius = nothing,
        smooth_passes = 3,
        wall_repulsion = 0.25,
        weld_mask = nothing
    ) -> Vector{Vector{Point3f}}
"""
function generate_streamlines(
    positions::AbstractVector,
    vector_field::AbstractArray{<:StaticVector{3}, 3},
    voxeldata::VoxelData;
    fill_time::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
    r_sq::Union{Nothing, AbstractArray{UInt16, 3}} = nothing,
    num_streamlines::Int = 16,
    num_steps::Int = 1500,
    stepsize::Union{Nothing, Real} = nothing,
    seed_radius::Union{Nothing, Real} = nothing,
    smooth_passes::Int = 3,
    wall_repulsion::Real = 0.25,
    weld_mask::Union{Nothing, AbstractArray{Bool, 3}} = nothing
)
    grid = voxeldata.grid
    spacing_raw = voxeldata.voxel_spacing
    corner_raw  = voxeldata.corner

    nx, ny, nz = size(grid)
    spacing = SVector{3, Float32}(spacing_raw[1], spacing_raw[2], spacing_raw[3])
    corner  = SVector{3, Float32}(corner_raw[1],  corner_raw[2],  corner_raw[3])
    dx_min  = minimum(spacing)

    ds     = isnothing(stepsize) ? 0.5f0 * dx_min : Float32(stepsize)
    r_seed = isnothing(seed_radius) ? 1.5f0 * dx_min : Float32(seed_radius)
    w_rep  = Float32(wall_repulsion)

    # Pre-smooth both the flow gradient field and the inward wall normals
    smooth_flow = smooth_passes > 0 ?
        smooth_vector_field(vector_field, grid; passes = smooth_passes) : vector_field
    smooth_wall_n = !isnothing(r_sq) ?
        compute_smooth_wall_normals(r_sq, grid, spacing; passes = max(smooth_passes, 2)) : nothing

    function snap_to_core(p::SVector{3, Float32}, search_r::Int = 4)
        i0 = clamp(round(Int, (p[1] - corner[1]) / spacing[1] + 0.5f0), 1, nx)
        j0 = clamp(round(Int, (p[2] - corner[2]) / spacing[2] + 0.5f0), 1, ny)
        k0 = clamp(round(Int, (p[3] - corner[3]) / spacing[3] + 0.5f0), 1, nz)

        best_p = nothing
        best_score = -Inf32

        @inbounds for kk in max(1, k0 - search_r):min(nz, k0 + search_r),
                      jj in max(1, j0 - search_r):min(ny, j0 + search_r),
                      ii in max(1, i0 - search_r):min(nx, i0 + search_r)
            grid[ii, jj, kk] || continue
            vp = SVector{3, Float32}(
                corner[1] + (ii - 0.5f0) * spacing[1],
                corner[2] + (jj - 0.5f0) * spacing[2],
                corner[3] + (kk - 0.5f0) * spacing[3]
            )
            dist_pen   = norm(vp - p) / dx_min
            core_bonus = isnothing(r_sq) ? 0.0f0 : 2.0f0 * Float32(r_sq[ii, jj, kk])
            score      = core_bonus - dist_pen
            if score > best_score
                best_score = score
                best_p = vp
            end
        end
        return best_p
    end

    function eval_guided_velocity(p::SVector{3, Float32})
        v = sample_flow_trilinear(p, smooth_flow, grid, spacing, corner)
        norm(v) < 1.0f-5 && return zero(SVector{3, Float32})

        if !isnothing(smooth_wall_n) && w_rep > 0.0f0
            n_in = sample_flow_trilinear(p, smooth_wall_n, grid, spacing, corner)
            if norm(n_in) > 1.0f-5
                vn = dot(v, n_in)
                # Strip outward wall-pointing component + add gentle smooth centering
                v_tan = vn < 0.0f0 ? (v - vn * n_in) : v
                v_guided = v_tan + w_rep * n_in
                if norm(v_guided) > 1.0f-5
                    return normalize(v_guided)
                end
            end
        end
        return v
    end

    golden_ratio = (1.0f0 + sqrt(5.0f0)) / 2.0f0
    seed_offsets = SVector{3, Float32}[]
    if num_streamlines == 1
        push!(seed_offsets, zero(SVector{3, Float32}))
    else
        for m in 0:(num_streamlines - 1)
            θ = 2.0f0 * Float32(π) * m / golden_ratio
            ϕ = acos(1.0f0 - 2.0f0 * (m + 0.5f0) / num_streamlines)
            push!(seed_offsets, SVector{3, Float32}(
                r_seed * sin(ϕ) * cos(θ),
                r_seed * sin(ϕ) * sin(θ),
                r_seed * cos(ϕ)
            ))
        end
    end

    streamlines = Vector{Point3f}[]

    for gate_item in positions
        raw_pos = hasproperty(gate_item, :position) ? gate_item.position : gate_item
        g_pos = SVector{3, Float32}(raw_pos[1], raw_pos[2], raw_pos[3])

        core_gate = snap_to_core(g_pos, 5)
        isnothing(core_gate) && continue

        for off in seed_offsets
            start_pt = snap_to_core(core_gate + off, 3)
            isnothing(start_pt) && continue

            pts = Point3f[Point3f(start_pt)]
            curr = start_pt
            t_prev = isnothing(fill_time) ? -Inf32 : sample_scalar_trilinear(curr, fill_time, grid, spacing, corner)

            for _ in 1:num_steps
                v1 = eval_guided_velocity(curr)
                norm(v1) < 1.0f-5 && break

                mid = curr + 0.5f0 * ds * v1
                v2 = eval_guided_velocity(mid)
                v_step = norm(v2) > 1.0f-5 ? v2 : v1

                next_pt = curr + ds * v_step

                ii = round(Int, (next_pt[1] - corner[1]) / spacing[1] + 0.5f0)
                jj = round(Int, (next_pt[2] - corner[2]) / spacing[2] + 0.5f0)
                kk = round(Int, (next_pt[3] - corner[3]) / spacing[3] + 0.5f0)

                (1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz) || break
                @inbounds grid[ii, jj, kk] || break

                # 1. Monotonic arrival-time check: stops immediately at air traps / weld sinks
                if !isnothing(fill_time)
                    t_next = sample_scalar_trilinear(next_pt, fill_time, grid, spacing, corner)
                    if isfinite(t_prev) && isfinite(t_next) && t_next <= t_prev + 1.0f-6
                        break
                    end
                    t_prev = t_next
                end

                # 2. Anti-reversal check: stops immediately if direction flips > 110° at a sink
                if length(pts) >= 2
                    prev_dir = normalize(SVector{3, Float32}(pts[end]) - SVector{3, Float32}(pts[end-1]))
                    if dot(prev_dir, v_step) < -0.35f0
                        break
                    end
                end

                push!(pts, Point3f(next_pt))
                curr = next_pt

                if !isnothing(weld_mask) && @inbounds weld_mask[ii, jj, kk]
                    break
                end

                # 3. Sliding-window loop/stagnation check: catches any tight orbital "ball"
                if length(pts) >= 12
                    net_disp = norm(SVector{3, Float32}(pts[end]) - SVector{3, Float32}(pts[end-10]))
                    if net_disp < 2.0f0 * ds
                        # Trim the last few orbital points before breaking
                        resize!(pts, length(pts) - 5)
                        break
                    end
                end
            end

            length(pts) >= 3 && push!(streamlines, pts)
        end
    end

    return streamlines
end