

"""
    MeshAdjacency

Compact, thread-safe Compressed Sparse Row (CSR) triangle-to-triangle edge adjacency.
Build once per mesh and reuse across all interactive gate evaluations.
"""
struct MeshAdjacency
    offsets::Vector{Int32}   # length n_tris + 1
    neighbors::Vector{Int32} # flat array of adjacent triangle indices
end

@inline tri_neighbors(adj::MeshAdjacency, idx::Int) =
    @view adj.neighbors[adj.offsets[idx]:(adj.offsets[idx + 1] - 1)]

function build_mesh_adjacency(mesh::GeometryBasics.Mesh)
    tris = GeometryBasics.faces(mesh)
    n_tris = length(tris)

    # 3 half-edges per triangle: (v_min, v_max, tri_idx)
    edges = Vector{Tuple{Int32, Int32, Int32}}(undef, 3 * n_tris)
    @inbounds for idx in 1:n_tris
        tri = tris[idx]
        v1, v2, v3 = Int32(tri[1]), Int32(tri[2]), Int32(tri[3])
        t = Int32(idx)
        edges[3idx - 2] = (min(v1, v2), max(v1, v2), t)
        edges[3idx - 1] = (min(v2, v3), max(v2, v3), t)
        edges[3idx]     = (min(v3, v1), max(v3, v1), t)
    end

    sort!(edges)

    counts = zeros(Int32, n_tris)
    @inbounds for e in 1:(length(edges) - 1)
        if edges[e][1] == edges[e + 1][1] && edges[e][2] == edges[e + 1][2]
            t1, t2 = edges[e][3], edges[e + 1][3]
            counts[t1] += Int32(1)
            counts[t2] += Int32(1)
        end
    end

    offsets = Vector{Int32}(undef, n_tris + 1)
    offsets[1] = Int32(1)
    @inbounds for i in 1:n_tris
        offsets[i + 1] = offsets[i] + counts[i]
    end

    neighbors = Vector{Int32}(undef, offsets[end] - 1)
    cursor = copy(offsets)

    @inbounds for e in 1:(length(edges) - 1)
        if edges[e][1] == edges[e + 1][1] && edges[e][2] == edges[e + 1][2]
            t1, t2 = edges[e][3], edges[e + 1][3]
            neighbors[cursor[t1]] = t2; cursor[t1] += Int32(1)
            neighbors[cursor[t2]] = t1; cursor[t2] += Int32(1)
        end
    end

    return MeshAdjacency(offsets, neighbors)
end

"""
    voxel_gradient_at(i, j, k, T_field, grid, dx, dy, dz) -> SVector{3, T}

Computes the local Eikonal arrival-time gradient ∇T at a single interior voxel (i, j, k)
on the fly—requiring zero 3D gradient array allocation.
"""
@inline function voxel_gradient_at(
    i::Int, j::Int, k::Int,
    T_field::AbstractArray{T1, 3},
    grid::AbstractArray{Bool, 3},
    dx::T, dy::T, dz::T
) where {T1<:Real, T<:AbstractFloat}
    nx, ny, nz = size(grid)
    @inbounds begin
        (grid[i, j, k] && isfinite(T_field[i, j, k])) || return zero(SVector{3, T})
        tc = T(T_field[i, j, k])

        xm = i > 1  && grid[i-1, j, k] && isfinite(T_field[i-1, j, k])
        xp = i < nx && grid[i+1, j, k] && isfinite(T_field[i+1, j, k])
        gx = (xm && xp) ? (T(T_field[i+1, j, k]) - T(T_field[i-1, j, k])) / (T(2) * dx) :
             xp         ? (T(T_field[i+1, j, k]) - tc) / dx :
             xm         ? (tc - T(T_field[i-1, j, k])) / dx : zero(T)

        ym = j > 1  && grid[i, j-1, k] && isfinite(T_field[i, j-1, k])
        yp = j < ny && grid[i, j+1, k] && isfinite(T_field[i, j+1, k])
        gy = (ym && yp) ? (T(T_field[i, j+1, k]) - T(T_field[i, j-1, k])) / (T(2) * dy) :
             yp         ? (T(T_field[i, j+1, k]) - tc) / dy :
             ym         ? (tc - T(T_field[i, j-1, k])) / dy : zero(T)

        zm = k > 1  && grid[i, j, k-1] && isfinite(T_field[i, j, k-1])
        zp = k < nz && grid[i, j, k+1] && isfinite(T_field[i, j, k+1])
        gz = (zm && zp) ? (T(T_field[i, j, k+1]) - T(T_field[i, j, k-1])) / (T(2) * dz) :
             zp         ? (T(T_field[i, j, k+1]) - tc) / dz :
             zm         ? (tc - T(T_field[i, j, k-1])) / dz : zero(T)

        g = SVector{3, T}(gx, gy, gz)
        gn = norm(g)
        return gn > T(1e-9) ? (g / gn) : zero(SVector{3, T})
    end
end

"""
    project_and_compute_fot_parallel(
        meshdata, voxeldata, T_field, thick_r_sq;
        adj = build_mesh_adjacency(meshdata.mesh),
        n_layers = 10,
        inward_offset_vox = 0.75,
        weld_dot_thresh = -0.3,
        meld_dot_thresh = 0.4,
        div_thresh = 0.05,
        core_exponent = 2.0,
        div_scale = 5.0,
        max_shear_tilt = 0.08
    )

Multi-threaded voxel-to-triangle projection and N-layer 6-component FOT solver.
Evaluates gradients on the fly only at surface triangles (zero 3D vector allocations).
"""
function project_and_compute_fot_parallel(
    meshdata,
    voxeldata,
    T_field::AbstractArray{T1, 3},
    thick_r_sq::AbstractArray{UInt16, 3};
    adj::MeshAdjacency = build_mesh_adjacency(meshdata.mesh),
    n_layers::Int = 10,
    inward_offset_vox::Real = 0.75,
    weld_dot_thresh::Real = -0.3,
    meld_dot_thresh::Real = 0.4,
    div_thresh::Real = 0.05,
    core_exponent::Real = 2.0,
    div_scale::Real = 5.0,
    max_shear_tilt::Real = 0.08
) where {T1<:Real}

    mesh   = meshdata.mesh
    pts    = GeometryBasics.coordinates(mesh)
    tris   = GeometryBasics.faces(mesh)
    n_tris = length(tris)

    grid = voxeldata.grid
    spacing_raw = hasproperty(voxeldata, :voxel_spacing) ? voxeldata.voxel_spacing : voxeldata.stepsize
    corner_raw  = voxeldata.corner

    T = float(promote_type(T1, eltype(spacing_raw), eltype(corner_raw)))
    nx, ny, nz = size(grid)
    dx, dy, dz = T(spacing_raw[1]), T(spacing_raw[2]), T(spacing_raw[3])
    ox, oy, oz = T(corner_raw[1]),  T(corner_raw[2]),  T(corner_raw[3])
    dx_mean     = (dx + dy + dz) / T(3)
    offset_dist = T(inward_offset_vox) * dx_mean

    tri_centroids = Vector{SVector{3, T}}(undef, n_tris)
    tri_normals   = Vector{SVector{3, T}}(undef, n_tris)
    tri_time      = Vector{T}(undef, n_tris)
    tri_flow_dir  = Vector{SVector{3, T}}(undef, n_tris)
    tri_thickness = Vector{T}(undef, n_tris)

    # STAGE 1 (Parallel): Project fill_time, local flow_dir, and thickness to each triangle
    Threads.@threads for idx in 1:n_tris
        @inbounds begin
            tri = tris[idx]
            p1 = SVector{3, T}(pts[Int(tri[1])])
            p2 = SVector{3, T}(pts[Int(tri[2])])
            p3 = SVector{3, T}(pts[Int(tri[3])])

            c = (p1 + p2 + p3) / T(3)
            n_raw = cross(p2 - p1, p3 - p1)
            n_norm = norm(n_raw)
            n_hat = n_norm > T(1e-12) ? (n_raw / n_norm) : SVector{3, T}(0, 0, 1)

            tri_centroids[idx] = c
            tri_normals[idx]   = n_hat

            p_in = c - offset_dist * n_hat
            fx = (p_in[1] - ox) / dx + T(0.5)
            fy = (p_in[2] - oy) / dy + T(0.5)
            fz = (p_in[3] - oz) / dz + T(0.5)

            i0 = clamp(floor(Int, fx), 1, nx - 1)
            j0 = clamp(floor(Int, fy), 1, ny - 1)
            k0 = clamp(floor(Int, fz), 1, nz - 1)

            tx = clamp(fx - T(i0), zero(T), one(T))
            ty = clamp(fy - T(j0), zero(T), one(T))
            tz = clamp(fz - T(k0), zero(T), one(T))

            wx0, wx1 = one(T) - tx, tx
            wy0, wy1 = one(T) - ty, ty
            wz0, wz1 = one(T) - tz, tz

            t_acc   = zero(T)
            u_acc   = zero(SVector{3, T})
            rsq_acc = zero(T)
            w_acc   = zero(T)

            for dk in 0:1, dj in 0:1, di in 0:1
                ii, jj, kk = i0 + di, j0 + dj, k0 + dk
                (grid[ii, jj, kk] && isfinite(T_field[ii, jj, kk])) || continue
                w = (di == 0 ? wx0 : wx1) * (dj == 0 ? wy0 : wy1) * (dk == 0 ? wz0 : wz1)
                t_acc   += w * T(T_field[ii, jj, kk])
                u_acc   += w * voxel_gradient_at(ii, jj, kk, T_field, grid, dx, dy, dz)
                rsq_acc += w * T(thick_r_sq[ii, jj, kk])
                w_acc   += w
            end

            if w_acc > T(1e-7)
                t_val   = t_acc / w_acc
                u_raw   = u_acc / w_acc
                rsq_val = rsq_acc / w_acc
            else
                # Nearest-interior fallback (4-voxel radius)
                ic = clamp(round(Int, fx), 1, nx)
                jc = clamp(round(Int, fy), 1, ny)
                kc = clamp(round(Int, fz), 1, nz)
                best_d2 = T(Inf)
                t_val   = T(Inf)
                u_raw   = zero(SVector{3, T})
                rsq_val = zero(T)

                for r in 1:4
                    for kk in max(1, kc - r):min(nz, kc + r),
                        jj in max(1, jc - r):min(ny, jc + r),
                        ii in max(1, ic - r):min(nx, ic + r)
                        (grid[ii, jj, kk] && isfinite(T_field[ii, jj, kk])) || continue
                        vx = ox + (T(ii) - T(0.5)) * dx
                        vy = oy + (T(jj) - T(0.5)) * dy
                        vz = oz + (T(kk) - T(0.5)) * dz
                        d2 = (vx - p_in[1])^2 + (vy - p_in[2])^2 + (vz - p_in[3])^2
                        if d2 < best_d2
                            best_d2 = d2
                            t_val   = T(T_field[ii, jj, kk])
                            u_raw   = voxel_gradient_at(ii, jj, kk, T_field, grid, dx, dy, dz)
                            rsq_val = T(thick_r_sq[ii, jj, kk])
                        end
                    end
                    isfinite(best_d2) && break
                end
            end

            tri_time[idx]      = t_val
            tri_thickness[idx] = T(2) * sqrt(max(rsq_val, zero(T))) * dx_mean
            u_tan = u_raw - dot(u_raw, n_hat) * n_hat
            tri_flow_dir[idx] = norm(u_tan) > T(1e-6) ? normalize(u_tan) : zero(SVector{3, T})
        end
    end

    # STAGE 2 (Parallel): Surface divergence, defect masks, and N-layer 6-component FOT
    tri_div        = zeros(T, n_tris)
    weld_lines     = Vector{Bool}(undef, n_tris)
    meld_lines     = Vector{Bool}(undef, n_tris)
    air_traps      = Vector{Bool}(undef, n_tris)
    diverging_flow = Vector{Bool}(undef, n_tris)

    zeta = [T(l - 0.5) / T(n_layers) for l in 1:n_layers]
    a11 = zeros(T, n_tris, n_layers)
    a22 = zeros(T, n_tris, n_layers)
    a33 = zeros(T, n_tris, n_layers)
    a12 = zeros(T, n_tris, n_layers)
    a13 = zeros(T, n_tris, n_layers)
    a23 = zeros(T, n_tris, n_layers)

    p_exp   = T(core_exponent)
    d_scale = T(div_scale)
    s_tilt  = T(max_shear_tilt)

    Threads.@threads for idx in 1:n_tris
        @inbounds begin
            tc    = tri_time[idx]
            e1    = tri_flow_dir[idx]
            e3    = tri_normals[idx]
            c_c   = tri_centroids[idx]
            has_u = isfinite(tc) && norm(e1) > T(1e-6)

            nbrs = tri_neighbors(adj, idx)
            min_dot = one(T)
            is_local_max = !isempty(nbrs)
            div_acc = zero(T)
            n_nbrs = 0

            for nbr_i32 in nbrs
                nbr = Int(nbr_i32)
                tn = tri_time[nbr]
                isfinite(tn) || continue
                if tn > tc
                    is_local_max = false
                end

                u_n = tri_flow_dir[nbr]
                if has_u && norm(u_n) > T(1e-6)
                    min_dot = min(min_dot, dot(e1, u_n))
                    dc = tri_centroids[nbr] - c_c
                    dist = norm(dc)
                    if dist > T(1e-8)
                        div_acc += dot(u_n - e1, dc / dist) / dist
                        n_nbrs += 1
                    end
                end
            end

            is_trap = has_u && is_local_max && tc > zero(T)
            is_weld = has_u && (min_dot < T(weld_dot_thresh))
            is_meld = has_u && !is_weld && (min_dot < T(meld_dot_thresh))
            div_u   = n_nbrs > 0 ? (div_acc / T(n_nbrs)) : zero(T)

            air_traps[idx]      = is_trap
            weld_lines[idx]     = is_weld
            meld_lines[idx]     = is_meld
            tri_div[idx]        = div_u
            diverging_flow[idx] = has_u && (div_u > T(div_thresh))

            if !has_u
                for l in 1:n_layers
                    a11[idx, l] = T(1) / T(3)
                    a22[idx, l] = T(1) / T(3)
                    a33[idx, l] = T(1) / T(3)
                end
                continue
            end

            e2 = normalize(cross(e3, e1))

            α_div   = T(0.5) * (one(T) + tanh(div_u * d_scale))
            c_flow  = (one(T) - α_div) * T(0.72) + α_div * T(0.18)
            c_trans = (one(T) - α_div) * T(0.20) + α_div * T(0.74)
            c_norm  = T(0.08)
            s_flow, s_trans, s_norm = T(0.84), T(0.12), T(0.04)

            for l in 1:n_layers
                ζ = zeta[l]
                if is_weld
                    λ_flow, λ_trans, λ_norm, λ_13 = T(0.15), T(0.78), T(0.07), zero(T)
                else
                    w_shear = ζ^p_exp
                    λ_flow  = (one(T) - w_shear) * c_flow  + w_shear * s_flow
                    λ_trans = (one(T) - w_shear) * c_trans + w_shear * s_trans
                    λ_norm  = (one(T) - w_shear) * c_norm  + w_shear * s_norm
                    λ_13    = s_tilt * T(4) * ζ * (one(T) - ζ) * w_shear
                end

                a11[idx, l] = λ_flow * e1[1]^2 + λ_trans * e2[1]^2 + λ_norm * e3[1]^2 +
                              T(2) * λ_13 * e1[1] * e3[1]
                a22[idx, l] = λ_flow * e1[2]^2 + λ_trans * e2[2]^2 + λ_norm * e3[2]^2 +
                              T(2) * λ_13 * e1[2] * e3[2]
                a33[idx, l] = λ_flow * e1[3]^2 + λ_trans * e2[3]^2 + λ_norm * e3[3]^2 +
                              T(2) * λ_13 * e1[3] * e3[3]

                a12[idx, l] = λ_flow * e1[1]*e1[2] + λ_trans * e2[1]*e2[2] + λ_norm * e3[1]*e3[2] +
                              λ_13 * (e1[1]*e3[2] + e3[1]*e1[2])
                a13[idx, l] = λ_flow * e1[1]*e1[3] + λ_trans * e2[1]*e2[3] + λ_norm * e3[1]*e3[3] +
                              λ_13 * (e1[1]*e3[3] + e3[1]*e1[3])
                a23[idx, l] = λ_flow * e1[2]*e1[3] + λ_trans * e2[2]*e2[3] + λ_norm * e3[2]*e3[3] +
                              λ_13 * (e1[2]*e3[3] + e3[2]*e1[3])
            end
        end
    end

    inv_L = inv(T(n_layers))
    shell_avg = (
        a11 = vec(sum(a11; dims = 2)) .* inv_L,
        a22 = vec(sum(a22; dims = 2)) .* inv_L,
        a33 = vec(sum(a33; dims = 2)) .* inv_L,
        a12 = vec(sum(a12; dims = 2)) .* inv_L,
        a13 = vec(sum(a13; dims = 2)) .* inv_L,
        a23 = vec(sum(a23; dims = 2)) .* inv_L
    )

    masks = (
        weld_lines     = BitVector(weld_lines),
        meld_lines     = BitVector(meld_lines),
        air_traps      = BitVector(air_traps),
        diverging_flow = BitVector(diverging_flow)
    )

    return (;
        centroids  = tri_centroids,
        normals    = tri_normals,
        fill_time  = tri_time,
        flow_dir   = tri_flow_dir,
        thickness  = tri_thickness,
        divergence = tri_div,
        masks,
        zeta,
        layers     = (; a11, a22, a33, a12, a13, a23),
        shell_avg
    )
end