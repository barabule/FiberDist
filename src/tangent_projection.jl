

"""
    get_a33(a11, a22)

Reconstructs the 6th symmetric Fiber Orientation Tensor component `a33` from the
unit-trace identity `a11 + a22 + a33 = 1`, clamped at zero against `Float32` round-off.
"""
@inline get_a33(a11::T, a22::T) where {T<:AbstractFloat} = max(zero(T), one(T) - a11 - a22)



@inline tri_neighbors(adj::MeshAdjacency, idx::Int) =
    @view adj.neighbors[adj.offsets[idx]:(adj.offsets[idx + 1] - 1)]

function build_mesh_adjacency(mesh)
    tris = mesh.faces
    n_tris = length(tris)

    edges = Vector{Tuple{Int32, Int32, Int32}}(undef, 3 * n_tris)
    @inbounds for idx in 1:n_tris
        v1, v2, v3 = tris[idx]
        t = Int32(idx)
        edges[3idx - 2] = (min(v1, v2), max(v1, v2), t)
        edges[3idx - 1] = (min(v2, v3), max(v2, v3), t)
        edges[3idx]     = (min(v3, v1), max(v3, v1), t)
    end

    sort!(edges)

    counts = zeros(Int32, n_tris)
    @inbounds for e in 1:(length(edges) - 1)
        if edges[e][1] == edges[e + 1][1] && edges[e][2] == edges[e + 1][2]
            counts[edges[e][3]]     += Int32(1)
            counts[edges[e + 1][3]] += Int32(1)
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

@inline function voxel_gradient_at(
    i::Int, j::Int, k::Int,
    fill_time::AbstractArray{T1, 3},
    grid::AbstractArray{Bool, 3},
    dx::T, dy::T, dz::T
) where {T1<:Real, T<:AbstractFloat}
    nx, ny, nz = size(grid)
    @inbounds begin
        (grid[i, j, k] && isfinite(fill_time[i, j, k])) || return zero(SVector{3, T})
        tc = T(fill_time[i, j, k])

        xm = i > 1  && grid[i-1, j, k] && isfinite(fill_time[i-1, j, k])
        xp = i < nx && grid[i+1, j, k] && isfinite(fill_time[i+1, j, k])
        gx = (xm && xp) ? (T(fill_time[i+1, j, k]) - T(fill_time[i-1, j, k])) / (T(2) * dx) :
             xp         ? (T(fill_time[i+1, j, k]) - tc) / dx :
             xm         ? (tc - T(fill_time[i-1, j, k])) / dx : zero(T)

        ym = j > 1  && grid[i, j-1, k] && isfinite(fill_time[i, j-1, k])
        yp = j < ny && grid[i, j+1, k] && isfinite(fill_time[i, j+1, k])
        gy = (ym && yp) ? (T(fill_time[i, j+1, k]) - T(fill_time[i, j-1, k])) / (T(2) * dy) :
             yp         ? (T(fill_time[i, j+1, k]) - tc) / dy :
             ym         ? (tc - T(fill_time[i, j-1, k])) / dy : zero(T)

        zm = k > 1  && grid[i, j, k-1] && isfinite(fill_time[i, j, k-1])
        zp = k < nz && grid[i, j, k+1] && isfinite(fill_time[i, j, k+1])
        gz = (zm && zp) ? (T(fill_time[i, j, k+1]) - T(fill_time[i, j, k-1])) / (T(2) * dz) :
             zp         ? (T(fill_time[i, j, k+1]) - tc) / dz :
             zm         ? (tc - T(fill_time[i, j, k-1])) / dz : zero(T)

        g = SVector{3, T}(gx, gy, gz)
        gn = norm(g)
        return gn > T(1e-9) ? (g / gn) : zero(SVector{3, T})
    end
end

"""
    project_voxel_to_mesh_triangles(
        meshdata::MeshData,
        voxeldata::VoxelData,
        fill_time::AbstractArray{T1, 3},
        thickness::ThicknessField;
        adj::MeshAdjacency = build_mesh_adjacency(meshdata.mesh),
        inward_offset_vox::Real = 0.75,
        weld_dot_thresh::Real = -0.3,
        meld_dot_thresh::Real = 0.4,
        div_thresh::Real = 0.05
    ) where {T1<:Real}

Multi-threaded projection of Eikonal `fill_time`, tangent-projected unit flow directions `e1`,
local physical wall thickness `h`, in-plane flow divergence `div_u`, and surface defect masks
(`weld_lines`, `meld_lines`, `air_traps`, `diverging_flow`) onto every triangle of `meshdata.mesh`.
Computes 3D gradients on the fly at triangle neighborhoods with zero full-grid vector allocations.
"""
function project_voxel_to_mesh_triangles(
    meshdata::MeshData,
    voxeldata::VoxelData,
    fill_time::AbstractArray{T1, 3},
    thickness::ThicknessField;
    adj::MeshAdjacency = build_mesh_adjacency(meshdata.mesh),
    inward_offset_vox::Real = 0.75,
    weld_dot_thresh::Real = -0.3,
    meld_dot_thresh::Real = 0.4,
    div_thresh::Real = 0.05
) where {T1<:Real}

    mesh   = meshdata.mesh
    pts    = mesh.position
    tris   = mesh.faces
    n_tris = length(tris)

    grid        = voxeldata.grid
    spacing_raw = voxeldata.voxel_spacing
    corner_raw  = voxeldata.corner
    thick_r_sq  = thickness.thick_r_sq

    T = float(promote_type(T1, eltype(spacing_raw), eltype(corner_raw)))
    nx, ny, nz  = size(grid)
    dx, dy, dz  = T(spacing_raw[1]), T(spacing_raw[2]), T(spacing_raw[3])
    ox, oy, oz  = T(corner_raw[1]),  T(corner_raw[2]),  T(corner_raw[3])
    dx_mean     = (dx + dy + dz) / T(3)
    offset_dist = T(inward_offset_vox) * dx_mean

    tri_centroids = Vector{SVector{3, T}}(undef, n_tris)
    tri_normals   = Vector{SVector{3, T}}(undef, n_tris)
    tri_time      = Vector{T}(undef, n_tris)
    tri_flow_dir  = Vector{SVector{3, T}}(undef, n_tris)
    tri_thickness = Vector{T}(undef, n_tris)

    # Pass 1 (Parallel): Project fill_time, flow_dir, and thickness onto each triangle
    Threads.@threads for idx in 1:n_tris
        @inbounds begin
            i1, i2, i3 = tris[idx]
            p1 = SVector{3, T}(pts[i1])
            p2 = SVector{3, T}(pts[i2])
            p3 = SVector{3, T}(pts[i3])

            c = (p1 + p2 + p3) / T(3)
            n_raw = cross(p2 - p1, p3 - p1)
            n_norm = norm(n_raw)
            n_hat = n_norm > T(1e-12) ? (n_raw / n_norm) : SVector{3, T}(0, 0, 1)

            tri_centroids[idx] = c
            tri_normals[idx]   = n_hat

            # Sample slightly inside the wall along -n_hat to capture midplane-representative flow
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
                (grid[ii, jj, kk] && isfinite(fill_time[ii, jj, kk])) || continue
                w = (di == 0 ? wx0 : wx1) * (dj == 0 ? wy0 : wy1) * (dk == 0 ? wz0 : wz1)
                t_acc   += w * T(fill_time[ii, jj, kk])
                u_acc   += w * voxel_gradient_at(ii, jj, kk, fill_time, grid, dx, dy, dz)
                rsq_acc += w * T(thick_r_sq[ii, jj, kk])
                w_acc   += w
            end

            if w_acc > T(1e-7)
                t_val   = t_acc / w_acc
                u_raw   = u_acc / w_acc
                rsq_val = rsq_acc / w_acc
            else
                # Nearest-interior fallback within 4 voxels
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
                        (grid[ii, jj, kk] && isfinite(fill_time[ii, jj, kk])) || continue
                        vx = ox + (T(ii) - T(0.5)) * dx
                        vy = oy + (T(jj) - T(0.5)) * dy
                        vz = oz + (T(kk) - T(0.5)) * dz
                        d2 = (vx - p_in[1])^2 + (vy - p_in[2])^2 + (vz - p_in[3])^2
                        if d2 < best_d2
                            best_d2 = d2
                            t_val   = T(fill_time[ii, jj, kk])
                            u_raw   = voxel_gradient_at(ii, jj, kk, fill_time, grid, dx, dy, dz)
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

    # Pass 2 (Parallel): Surface divergence and defect masks using lock-free CSR adjacency
    tri_div        = zeros(T, n_tris)
    weld_vec       = Vector{Bool}(undef, n_tris)
    meld_vec       = Vector{Bool}(undef, n_tris)
    trap_vec       = Vector{Bool}(undef, n_tris)
    div_vec        = Vector{Bool}(undef, n_tris)

    w_thresh = T(weld_dot_thresh)
    m_thresh = T(meld_dot_thresh)
    d_thresh = T(div_thresh)

    Threads.@threads for idx in 1:n_tris
        @inbounds begin
            tc    = tri_time[idx]
            u_c   = tri_flow_dir[idx]
            c_c   = tri_centroids[idx]
            has_u = isfinite(tc) && norm(u_c) > T(1e-6)

            nbrs         = tri_neighbors(adj, idx)
            min_dot      = one(T)
            is_local_max = !isempty(nbrs)
            div_acc      = zero(T)
            n_nbrs       = 0

            for nbr_i32 in nbrs
                nbr = Int(nbr_i32)
                tn  = tri_time[nbr]
                isfinite(tn) || continue
                if tn > tc
                    is_local_max = false
                end

                u_n = tri_flow_dir[nbr]
                if has_u && norm(u_n) > T(1e-6)
                    d_un = dot(u_c, u_n)
                    if d_un < min_dot
                        min_dot = d_un
                    end

                    dc   = tri_centroids[nbr] - c_c
                    dist = norm(dc)
                    if dist > T(1e-8)
                        div_acc += dot(u_n - u_c, dc / dist) / dist
                        n_nbrs += 1
                    end
                end
            end

            trap_vec[idx] = has_u && is_local_max && tc > zero(T)
            is_weld       = has_u && (min_dot < w_thresh)
            weld_vec[idx] = is_weld
            meld_vec[idx] = has_u && !is_weld && (min_dot < m_thresh)

            if n_nbrs > 0
                d_val        = div_acc / T(n_nbrs)
                tri_div[idx] = d_val
                div_vec[idx] = has_u && (d_val > d_thresh)
            else
                div_vec[idx] = false
            end
        end
    end

    masks = (;
        weld_lines     = BitVector(weld_vec),
        meld_lines     = BitVector(meld_vec),
        air_traps      = BitVector(trap_vec),
        diverging_flow = BitVector(div_vec)
    )

    return (;
        centroids  = tri_centroids,
        normals    = tri_normals,
        fill_time  = tri_time,
        flow_dir   = tri_flow_dir,
        thickness  = tri_thickness,
        divergence = tri_div,
        masks
    )
end


"""
    compute_triangle_layered_fot(
        tri_fields;
        n_layers::Int = 10,
        core_exponent::Real = 2.0,
        div_scale::Real = 5.0,
        max_shear_tilt::Real = 0.08
    )

Multi-threaded computation of the 5 independent components `(a11, a22, a12, a13, a23)`
of the symmetric Fiber Orientation Tensor across `n_layers` through-thickness stations
from the midplane core (`ζ = 0`) to the mold wall skin (`ζ = 1`).
Use `get_a33(a11, a22)` to reconstruct `a33 = max(0, 1 - a11 - a22)`.

Returns:
- `zeta`: Vector of normalized layer depths `ζ ∈ [0, 1]` (length `n_layers`)
- `layers`: NamedTuple of `(n_tris × n_layers)` matrices `(a11, a22, a12, a13, a23)`
- `shell_avg`: NamedTuple of thickness-integrated `(n_tris,)` vectors `(a11, a22, a12, a13, a23)`
"""
function compute_triangle_layered_fot(
    tri_fields;
    n_layers::Int = 10,
    core_exponent::Real = 2.0,
    div_scale::Real = 5.0,
    max_shear_tilt::Real = 0.08
)
    flow_dirs = tri_fields.flow_dir
    normals   = tri_fields.normals
    divs      = tri_fields.divergence
    welds     = tri_fields.masks.weld_lines

    T = eltype(divs)
    n_tris = length(flow_dirs)
    @assert n_layers >= 1 "n_layers must be >= 1"

    zeta = [T(l - 0.5) / T(n_layers) for l in 1:n_layers]

    a11 = zeros(T, n_tris, n_layers)
    a22 = zeros(T, n_tris, n_layers)
    a12 = zeros(T, n_tris, n_layers)
    a13 = zeros(T, n_tris, n_layers)
    a23 = zeros(T, n_tris, n_layers)

    p_exp   = T(core_exponent)
    d_scale = T(div_scale)
    s_tilt  = T(max_shear_tilt)

    Threads.@threads for idx in 1:n_tris
        @inbounds begin
            e1 = flow_dirs[idx]
            e3 = normals[idx]

            if norm(e1) < T(1e-6)
                for l in 1:n_layers
                    a11[idx, l] = T(1) / T(3)
                    a22[idx, l] = T(1) / T(3)
                end
                continue
            end

            e2      = normalize(cross(e3, e1))
            is_weld = welds[idx]
            div_u   = divs[idx]

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

                a12[idx, l] = λ_flow * e1[1]*e1[2] + λ_trans * e2[1]*e2[2] + λ_norm * e3[1]*e3[2] +
                              λ_13 * (e1[1]*e3[2] + e3[1]*e1[2])
                a13[idx, l] = λ_flow * e1[1]*e1[3] + λ_trans * e2[1]*e2[3] + λ_norm * e3[1]*e3[3] +
                              λ_13 * (e1[1]*e3[3] + e3[1]*e1[3])
                a23[idx, l] = λ_flow * e1[2]*e1[3] + λ_trans * e2[2]*e2[3] + λ_norm * e3[2]*e3[3] +
                              λ_13 * (e1[2]*e3[3] + e3[2]*e3[3])
            end
        end
    end

    inv_L = inv(T(n_layers))
    shell_avg = (
        a11 = vec(sum(a11; dims = 2)) .* inv_L,
        a22 = vec(sum(a22; dims = 2)) .* inv_L,
        a12 = vec(sum(a12; dims = 2)) .* inv_L,
        a13 = vec(sum(a13; dims = 2)) .* inv_L,
        a23 = vec(sum(a23; dims = 2)) .* inv_L
    )

    layers = (; a11, a22, a12, a13, a23)
    return (; zeta, layers, shell_avg)
end

