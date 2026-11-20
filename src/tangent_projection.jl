

"""
    project_voxel_to_mesh_triangles(
        meshdata, voxeldata, T_field, thick_r_sq;
        inward_offset_vox = 0.75,
        weld_dot_thresh = -0.3,
        meld_dot_thresh = 0.4,
        div_thresh = 0.05
    )

Maps Eikonal arrival time `T_field`, tangent-projected unit flow directions `e1`,
local physical wall thickness `h`, in-plane flow divergence `div_u`, and surface defect
masks (`weld_lines`, `meld_lines`, `air_traps`) onto every triangle of `meshdata.mesh`.
"""
function project_voxel_to_mesh_triangles(
    meshdata,
    voxeldata,
    T_field::AbstractArray{T1, 3},
    thick_r_sq::AbstractArray{UInt16, 3};
    inward_offset_vox::Real = 0.75,
    weld_dot_thresh::Real = -0.3,
    meld_dot_thresh::Real = 0.4,
    div_thresh::Real = 0.05
) where {T1<:Real}

    mesh = meshdata.mesh
    pts  = GeometryBasics.coordinates(mesh)
    tris = GeometryBasics.faces(mesh)
    n_tris = length(tris)

    grid = voxeldata.grid
    spacing_raw = hasproperty(voxeldata, :voxel_spacing) ? voxeldata.voxel_spacing : voxeldata.stepsize
    corner_raw  = voxeldata.corner

    T = float(promote_type(T1, eltype(spacing_raw), eltype(corner_raw)))
    nx, ny, nz = size(grid)
    dx, dy, dz = T(spacing_raw[1]), T(spacing_raw[2]), T(spacing_raw[3])
    ox, oy, oz = T(corner_raw[1]),  T(corner_raw[2]),  T(corner_raw[3])
    dx_mean    = (dx + dy + dz) / T(3)
    offset_dist = T(inward_offset_vox) * dx_mean

    # 1. Precompute smooth interior 3D flow gradient ∇T and divergence on valid voxels
    flow_dirs = fill(zero(SVector{3, T}), nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        (grid[i, j, k] && isfinite(T_field[i, j, k])) || continue
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
        if gn > T(1e-9)
            flow_dirs[i, j, k] = g / gn
        end
    end

    # Helper: mask-aware trilinear lookup at a 3D point with nearest-interior fallback
    function sample_triangle_fields(p_query::SVector{3, T})
        fx = (p_query[1] - ox) / dx + T(0.5)
        fy = (p_query[2] - oy) / dy + T(0.5)
        fz = (p_query[3] - oz) / dz + T(0.5)

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

        @inbounds for dk in 0:1, dj in 0:1, di in 0:1
            ii, jj, kk = i0 + di, j0 + dj, k0 + dk
            (grid[ii, jj, kk] && isfinite(T_field[ii, jj, kk])) || continue
            w = (di == 0 ? wx0 : wx1) * (dj == 0 ? wy0 : wy1) * (dk == 0 ? wz0 : wz1)
            t_acc   += w * T(T_field[ii, jj, kk])
            u_acc   += w * flow_dirs[ii, jj, kk]
            rsq_acc += w * T(thick_r_sq[ii, jj, kk])
            w_acc   += w
        end

        if w_acc > T(1e-7)
            return (t_acc / w_acc, u_acc / w_acc, rsq_acc / w_acc)
        end

        # Fallback: snap to nearest interior voxel within 4 voxels
        ic = clamp(round(Int, fx), 1, nx)
        jc = clamp(round(Int, fy), 1, ny)
        kc = clamp(round(Int, fz), 1, nz)
        best_d2 = T(Inf)
        best_res = (T(Inf), zero(SVector{3, T}), zero(T))

        @inbounds for r in 1:4
            for kk in max(1, kc - r):min(nz, kc + r),
                jj in max(1, jc - r):min(ny, jc + r),
                ii in max(1, ic - r):min(nx, ic + r)
                (grid[ii, jj, kk] && isfinite(T_field[ii, jj, kk])) || continue
                vx = ox + (T(ii) - T(0.5)) * dx
                vy = oy + (T(jj) - T(0.5)) * dy
                vz = oz + (T(kk) - T(0.5)) * dz
                d2 = (vx - p_query[1])^2 + (vy - p_query[2])^2 + (vz - p_query[3])^2
                if d2 < best_d2
                    best_d2 = d2
                    best_res = (T(T_field[ii, jj, kk]), flow_dirs[ii, jj, kk], T(thick_r_sq[ii, jj, kk]))
                end
            end
            isfinite(best_d2) && return best_res
        end
        return best_res
    end

    # 2. Allocate compact per-triangle arrays
    tri_centroids = Vector{SVector{3, T}}(undef, n_tris)
    tri_normals   = Vector{SVector{3, T}}(undef, n_tris)
    tri_time      = Vector{T}(undef, n_tris)
    tri_flow_dir  = Vector{SVector{3, T}}(undef, n_tris)
    tri_thickness = Vector{T}(undef, n_tris)
    tri_div       = zeros(T, n_tris)

    @inbounds for idx in 1:n_tris
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

        # Sample slightly inside the wall along -n_hat to capture midplane-representative flow
        p_in = c - offset_dist * n_hat
        t_val, u_raw, rsq_val = sample_triangle_fields(p_in)

        tri_time[idx]      = t_val
        tri_thickness[idx] = T(2) * sqrt(max(rsq_val, zero(T))) * dx_mean

        # Project flow direction onto the triangle tangent plane (orthogonal to n_hat)
        u_tan = u_raw - dot(u_raw, n_hat) * n_hat
        tri_flow_dir[idx] = norm(u_tan) > T(1e-6) ? normalize(u_tan) : zero(SVector{3, T})
    end

    # 3. Build triangle-to-triangle edge adjacency to compute surface divergence & weld lines
    edge_map = Dict{Tuple{Int, Int}, Int}()
    adj = [Int[] for _ in 1:n_tris]

    @inbounds for idx in 1:n_tris
        tri = tris[idx]
        v = (Int(tri[1]), Int(tri[2]), Int(tri[3]))
        edges = (
            (min(v[1], v[2]), max(v[1], v[2])),
            (min(v[2], v[3]), max(v[2], v[3])),
            (min(v[3], v[1]), max(v[3], v[1]))
        )
        for e in edges
            other = get(edge_map, e, 0)
            if other == 0
                edge_map[e] = idx
            else
                push!(adj[idx], other)
                push!(adj[other], idx)
            end
        end
    end

    weld_lines     = falses(n_tris)
    meld_lines     = falses(n_tris)
    air_traps      = falses(n_tris)
    diverging_flow = falses(n_tris)

    @inbounds for idx in 1:n_tris
        tc = tri_time[idx]
        isfinite(tc) || continue
        u_c = tri_flow_dir[idx]
        c_c = tri_centroids[idx]
        has_u = norm(u_c) > T(1e-6)

        min_dot = one(T)
        is_local_max = !isempty(adj[idx])
        div_acc = zero(T)
        n_nbrs = 0

        for nbr in adj[idx]
            tn = tri_time[nbr]
            isfinite(tn) || continue
            if tn > tc
                is_local_max = false
            end

            u_n = tri_flow_dir[nbr]
            if has_u && norm(u_n) > T(1e-6)
                d_un = dot(u_c, u_n)
                min_dot = min(min_dot, d_un)

                dc = tri_centroids[nbr] - c_c
                dist = norm(dc)
                if dist > T(1e-8)
                    # Surface directional derivative of unit flow vector: (u_n - u_c) ⋅ (dc / |dc|)
                    div_acc += dot(u_n - u_c, dc / dist) / dist
                    n_nbrs += 1
                end
            end
        end

        if is_local_max && tc > zero(T)
            air_traps[idx] = true
        end
        if has_u
            if min_dot < T(weld_dot_thresh)
                weld_lines[idx] = true
            elseif min_dot < T(meld_dot_thresh)
                meld_lines[idx] = true
            end
        end
        if n_nbrs > 0
            d_val = div_acc / T(n_nbrs)
            tri_div[idx] = d_val
            if d_val > T(div_thresh)
                diverging_flow[idx] = true
            end
        end
    end

    masks = (; weld_lines, meld_lines, air_traps, diverging_flow)
    return (;
        centroids = tri_centroids,
        normals   = tri_normals,
        fill_time = tri_time,
        flow_dir  = tri_flow_dir,
        thickness = tri_thickness,
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

Computes the 6-component symmetric Fiber Orientation Tensor `(a11, a22, a33, a12, a13, a23)`
for every surface triangle across `n_layers` through-thickness stations from the midplane
core (`ζ = 0`) to the mold wall skin (`ζ = 1`).

Returns:
- `zeta`: Vector of normalized layer depths `ζ ∈ [0, 1]` (length `n_layers`)
- `layers`: NamedTuple of `(n_tris × n_layers)` matrices `(a11, a22, a33, a12, a13, a23)`
- `shell_avg`: NamedTuple of thickness-integrated `(n_tris,)` vectors `(a11, a22, a33, a12, a13, a23)`
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

    # Midpoint quadrature stations ζ_l ∈ (0, 1) from core (0) to wall skin (1)
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

    @inbounds for idx in 1:n_tris
        e1 = flow_dirs[idx]
        e3 = normals[idx]

        if norm(e1) < T(1e-6)
            # Isotropic fallback (1/3 * I) for unreached/stagnant triangles
            for l in 1:n_layers
                a11[idx, l] = T(1) / T(3)
                a22[idx, l] = T(1) / T(3)
                a33[idx, l] = T(1) / T(3)
            end
            continue
        end

        # Orthonormal in-plane transverse vector e2 = e3 × e1
        e2 = normalize(cross(e3, e1))
        is_weld = welds[idx]
        div_u   = divs[idx]

        # Precompute core orientation eigenvalues for this triangle
        α_div   = T(0.5) * (one(T) + tanh(div_u * d_scale))
        c_flow  = (one(T) - α_div) * T(0.72) + α_div * T(0.18)
        c_trans = (one(T) - α_div) * T(0.20) + α_div * T(0.74)
        c_norm  = T(0.08)

        # Skin shear orientation eigenvalues
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

            # Rotate local symmetric tensor diag/shear into global (X, Y, Z) coordinates
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
                          λ_13 * (e1[2]*e3[3] + e3[2]*e3[3])
        end
    end

    # Thickness-integrated (shell-averaged) FOT per triangle
    inv_L = inv(T(n_layers))
    shell_avg = (
        a11 = vec(sum(a11; dims = 2)) .* inv_L,
        a22 = vec(sum(a22; dims = 2)) .* inv_L,
        a33 = vec(sum(a33; dims = 2)) .* inv_L,
        a12 = vec(sum(a12; dims = 2)) .* inv_L,
        a13 = vec(sum(a13; dims = 2)) .* inv_L,
        a23 = vec(sum(a23; dims = 2)) .* inv_L
    )

    layers = (; a11, a22, a33, a12, a13, a23)
    return (; zeta, layers, shell_avg)
end