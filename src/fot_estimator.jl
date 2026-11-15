

"""
    compute_divergence_and_weldlines(centroids, normals, thicknesses, tri_flow_dirs, neighbors)

Computes in-plane flux divergence κ = ∇_s ⋅ û + (û ⋅ ∇_s h)/h and detects weld-line 
triangles where opposing Eikonal wavefronts collide.
"""
function compute_divergence_and_weldlines(
    centroids::AbstractVector{<:StaticVector{3, T1}},
    normals::AbstractVector{<:StaticVector{3, T2}},
    thicknesses::AbstractVector{T3},
    tri_flow_dirs::AbstractVector{<:StaticVector{3, T4}},
    neighbors::AbstractVector{<:AbstractVector{<:Integer}}
) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}

    T = float(promote_type(T1, T2, T3, T4))
    Vec3 = SVector{3, T}
    Mat3 = SMatrix{3, 3, T, 9}

    n_tris = length(centroids)
    divergence = zeros(T, n_tris)
    is_weldline = falses(n_tris)

    min_h    = T(0.1)
    eps_norm = T(1e-6)
    eps_det  = T(1e-12)

    for i in 1:n_tris
        nbrs = neighbors[i]
        u_i = Vec3(tri_flow_dirs[i])
        if isempty(nbrs) || norm(u_i) < eps_norm
            continue
        end

        c_i = Vec3(centroids[i])
        n_i = Vec3(normals[i])
        h_i = max(T(thicknesses[i]), min_h)

        M = zero(Mat3)
        rhs_u = zero(Mat3)
        rhs_h = zero(Vec3)
        min_dot = one(T)

        for j in nbrs
            u_j = Vec3(tri_flow_dirs[j])
            norm(u_j) < eps_norm && continue

            d_ij = dot(u_i, u_j)
            if d_ij < min_dot
                min_dot = d_ij
            end

            dx = Vec3(centroids[j]) - c_i
            dx_tan = dx - dot(dx, n_i) * n_i
            dist_sq = dot(dx_tan, dx_tan)
            dist_sq > eps_det || continue

            w = inv(dist_sq)
            M += w * (dx_tan * dx_tan')

            u_j_tan = u_j - dot(u_j, n_i) * n_i
            un = norm(u_j_tan)
            u_j_tan = un > eps_norm ? (u_j_tan / un) : u_i

            rhs_u += w * (dx_tan * (u_j_tan - u_i)')
            rhs_h += w * dx_tan * (max(T(thicknesses[j]), min_h) - h_i)
        end

        if min_dot < T(-0.25)
            is_weldline[i] = true
        end

        M_reg = M + (n_i * n_i')
        if abs(det(M_reg)) > eps_det
            invM = inv(M_reg)
            grad_u = invM * rhs_u
            grad_h = invM * rhs_h
            divergence[i] = tr(grad_u) + dot(u_i, grad_h) / h_i
        end
    end

    return divergence, is_weldline
end



"""
    reconstruct_fot_layers(
        centroids, normals, thicknesses, tri_flow_dirs, neighbors;
        n_layers::Int = 11,
        core_exponent::Float32 = 2.0f0,
        fountain_skin_frac::Float32 = 0.08f0,
        div_scale::Float32 = 5.0f0
    )

Computes the 3D second-order Fiber Orientation Tensor (3x3 SMatrix) across `n_layers`
through-thickness points ζ ∈ [-1, 1] and the thickness-averaged FOT per triangle.
"""
function reconstruct_fot_layers(
    centroids::AbstractVector{<:StaticVector{3, T1}},
    normals::AbstractVector{<:StaticVector{3, T2}},
    thicknesses::AbstractVector{T3},
    tri_flow_dirs::AbstractVector{<:StaticVector{3, T4}},
    neighbors::AbstractVector{<:AbstractVector{<:Integer}};
    n_layers::Integer = 11,
    core_exponent::Real = 2.0,
    fountain_skin_frac::Real = 0.08,
    div_scale::Real = 5.0
) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}

    T = float(promote_type(T1, T2, T3, T4))

    n_tris = length(centroids)

    divergence, is_weldline = compute_divergence_and_weldlines(
        centroids, normals, thicknesses, tri_flow_dirs, neighbors
    )

    # Canonical local orientation tensors in frame (e1 = flow, e2 = transverse, e3 = normal)
    A_shear_pos = @SMatrix [
        0.84f0  0.00f0  0.06f0;
        0.00f0  0.12f0  0.00f0;
        0.06f0  0.00f0  0.04f0
    ]
    A_shear_neg = @SMatrix [
        0.84f0  0.00f0 -0.06f0;
        0.00f0  0.12f0  0.00f0;
       -0.06f0  0.00f0  0.04f0
    ]
    A_core_conv = @SMatrix [
        0.72f0  0.00f0  0.00f0;
        0.00f0  0.20f0  0.00f0;
        0.00f0  0.00f0  0.08f0
    ]
    A_core_div = @SMatrix [
        0.18f0  0.00f0  0.00f0;
        0.00f0  0.74f0  0.00f0;
        0.00f0  0.00f0  0.08f0
    ]
    A_fountain = @SMatrix [
        0.44f0  0.00f0  0.00f0;
        0.00f0  0.44f0  0.00f0;
        0.00f0  0.00f0  0.12f0
    ]
    # Weld-line tensor: fibers align parallel to weld front (e2)
    A_weldline = @SMatrix [
        0.15f0  0.00f0  0.00f0;
        0.00f0  0.78f0  0.00f0;
        0.00f0  0.00f0  0.07f0
    ]
    A_iso = @SMatrix [
        1f0/3f0 0f0     0f0;
        0f0     1f0/3f0 0f0;
        0f0     0f0     1f0/3f0
    ]

    # Normalized through-thickness coordinates ζ ∈ [-1, 1]
    
    zetas = n_layers <=1 ? [zero(T)] : collect(range(T(-1), T(1), length = n_layers))

    fot_layers = Matrix{SMatrix{3, 3, T, 9}}(undef, n_tris, n_layers)
    fot_avg    = Vector{SMatrix{3, 3, T, 9}}(undef, n_tris)

    for i in 1:n_tris
        n_hat = normals[i]
        u_raw = tri_flow_dirs[i]

        # Build orthonormal frame (e1, e2, e3)
        if norm(u_raw) < 1f-6
            # Fallback for stagnation zones
            ref = abs(n_hat[1]) < 0.9f0 ? SVector(1f0, 0f0, 0f0) : SVector(0f0, 1f0, 0f0)
            e1 = normalize(cross(n_hat, ref))
            e2 = cross(n_hat, e1)
            R = hcat(e1, e2, n_hat)
            for k in 1:n_layers
                fot_layers[i, k] = A_iso
            end
            fot_avg[i] = A_iso
            continue
        end

        e1 = normalize(u_raw - dot(u_raw, n_hat) * n_hat)
        e2 = cross(n_hat, e1)
        e3 = n_hat
        R = hcat(e1, e2, e3)

        # Determine local core tensor from in-plane divergence (or weld-line override)
        if is_weldline[i]
            A_core = A_weldline
        else
            # Map divergence κ ∈ (-∞, +∞) smoothly into α_div ∈ [0, 1]
            # κ > 0 (diverging) -> α_div -> 1 (transverse core)
            # κ < 0 (converging) -> α_div -> 0 (flow-aligned core)
            α_div = 0.5f0 * (1.0f0 + tanh(divergence[i] * div_scale))
            A_core = (1.0f0 - α_div) * A_core_conv + α_div * A_core_div
        end

        A_sum = @SMatrix zeros(Float32, 3, 3)

        for (k, ζ) in enumerate(zetas)
            abs_ζ = abs(ζ)

            if is_weldline[i]
                A_local = A_weldline
            else
                # Shear weight increases from 0 at midplane (ζ=0) to 1 near walls (|ζ|→1)
                w_shear = abs_ζ^core_exponent
                A_shear = ζ >= 0.0f0 ? A_shear_pos : A_shear_neg
                A_bulk  = (1.0f0 - w_shear) * A_core + w_shear * A_shear

                # Blend thin fountain-flow frozen skin right at the mold walls (|ζ| > 1 - fountain_skin_frac)
                if fountain_skin_frac > 0.0f0 && abs_ζ > (1.0f0 - fountain_skin_frac)
                    w_fountain = (abs_ζ - (1.0f0 - fountain_skin_frac)) / fountain_skin_frac
                    A_local = (1.0f0 - w_fountain) * A_bulk + w_fountain * A_fountain
                else
                    A_local = A_bulk
                end
            end

            # Rotate local orientation tensor into global 3D coordinates: A_global = R * A_local * Rᵀ
            A_global = R * A_local * R'
            fot_layers[i, k] = A_global
            A_sum += A_global
        end

        fot_avg[i] = A_sum / Float32(n_layers)
    end

    return fot_layers, fot_avg, is_weldline, divergence
end



# 3. Compute Local Thickness and Mid-Surface via Raycore Ray-Casting
function compute_thickness_field(tlasdata)
    
    mesh, tlas = tlasdata

    # thicknesses = zeros(Float32, size(nodes, 2))
    thicknesses = zeros(Float32, length(mesh.position))
    midpoints = zeros(eltype(mesh.position),length(mesh.position))
    

    for i in eachindex(mesh.position)
        pos = mesh.position[i]
        normal = mesh.normal[i]
        ray_dir = -normal
        ray = Raycore.Ray(o = pos - 1f-3 * normal,d =  ray_dir)
        # Query Raycore BVH for closest hit
        # result_closest = Raycore.closest_hit(tlas, ray)
        hit_found, triangle, distance, bary, instance_idx = Raycore.closest_hit(tlas, ray)
        # @info result_closest
        # return result_closest
        if hit_found && distance > 1f-4
            thicknesses[i] = distance
            midpoints[i] = pos .+ (ray_dir * (distance / 2.0f0))
        else
            thicknesses[i] = 0.0f0
            midpoints[i] = pos
        end

    end
    
    
    return (;thicknesses, midpoints)
end

function compute_thickness_field_by_face(tlasdata)
    
    mesh, tlas = tlasdata

    # thicknesses = zeros(Float32, size(nodes, 2))
    thicknesses = zeros(Float32, length(mesh.faces))
    midpoints = zeros(eltype(mesh.position),length(mesh.faces))
    

    for i in eachindex(mesh.faces)
        face = mesh.faces[i]
        i1, i2, i3 = face
        V1, V2, V3 = mesh.position[i1], mesh.position[i2], mesh.position[i3]
        pos = 1/3 * (V1 + V2 + V3)
        N1, N2, N3 = mesh.normal[i1], mesh.normal[i2], mesh.normal[i3]
        normal = 1/3 * (N1 + N2 + N3)
        ray_dir = -normal
        ray = Raycore.Ray(o = pos - 1f-3 * normal,d =  ray_dir)
        # Query Raycore BVH for closest hit
        # result_closest = Raycore.closest_hit(tlas, ray)
        hit_found, triangle, distance, bary, instance_idx = Raycore.closest_hit(tlas, ray)
        # @info result_closest
        # return result_closest
        if hit_found && distance > 1f-4
            thicknesses[i] = distance
            midpoints[i] = pos .+ (ray_dir * (distance / 2.0f0))
        else
            thicknesses[i] = 0.0f0
            midpoints[i] = pos
        end

    end
    
    
    return (;thicknesses, midpoints)
end




"""
    build_cone_directions(n::SVector{3, Float32})

Generates a deterministic 17-ray cone pointing into the interior hemisphere (-n):
1 axial ray along -n, plus 2 rings of 8 rays at 30° and 60° off-axis.
"""
function build_cone_directions(n::SVector{3, Float32})
    inward = -n
    # Construct an orthonormal basis (t1, t2, inward)
    ref = abs(inward[1]) < 0.9f0 ? SVector(1.0f0, 0.0f0, 0.0f0) : SVector(0.0f0, 1.0f0, 0.0f0)
    t1 = normalize(cross(inward, ref))
    t2 = cross(inward, t1)

    dirs = MVector{17, SVector{3, Float32}}(undef)
    dirs[1] = inward

    idx = 2
    for tilt_deg in (30.0f0, 60.0f0)
        θ = deg2rad(tilt_deg)
        sin_θ, cos_θ = sincos(θ)
        for k in 0:7
            ϕ = Float32(k) * (2.0f0 * Float32(π) / 8.0f0)
            sin_ϕ, cos_ϕ = sincos(ϕ)
            dirs[idx] = normalize(cos_θ * inward + sin_θ * (cos_ϕ * t1 + sin_ϕ * t2))
            idx += 1
        end
    end
    return SVector(dirs)
end

"""
    compute_triangle_thicknesses(bvh, points, elements; eps_offset=1e-4f0)

Computes local wall thickness and mid-surface points at every triangle centroid
using the Shrinking Tangent Sphere method via Raycore.
"""
function compute_triangle_thicknesses(meshdata; eps_offset::Float32 = 1f-4)
    # bvh, points::Vector{Point3f}, elements::AbstractMatrix
    bvh, mesh = meshdata.tlas, meshdata.mesh

    n_tris = length(mesh.faces)
    thicknesses = zeros(Float32, n_tris)
    midpoints   = Vector{Point3f}(undef, n_tris)
    centroids   = Vector{Point3f}(undef, n_tris)
    normals     = Vector{SVector{3, Float32}}(undef, n_tris)

    for i in 1:n_tris
        i1, i2, i3 = mesh.faces[i]
        p1 = mesh.position[i1]
        p2 = mesh.position[i2]
        p3 = mesh.position[i3]

        c = (p1 + p2 + p3) / 3.0f0
        raw_n = cross(p2 - p1, p3 - p1)
        norm_len = norm(raw_n)

        # Guard against degenerate zero-area triangles
        if norm_len < 1f-12
            centroids[i]   = Point3f(c)
            normals[i]     = SVector(0.0f0, 0.0f0, 1.0f0)
            midpoints[i]   = Point3f(c)
            thicknesses[i] = 0.0f0
            continue
        end

        n = raw_n / norm_len
        centroids[i] = c
        normals[i]   = n

        # Ensure Gmsh outward normals point out of the solid;
        # offset origin slightly inward along -n to avoid self-hitting triangle i
        origin = c - n * eps_offset
        cone_dirs = build_cone_directions(n)

        min_diameter = Inf32

        for dir in cone_dirs
            ray = Raycore.Ray(o = origin, d = dir)
            hit_found, _, dist, _, _ = Raycore.closest_hit(bvh, ray)

            if hit_found && dist > eps_offset
                # Reconstruct hit point q in 3D space
                q = SVector{3, Float32}(origin) + dir * dist
                pq = c - q
                denom = dot(pq, n)

                # Only consider hits strictly behind the tangent plane (denom > 0)
                if denom > 1f-7
                    # Exact diameter of sphere tangent to (c, n) passing through q
                    d_sphere = dot(pq, pq) / denom
                    if d_sphere < min_diameter
                        min_diameter = d_sphere
                    end
                end
            end
        end

        if isfinite(min_diameter)
            thicknesses[i] = min_diameter
            midpoints[i]   = Point3f(c - n * (0.5f0 * min_diameter))
        else
            thicknesses[i] = 0.0f0
            midpoints[i]   = Point3f(c)
        end
    end

    return (;thicknesses, midpoints, centroids, normals)
end