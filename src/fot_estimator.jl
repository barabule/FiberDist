

"""
    compute_divergence_and_weldlines(centroids, normals, thicknesses, tri_flow_dirs, neighbors)

Computes in-plane flux divergence κ = ∇_s ⋅ û + (û ⋅ ∇_s h)/h and detects weld-line 
triangles where opposing Eikonal wavefronts collide.
"""
function compute_divergence_and_weldlines(
    centroids::Vector{Point3f},
    normals::Vector{SVector{3, Float32}},
    thicknesses::Vector{Float32},
    tri_flow_dirs::Vector{SVector{3, Float32}},
    neighbors::Vector{Vector{Int}}
)
    n_tris = length(centroids)
    divergence = zeros(Float32, n_tris)
    is_weldline = falses(n_tris)

    for i in 1:n_tris
        nbrs = neighbors[i]
        u_i = tri_flow_dirs[i]
        if isempty(nbrs) || norm(u_i) < 1f-6
            continue
        end

        c_i = SVector{3, Float32}(centroids[i])
        n_i = normals[i]
        h_i = max(thicknesses[i], 0.1f0)

        # Least-squares tangential gradient of û and h over dual-graph neighbors
        # Solve min Σ || (Δx_tan) * J - Δu_tan ||^2
        M = @SMatrix zeros(Float32, 3, 3)
        rhs_u = @SMatrix zeros(Float32, 3, 3)
        rhs_h = SVector(0.0f0, 0.0f0, 0.0f0)
        min_dot = 1.0f0

        for j in nbrs
            u_j = tri_flow_dirs[j]
            norm(u_j) < 1f-6 && continue

            # Check for colliding wavefronts (opposing flow directions)
            d_ij = dot(u_i, u_j)
            if d_ij < min_dot
                min_dot = d_ij
            end

            dx = SVector{3, Float32}(centroids[j]) - c_i
            # Project edge vector onto tangent plane of triangle i
            dx_tan = dx - dot(dx, n_i) * n_i
            dist_sq = dot(dx_tan, dx_tan)
            dist_sq > 1f-12 || continue

            w = 1.0f0 / dist_sq
            M += w * (dx_tan * dx_tan')

            # Parallel-transport / project u_j onto tangent plane of triangle i
            u_j_tan = u_j - dot(u_j, n_i) * n_i
            un = norm(u_j_tan)
            u_j_tan = un > 1f-6 ? (u_j_tan / un) : u_i

            rhs_u += w * (dx_tan * (u_j_tan - u_i)')
            rhs_h += w * dx_tan * (max(thicknesses[j], 0.1f0) - h_i)
        end

        # Flag weld line if opposing fronts collide across shared edge
        if min_dot < -0.25f0
            is_weldline[i] = true
        end

        # Regularized pseudo-inverse in the 2D tangent plane (adding n_i * n_i' makes M full-rank 3x3)
        M_reg = M + (n_i * n_i')
        if abs(det(M_reg)) > 1f-12
            invM = inv(M_reg)
            grad_u = invM * rhs_u          # 3x3 tangential Jacobian ∂u_β / ∂x_α
            grad_h = invM * rhs_h          # 3x1 tangential thickness gradient ∇_s h

            # Surface divergence tr(grad_u) + (u_i ⋅ ∇_s h) / h_i
            div_u = tr(grad_u)
            div_h = dot(u_i, grad_h) / h_i
            divergence[i] = div_u + div_h
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
    centroids::Vector{Point3f},
    normals::Vector{SVector{3, Float32}},
    thicknesses::Vector{Float32},
    tri_flow_dirs::Vector{SVector{3, Float32}},
    neighbors::Vector{Vector{Int}};
    n_layers::Int = 11,
    core_exponent::Float32 = 2.0f0,
    fountain_skin_frac::Float32 = 0.08f0,
    div_scale::Float32 = 5.0f0
)
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
    zetas = n_layers == 1 ? [0.0f0] : collect(range(-1.0f0, 1.0f0, length = n_layers))

    fot_layers = Matrix{SMatrix{3, 3, Float32, 9}}(undef, n_tris, n_layers)
    fot_avg    = Vector{SMatrix{3, 3, Float32, 9}}(undef, n_tris)

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