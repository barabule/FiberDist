

struct HeatMethodSolver
    F_heat::SuiteSparse.CHOLMOD.Factor{Float64}
    F_poisson::SuiteSparse.CHOLMOD.Factor{Float64}
    G::SparseMatrixCSC{Float64, Int}
    D::SparseMatrixCSC{Float64, Int}
    nv::Int
    nf::Int
end

function build_operators(V::Matrix{Float64}, F::Matrix{Int})
    nv = size(V, 2)
    nf = size(F, 2)
    
    # Edge lengths for time step t = mean(edge_length)^2
    edge_lengths = Float64[]
    
    # Triplets for Mass (A), Cotan Laplacian (Lc), and Gradient (G)
    I_L, J_L, V_L = Int[], Int[], Float64[]
    I_G, J_G, V_G = Int[], Int[], Float64[]
    vertex_areas = zeros(Float64, nv)
    face_areas = zeros(Float64, nf)
    
    for f in 1:nf
        i, j, k = F[1, f], F[2, f], F[3, f]
        vi, vj, vk = V[:, i], V[:, j], V[:, k]
        
        e_ij = vj - vi
        e_jk = vk - vj
        e_ki = vi - vk
        
        push!(edge_lengths, norm(e_ij), norm(e_jk), norm(e_ki))
        
        # Face normal and area
        n = cross(e_ij, -e_ki)
        dbl_area = norm(n)
        area = 0.5 * dbl_area
        face_areas[f] = area
        n_unit = n / dbl_area
        
        # Barycentric vertex area accumulation
        vertex_areas[i] += area / 3.0
        vertex_areas[j] += area / 3.0
        vertex_areas[k] += area / 3.0
        
        # Cotangents of angles at i, j, k
        cot_i = dot(e_ij, -e_ki) / dbl_area
        cot_j = dot(e_jk, -e_ij) / dbl_area
        cot_k = dot(e_ki, -e_jk) / dbl_area
        
        # Assemble negative-semidefinite cotan Laplacian Lc
        for (u, v, c) in ((i, j, cot_k), (j, k, cot_i), (k, i, cot_j))
            w = 0.5 * c
            push!(I_L, u, v, u, v)
            push!(J_L, v, u, u, v)
            push!(V_L, w, w, -w, -w)
        end
        
        # Per-face gradient operator G (3*nf x nv)
        # grad(u) = (1 / 2A) * sum_v u_v * (n_unit x opposite_edge)
        for (v_idx, opp_edge) in ((i, e_jk), (j, e_ki), (k, e_ij))
            g_vec = cross(n_unit, opp_edge) / dbl_area
            for d in 1:3
                push!(I_G, 3*(f - 1) + d)
                push!(J_G, v_idx)
                push!(V_G, g_vec[d])
            end
        end
    end
    
    A = spdiagm(0 => vertex_areas)
    Lc = sparse(I_L, J_L, V_L, nv, nv)
    G = sparse(I_G, J_G, V_G, 3 * nf, nv)
    
    # Integrated divergence operator D = -G' * M_f (where M_f repeats face areas 3 times)
    M_f = spdiagm(0 => repeat(face_areas, inner=3))
    D = -G' * M_f
    
    h = sum(edge_lengths) / length(edge_lengths)
    t = h^2
    
    return A, Lc, G, D, t
end

function HeatMethodSolver(V::Matrix{Float64}, F::Matrix{Int})
    A, Lc, G, D, t = build_operators(V, F)
    nv, nf = size(V, 2), size(F, 2)
    
    # Pre-factorize symmetric positive-definite systems
    # Add tiny regularization to -Lc to pin down the constant nullspace
    F_heat = cholesky(Symmetric(A - t * Lc))
    F_poisson = cholesky(Symmetric(-Lc + 1e-10 * I))
    
    return HeatMethodSolver(F_heat, F_poisson, G, D, nv, nf)
end

function compute_geodesic(solver::HeatMethodSolver, sources::Vector{Int})
    # 1. Solve heat flow (A - t*Lc) u = delta
    rhs_heat = zeros(Float64, solver.nv)
    rhs_heat[sources] .= 1.0
    u = solver.F_heat \ rhs_heat
    
    # 2. Evaluate and normalize face gradients X = -grad(u) / |grad(u)|
    grad_u = reshape(solver.G * u, 3, solver.nf)
    X = similar(grad_u)
    for f in 1:solver.nf
        g = view(grad_u, :, f)
        nrm = norm(g)
        X[:, f] = nrm > 1e-12 ? (-g / nrm) : zeros(3)
    end
    
    # 3. Solve Poisson equation (-Lc) phi = -div(X)
    rhs_poisson = -(solver.D * vec(X))
    phi = solver.F_poisson \ rhs_poisson
    
    # Shift so minimum distance at sources is zero
    phi .-= minimum(phi[sources])
    return phi
end