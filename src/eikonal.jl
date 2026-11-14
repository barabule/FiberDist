

function solve_eikonal_dual_graph(
    centroids::Vector{Point3f},
    thicknesses::Vector{Float32},
    normals::Vector{SVector{3, Float32}},
    elements::AbstractMatrix,
    gates::Vector{ValveGate};
    min_thickness::Float32 = 0.1f0
)
    n_tris = length(centroids)
    c_vecs = [SVector{3, Float32}(c) for c in centroids]
    neighbors = build_dual_graph(elements)

    # Hele-Shaw speed function F = h^2
    speeds = [max(h, min_thickness)^2 for h in thicknesses]

    T = fill(Inf32, n_tris)
    accepted = falses(n_tris)
    pq = PriorityQueue{Int, Float32}()

    # Seed each valve gate with its respective opening time t_open >= 0
    for gate in gates
        @assert gate.open_time >= 0.0f0 "Gate opening time must be >= 0"
        g_vec = SVector{3, Float32}(gate.position)

        # Primary skin triangle closest to gate
        gate_tri = argmin([norm(c - g_vec) for c in c_vecs])
        if gate.open_time < T[gate_tri]
            T[gate_tri] = gate.open_time
            pq[gate_tri] = gate.open_time
        end

        # Opposite-skin partner triangle so both sides of a dual-skin mesh open together
        opp_target = c_vecs[gate_tri] - thicknesses[gate_tri] * normals[gate_tri]
        opp_tri = argmin([
            (i == gate_tri || dot(normals[i], normals[gate_tri]) > -0.5f0) ? Inf32 : norm(c_vecs[i] - opp_target)
            for i in 1:n_tris
        ])
        if opp_tri != gate_tri && gate.open_time < T[opp_tri]
            T[opp_tri] = gate.open_time
            pq[opp_tri] = gate.open_time
        end
    end

    # Fast Marching propagation loop (unchanged!)
    while !isempty(pq)
        curr_tri, curr_t = dequeue_pair!(pq)
        accepted[curr_tri] && continue
        accepted[curr_tri] = true

        for nbr in neighbors[curr_tri]
            accepted[nbr] && continue

            F_eff = 2.0f0 * speeds[curr_tri] * speeds[nbr] / (speeds[curr_tri] + speeds[nbr])
            t_cand = curr_t + norm(c_vecs[nbr] - c_vecs[curr_tri]) / F_eff

            for other_nbr in neighbors[nbr]
                if accepted[other_nbr] && other_nbr != curr_tri
                    t_2d = local_upwind_update(
                        c_vecs[nbr], F_eff,
                        c_vecs[curr_tri], curr_t,
                        c_vecs[other_nbr], T[other_nbr]
                    )
                    if t_2d < t_cand
                        t_cand = t_2d
                    end
                end
            end

            if t_cand < T[nbr]
                T[nbr] = t_cand
                pq[nbr] = t_cand
            end
        end
    end

    # Reconstruct in-plane unit flow vectors û = ∇T / |∇T|
    flow_dirs = Vector{SVector{3, Float32}}(undef, n_tris)
    for i in 1:n_tris
        nbrs = neighbors[i]
        if isempty(nbrs) || !isfinite(T[i])
            flow_dirs[i] = SVector(0.0f0, 0.0f0, 0.0f0)
            continue
        end

        grad = SVector(0.0f0, 0.0f0, 0.0f0)
        for j in nbrs
            isfinite(T[j]) || continue
            dx = c_vecs[j] - c_vecs[i]
            dist_sq = dot(dx, dx)
            dist_sq > 1f-12 || continue
            grad += ((T[j] - T[i]) / dist_sq) * dx
        end

        g_norm = norm(grad)
        flow_dirs[i] = g_norm > 1f-9 ? (grad / g_norm) : SVector(0.0f0, 0.0f0, 0.0f0)
    end

    return T, flow_dirs, neighbors
end


"""
    build_dual_graph(elements::AbstractMatrix)

Builds triangle-to-triangle adjacency lists from a 3×N_tris connectivity matrix.
Two triangles are neighbors if they share at least 2 vertices (an edge).
"""
function build_dual_graph(elements::AbstractMatrix)
    n_tris = size(elements, 2)
    # Map sorted vertex pairs (u, v) -> list of incident triangle indices
    edge_to_tris = Dict{Tuple{Int, Int}, Vector{Int}}()
    sizehint!(edge_to_tris, 3 * n_tris ÷ 2)

    for t in 1:n_tris
        v1, v2, v3 = Int(elements[1, t]), Int(elements[2, t]), Int(elements[3, t])
        for (a, b) in ((v1, v2), (v2, v3), (v3, v1))
            edge = a < b ? (a, b) : (b, a)
            push!(get!(edge_to_tris, edge, Int[]), t)
        end
    end

    neighbors = [Int[] for _ in 1:n_tris]
    for tris in values(edge_to_tris)
        if length(tris) >= 2
            for i in 1:length(tris), j in (i+1):length(tris)
                t1, t2 = tris[i], tris[j]
                push!(neighbors[t1], t2)
                push!(neighbors[t2], t1)
            end
        end
    end
    return neighbors
end

"""
    local_upwind_update(c_k, F_k, c_i, T_i, c_j, T_j)

Solves the local 2D Eikonal equation |∇T| = 1/F_k on the triangle spanned by 
target centroid c_k and two accepted neighbor centroids (c_i, c_j) with known 
arrival times (T_i, T_j). Falls back to 1D edge update if the characteristic ray 
lies outside the simplex (c_i, c_j).
"""
function local_upwind_update(
    c_k::SVector{3, Float32}, F_k::Float32,
    c_i::SVector{3, Float32}, T_i::Float32,
    c_j::SVector{3, Float32}, T_j::Float32
)
    # 1D fallback updates along each edge
    t_1d_i = T_i + norm(c_k - c_i) / F_k
    t_1d_j = T_j + norm(c_k - c_j) / F_k
    t_best = min(t_1d_i, t_1d_j)

    # 2D planar upwind update across segment (c_i, c_j)
    e_ij = c_j - c_i
    L_ij = norm(e_ij)
    L_ij < 1f-9 && return t_best

    dT = T_j - T_i
    abs(dT) >= L_ij / F_k && return t_best  # Characteristic lies outside segment

    # Orthonormal basis in the plane of (c_i, c_j, c_k)
    u_hat = e_ij / L_ij
    v_vec = (c_k - c_i) - dot(c_k - c_i, u_hat) * u_hat
    h_perp = norm(v_vec)
    h_perp < 1f-9 && return t_best
    v_hat = v_vec / h_perp

    # Coordinates of c_k relative to c_i in (u_hat, v_hat)
    x_k = dot(c_k - c_i, u_hat)
    y_k = h_perp

    # Wavefront slowness components: s_u = dT / L_ij, s_v = sqrt((1/F_k)^2 - s_u^2)
    s_u = dT / L_ij
    s_v_sq = (1.0f0 / F_k)^2 - s_u^2
    s_v_sq <= 0.0f0 && return t_best
    s_v = sqrt(s_v_sq)

    # Check upwind causality: ray must intersect segment between 0 and L_ij
    x_intersect = x_k - y_k * (s_u / s_v)
    if 0.0f0 <= x_intersect <= L_ij
        t_2d = T_i + s_u * x_k + s_v * y_k
        return min(t_best, t_2d)
    end

    return t_best
end