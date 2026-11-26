
"""
    AngleWeightInterpolator(mesh::AbstractMesh, ::Type{T}=Float64)

Precompute normalized tip-angle weights for a `GeometryBasics` triangle mesh.
"""
function AngleWeightInterpolator(mesh::AbstractMesh, ::Type{T}=Float64) where {T<:AbstractFloat}
    positions = mesh.position 
    tri_faces = mesh.faces      
    
    n_verts = length(positions)
    n_faces = length(tri_faces)
    
    # Pre-allocate COO triplets (3 vertices per triangle)
    I = Vector{Int}(undef, 3 * n_faces)
    J = Vector{Int}(undef, 3 * n_faces)
    V = Vector{T}(undef, 3 * n_faces)
    
    vertex_weight_sums = zeros(T, n_verts)
    
    idx = 1
    for (f_idx, f) in enumerate(tri_faces)
        
        v1, v2, v3 = f
        
        p1, p2, p3 = positions[v1], positions[v2], positions[v3]
        
        # Unit edge vectors
        e12 = normalize(p2 - p1)
        e13 = normalize(p3 - p1)
        e23 = normalize(p3 - p2)
        
        # Interior tip angles
        a1 = acos(clamp(dot(e12, e13), -one(T), one(T)))
        a2 = acos(clamp(dot(-e12, e23), -one(T), one(T)))
        a3 = T(π) - a1 - a2
        
        # Accumulate total angle around each vertex for normalization
        vertex_weight_sums[v1] += a1
        vertex_weight_sums[v2] += a2
        vertex_weight_sums[v3] += a3
        
        # Store unnormalized triplets
        I[idx] = v1; J[idx] = f_idx; V[idx] = a1
        I[idx+1] = v2; J[idx+1] = f_idx; V[idx+1] = a2
        I[idx+2] = v3; J[idx+2] = f_idx; V[idx+2] = a3
        idx += 3
    end
    
    # Normalize weights so each row sums to 1
    @inbounds for k in eachindex(V)
        V[k] /= vertex_weight_sums[I[k]]
    end
    
    W = sparse(I, J, V, n_verts, n_faces)
    return AngleWeightInterpolator{T}(W)
end

function AngleWeightInterpolator(meshdata::MeshData, FT::Type{T}=Float64) where {T<:AbstractFloat}
    return AngleWeightInterpolator(meshdata.mesh, FT)
end


"""
    extrapolate_to_vertices!(out::AbstractVector, interp::AngleWeightInterpolator, face_data::AbstractVector)

In-place extrapolation of `face_data` into the pre-allocated vertex buffer `out`.
Works for scalars, `Vec`, `Point`, `SVector`, `SMatrix`, etc.
"""
function extrapolate_to_vertices!(out::AbstractVector, interp::AngleWeightInterpolator, face_data::AbstractVector)
    ET = eltype(face_data)
    @assert applicable(zero, ET) "Unsupported element type $ET: `zero($ET)` must be defined (use scalars, SVector, SMatrix, Vec, or Point rather than dynamically sized Vector/Matrix)."
    @assert length(out) == size(interp.W, 1) "Output buffer length must match vertex count."
    @assert length(face_data) == size(interp.W, 2) "Input data length must match face count."

    fill!(out, zero(eltype(out)))
    mul!(out, interp.W, face_data)
    return out
end

"""
    extrapolate_to_vertices(interp::AngleWeightInterpolator, face_data::AbstractVector)

Allocating convenience method that creates an output buffer matching `eltype(face_data)`.
"""
function extrapolate_to_vertices(interp::AngleWeightInterpolator, face_data::AbstractVector)
    out = Vector{eltype(face_data)}(undef, size(interp.W, 1))
    return extrapolate_to_vertices!(out, interp, face_data)
end

# Functor syntax so you can call `interp(face_data)` or `interp(out, face_data)` directly
(interp::AngleWeightInterpolator)(face_data::AbstractVector) = extrapolate_to_vertices(interp, face_data)
(interp::AngleWeightInterpolator)(out::AbstractVector, face_data::AbstractVector) = extrapolate_to_vertices!(out, interp, face_data)


"""
    align_flow_segments!(pts::Vector{Point3f}, st, dx::Float32; n_iters::Int = 2)

In-place post-processing filter for `flow_segments` (`[p1_start, p1_end, p2_start, p2_end, ...]`).
Smooths tangent directions and laterally aligns upstream/downstream dashes onto coherent curves
while keeping points offset along surface normals without BVH raycasts.
"""
function align_flow_segments!(pts::Vector{Point3f}, st, dx::Float32; n_iters::Int = 2)
    n_segs = length(pts) ÷ 2
    n_segs == 0 && return pts

    centroids = st.centroids
    normals   = st.normals
    flow_dirs = st.flow_dir

    active_tris = Vector{Int}(undef, n_segs)
    k = 0
    @inbounds for i in eachindex(centroids)
        if sum(abs2, flow_dirs[i]) > 1f-6
            k += 1
            k <= n_segs && (active_tris[k] = i)
        end
    end
    k == n_segs || return pts

    cell_size = 1.6f0 * dx
    inv_cell  = 1.0f0 / cell_size
    search_r2 = (1.6f0 * dx)^2

    grid = Dict{NTuple{3, Int}, Vector{Int}}()
    sizehint!(grid, n_segs)

    mids = Vector{Point3f}(undef, n_segs)
    dirs = Vector{Vec3f}(undef, n_segs)
    half_lens = Vector{Float32}(undef, n_segs)

    @inbounds for s in 1:n_segs
        p0 = pts[2s - 1]
        p1 = pts[2s]
        mid = 0.5f0 * (p0 + p1)
        seg = p1 - p0
        len = norm(seg)

        mids[s] = mid
        half_lens[s] = 0.5f0 * len
        dirs[s] = len > 1f-6 ? (seg / len) : Vec3f(1, 0, 0)

        ck = (floor(Int, mid[1] * inv_cell), floor(Int, mid[2] * inv_cell), floor(Int, mid[3] * inv_cell))
        bucket = get!(Vector{Int}, grid, ck)
        push!(bucket, s)
    end

    anchors = copy(mids)
    max_nudge = 0.48f0 * dx

    for _ in 1:n_iters
        new_mids = copy(mids)
        new_dirs = copy(dirs)

        @inbounds for s in 1:n_segs
            tri_i = active_tris[s]
            n = Vec3f(normals[tri_i])
            u = dirs[s]
            p = mids[s]

            t_perp_raw = cross(n, u)
            tp_len = norm(t_perp_raw)
            tp_len > 1f-5 || continue
            t_perp = t_perp_raw / tp_len

            ck = (floor(Int, p[1] * inv_cell), floor(Int, p[2] * inv_cell), floor(Int, p[3] * inv_cell))

            u_accum = 2.0f0 * u
            lat_shift_sum = 0.0f0
            w_lat_sum = 0.0f0

            for dz in -1:1, dy in -1:1, dx_i in -1:1
                bucket = get(grid, (ck[1] + dx_i, ck[2] + dy, ck[3] + dz), nothing)
                bucket === nothing && continue

                for nb in bucket
                    nb == s && continue
                    tri_nb = active_tris[nb]

                    dot(n, Vec3f(normals[tri_nb])) > 0.75f0 || continue

                    delta = mids[nb] - p
                    d2 = sum(abs2, delta)
                    (1f-8 < d2 <= search_r2) || continue

                    u_nb = dirs[nb]
                    cos_u = dot(u, u_nb)
                    cos_u > 0.5f0 || continue

                    d = sqrt(d2)
                    w_dist = 1.0f0 - d / (1.6f0 * dx)

                    u_accum += w_dist * u_nb

                    along = abs(dot(delta, u)) / d
                    w_align = w_dist * (along^2)
                    if w_align > 1f-4
                        lat_offset = dot(delta, t_perp)
                        lat_shift_sum += w_align * lat_offset
                        w_lat_sum += w_align
                    end
                end
            end

            u_tan = u_accum - dot(u_accum, n) * n
            ut_len = norm(u_tan)
            u_smooth = ut_len > 1f-5 ? (u_tan / ut_len) : u
            new_dirs[s] = u_smooth

            if w_lat_sum > 1f-4
                p_candidate = p + (lat_shift_sum / w_lat_sum) * t_perp
                total_offset = p_candidate - anchors[s]
                tan_offset = total_offset - dot(total_offset, n) * n
                off_len = norm(tan_offset)
                if off_len > max_nudge
                    tan_offset *= (max_nudge / off_len)
                end
                new_mids[s] = anchors[s] + tan_offset
            end
        end

        mids .= new_mids
        dirs .= new_dirs
    end

    @inbounds for s in 1:n_segs
        step_vec = half_lens[s] * dirs[s]
        pts[2s - 1] = mids[s] - step_vec
        pts[2s]     = mids[s] + step_vec
    end

    return pts
end

function compute_flow_line_len(md::MeshData)
    pts, tris = md.mesh.position, md.mesh.faces
    isempty(tris) && return 1.0

    len = 0.0
    for tri in tris
        i1, i2, i3 = tri
        e12 = pts[i1] - pts[i2]
        e13 = pts[i1] - pts[i3]
        e23 = pts[i2] - pts[i3]
        lmax = max(dot(e12, e12), max(dot(e13, e13), dot(e23, e23)))
        len += sqrt(lmax)
    end
    return len / length(tris)
end

function first_triangle_centroid(md::MeshData)
    verts = md.mesh.position
    faces = md.mesh.faces
    isempty(faces) && return Point3f(0, 0, 0)
    f1 = first(faces)
    return Point3f((verts[f1[1]] + verts[f1[2]] + verts[f1[3]]) / 3.0f0)
end