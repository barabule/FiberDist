
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


