


"""
    ValveGate

Represents an injection gate at 3D world coordinates `position` with orifice `radius`
and optional `open_time` (default `0.0f0` for simultaneous gates, or `> 0` for sequential
cascade valve gating).
"""
Base.@kwdef struct ValveGate
    position::SVector{3, Float32}
    radius::Float32 = 0.0f0
    open_time::Float32 = 0.0f0
end


"""
    MeshAdjacency

Thread-safe Compressed Sparse Row (CSR) triangle-to-triangle edge adjacency.
Build once per `MeshData` and pass into `project_voxel_to_mesh_triangles` across gate runs.
"""
struct MeshAdjacency
    offsets::Vector{Int32}
    neighbors::Vector{Int32}
end



struct MeshData{T<:Real} #Struct contains a mesh and its acceleration struct
    mesh::GeometryBasics.AbstractMesh{3, T}
    tlas::Raycore.AbstractAdaptedAccel
    adj::MeshAdjacency
end

function MeshData(mesh::GeometryBasics.AbstractMesh{3, T}, tlas::Raycore.AbstractAdaptedAccel) where T<:Real
    adj = build_mesh_adjacency(mesh)
    return MeshData{T}(mesh, tlas, adj)
end



struct VoxelData{T<:Real} # in/out indicator
    grid::BitArray{3} 
    thickness::Array{UInt16, 3} #squared voxel radius
    voxel_spacing::Tuple{T, T, T}
    corner::StaticVector{3, T}
end

# """
#     ThicknessField

# Compact 3D inscribed-sphere local wall half-thickness stored in squared voxel
# units (`UInt16`, 2 bytes/voxel). Physical full wall thickness at `(i, j, k)` is:
# `h = 2 * sqrt(thick_r_sq[i, j, k]) * Δx`.
# """
# struct ThicknessField
#     thick_r_sq::Array{UInt16, 3}
# end

# # Forward basic array indexing/size helpers for convenience
# Base.size(tf::ThicknessField) = size(tf.thick_r_sq)
# @inline Base.getindex(tf::ThicknessField, i::Int, j::Int, k::Int) = @inbounds tf.thick_r_sq[i, j, k]
# @inline Base.getindex(tf::ThicknessField, idx::Int) = @inbounds tf.thick_r_sq[idx]



"""
    SparseVoxelMask <: AbstractArray{Bool, 3}

Memory-compact 3D boolean mask storing sorted `Int32` linear indices of `true` voxels
(4 bytes per active voxel). Supports O(log k) random indexing `mask[i, j, k]` and O(k)
iteration over active voxels via `active_indices(mask)`.
"""
struct SparseVoxelMask <: AbstractArray{Bool, 3}
    dims::NTuple{3, Int}
    nz_inds::Vector{Int32} # strictly sorted linear indices
end

function SparseVoxelMask(bitmask::BitArray{3})
    inds = Int32.(vec(findall(vec(bitmask))))
    return SparseVoxelMask(size(bitmask), inds)
end

Base.size(m::SparseVoxelMask) = m.dims
Base.count(m::SparseVoxelMask) = length(m.nz_inds)
active_indices(m::SparseVoxelMask) = m.nz_inds

@inline function Base.getindex(m::SparseVoxelMask, i::Int, j::Int, k::Int)
    @boundscheck checkbounds(m, i, j, k)
    lin = Int32(LinearIndices(m.dims)[i, j, k])
    idx = searchsortedfirst(m.nz_inds, lin)
    return idx <= length(m.nz_inds) && @inbounds(m.nz_inds[idx]) == lin
end

# Fast O(k) coordinate extraction for GLMakie scatter!
function coordinates(m::SparseVoxelMask, spacing, corner)
    cart = CartesianIndices(m.dims)
    pts = Vector{Point3f}(undef, length(m.nz_inds))
    @inbounds for (idx, lin) in enumerate(m.nz_inds)
        c = cart[Int(lin)]
        pts[idx] = Point3f(
            corner[1] + (c[1] - 0.5f0) * spacing[1],
            corner[2] + (c[2] - 0.5f0) * spacing[2],
            corner[3] + (c[3] - 0.5f0) * spacing[3]
        )
    end
    return pts
end


function coordinates(m::SparseVoxelMask, voxeldata::VoxelData)
    return coordinates(m, voxeldata.voxel_spacing, voxeldata.corner)
end