struct ValveGate{T<:Real, V<:StaticVector{3, T}}
    position::V
    open_time::T
end

# Convenience constructor so mixed precision (e.g., Point3f + Float64) works seamlessly:
function ValveGate(pos::StaticVector{3, T1}, open_time::T2) where {T1<:Real, T2<:Real}
    T = float(promote_type(T1, T2))
    return ValveGate{T, SVector{3, T}}(SVector{3, T}(pos), T(open_time))
end


struct MeshData{T<:Real}
    mesh::GeometryBasics.AbstractMesh{3, T}
    tlas::Raycore.AbstractAdaptedAccel
end

struct VoxelData{T<:Real}
    grid::BitArray{3}
    voxel_spacing::Tuple{T, T, T}
    corner::StaticVector{3, T}
end

# thick_r_sq, r_sq

struct TField
    thick_r_sq::Array{UInt16, 3}
    r_sq::Array{UInt16, 3}
end



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
function mask_points(m::SparseVoxelMask, spacing, corner)
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