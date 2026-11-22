


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
    adj = MeshAdjacency(mesh)
    return MeshData{T}(mesh, tlas, adj)
end

function MeshData(mesh::GeometryBasics.AbstractMesh{3, T}) where T<:Real
    faces=  mesh.faces
    if eltype(faces) <: TriangleFace
        trimesh =  mesh
    end
    tri_faces = decompose(TriangleFace, faces)
    trimesh = GeometryBasics.Mesh(vertex_attributes(mesh), tri_faces)

    tlas = TLAS([mesh], (mi, ti) -> UInt32(mi))

    adj  = MeshAdjacency(trimesh)

    return MeshData{T}(trimesh, tlas, adj)
end



struct VoxelData{T<:Real} # in/out indicator
    grid::BitArray{3} 
    thickness::Array{UInt16, 3} #squared voxel radius
    voxel_spacing::Tuple{T, T, T}
    corner::StaticVector{3, T}

    function VoxelData(g::BitArray{3}, thickness::Array{UInt16, 3}, vs::Tuple{T, T, T}, CP::StaticVector{3, T}) where {T<:Real}
        size(g) == size(thickness) || error("grid and thickness arrays must have the same size")
       
        return new{T}(g, thickness, vs, CP)
    end
end



function wallthickness(VD::VoxelData, idx::Int)
    if VD.grid[idx]
        return 2 * sqrt(VD.thickness[idx]) * VD.voxel_spacing[1]
    else
        return 0.0
    end
end

function wallthickness(VD::VoxelData, i::Int, j::Int, k::Int)
    if VD.grid[i, j, k]
        return 2 * sqrt(VD.thickness[i, j, k]) * VD.voxel_spacing[1]
    else
        return 0.0
    end
end

#Eikonal Hele-Shaw speed F = h² needs NO sqrt() at all!
# Since h = 2 * sqrt(r_sq) * dx,  h² = 4 * r_sq * dx²
function flowvelocity(VD::VoxelData, idx::Int)
    if VD.grid[idx]
        dx = VD.voxel_spacing[1]
        return 4 * VD.thickness[idx] * dx * dx
    else
        return 0.0
    end
end

function flowvelocity(VD::VoxelData, i::Int, j::Int, k::Int)
    if VD.grid[i, j, k]
        dx = VD.voxel_spacing[1]
        return 4 * VD.thickness[i, j, k] * dx * dx
    else
        return 0.0
    end
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



"""
    AngleWeightInterpolator{T}

Precomputed sparse interpolation matrix mapping per-face values to per-vertex
values using tip-angle weights.
"""
struct AngleWeightInterpolator{T<:AbstractFloat}
    W::SparseMatrixCSC{T, Int}
end