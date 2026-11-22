


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


"""
    HalfResThickness <: AbstractArray{UInt16, 3}

Stores squared voxel thickness at 1/2 resolution per axis (1/8 total elements = 2 bits
per full-res voxel) and transparently reconstructs full-resolution values on `getindex`
via exact dyadic {3/4, 1/4} integer trilinear interpolation (`>> 6`).
"""
struct HalfResThickness <: AbstractArray{UInt16, 3}
    data::Array{UInt16, 3}   # size = cld.(dims, 2), stored in fine squared-voxel units
    dims::NTuple{3, Int}     # full-resolution grid dimensions
end

Base.size(A::HalfResThickness) = A.dims
Base.IndexStyle(::Type{<:HalfResThickness}) = IndexCartesian()

@inline function Base.getindex(A::HalfResThickness, i::Int, j::Int, k::Int)::UInt16
    @boundscheck checkbounds(A, i, j, k)
    nx, ny, nz = size(A.data)

    @inbounds begin
        # Coarse cell containing fine voxel (i, j, k)
        ic = (i + 1) >> 1
        jc = (j + 1) >> 1
        kc = (k + 1) >> 1

        # Odd fine index -> neighbor is ic - 1; Even fine index -> neighbor is ic + 1
        inbr = clamp(ic + ifelse(isodd(i), -1, 1), 1, nx)
        jnbr = clamp(jc + ifelse(isodd(j), -1, 1), 1, ny)
        knbr = clamp(kc + ifelse(isodd(k), -1, 1), 1, nz)

        v000 = UInt32(A.data[ic,   jc,   kc])
        v100 = UInt32(A.data[inbr, jc,   kc])
        v010 = UInt32(A.data[ic,   jnbr, kc])
        v001 = UInt32(A.data[ic,   jc,   knbr])
        v110 = UInt32(A.data[inbr, jnbr, kc])
        v101 = UInt32(A.data[inbr, jc,   knbr])
        v011 = UInt32(A.data[ic,   jnbr, knbr])
        v111 = UInt32(A.data[inbr, jnbr, knbr])

        acc = UInt32(27) * v000 +
              UInt32(9)  * (v100 + v010 + v001) +
              UInt32(3)  * (v110 + v101 + v011) +
              v111 + UInt32(32) # +32 for nearest-integer rounding before >> 6

        return UInt16(acc >> 6)
    end
end

"""
    HalfResThickness(full_thick::AbstractArray{UInt16, 3})

Downsamples a full-resolution `thick_r_sq` array into a `HalfResThickness` field using
the maximum value in each 2×2×2 block (preventing air zeros from eroding wall thickness).
"""
function HalfResThickness(full_thick::AbstractArray{UInt16, 3})
    nx, ny, nz = size(full_thick)
    cnx, cny, cnz = cld(nx, 2), cld(ny, 2), cld(nz, 2)
    coarse = zeros(UInt16, cnx, cny, cnz)

    @inbounds for ck in 1:cnz, cj in 1:cny, ci in 1:cnx
        i0, i1 = 2ci - 1, min(2ci, nx)
        j0, j1 = 2cj - 1, min(2cj, ny)
        k0, k1 = 2ck - 1, min(2ck, nz)

        m = UInt16(0)
        for k in k0:k1, j in j0:j1, i in i0:i1
            v = full_thick[i, j, k]
            if v > m
                m = v
            end
        end
        coarse[ci, cj, ck] = m
    end

    return HalfResThickness(coarse, (nx, ny, nz))
end



struct VoxelData{T<:Real, A<:AbstractArray{UInt16, 3}}
    grid::BitArray{3}
    thickness::A # supports Array{UInt16, 3} or HalfResThickness
    voxel_spacing::Tuple{T, T, T}
    corner::StaticVector{3, T}

    function VoxelData(
        g::BitArray{3},
        thickness::A,
        vs::Tuple{T, T, T},
        CP::StaticVector{3, T}
    ) where {T<:Real, A<:AbstractArray{UInt16, 3}}
        size(g) == size(thickness) || error("grid and thickness arrays must have the same size")
        return new{T, A}(g, thickness, vs, CP)
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