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