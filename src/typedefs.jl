struct ValveGate{T<:Real, V<:StaticVector{3, T}}
    position::V
    open_time::T
end

# Convenience constructor so mixed precision (e.g., Point3f + Float64) works seamlessly:
function ValveGate(pos::StaticVector{3, T1}, open_time::T2) where {T1<:Real, T2<:Real}
    T = float(promote_type(T1, T2))
    return ValveGate{T, SVector{3, T}}(SVector{3, T}(pos), T(open_time))
end