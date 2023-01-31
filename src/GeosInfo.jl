
function getGeosInfo(fn)

    @unpack data, size, indice = load(fn)

    geosInfolw = sparsevec(indice[1], data, size[1])

    return geosInfolw

end

using MoM_Kernels:getGeosInterval

"""
获取几何信息数组的区间，针对 AbstractVector{TriangleInfo} 派发
"""
function MoM_Kernels.getGeosInterval(::AbstractVector{T})::UnitRange{Int} where {T<:TriangleInfo}
    return GeosInterval.tri
end

"""
获取几何信息数组的区间，针对 AbstractVector{TetrahedraInfo} 派发
"""
function MoM_Kernels.getGeosInterval(::AbstractVector{T})::UnitRange{Int} where {T<:TetrahedraInfo}
    return GeosInterval.tetra
end

"""
获取几何信息数组的区间，针对 AbstractVector{HexahedraInfo} 派发
"""
function MoM_Kernels.getGeosInterval(::AbstractVector{T})::UnitRange{Int} where {T<:HexahedraInfo}
    return GeosInterval.hexa
end