include("ExcitedVectors/CFIEExcitedVectors.jl")
include("ExcitedVectors/EFIEExcitedVectors.jl")
include("ExcitedVectors/MFIEExcitedVectors.jl")

"""
根据几何信息与基函数数量，计算激励向量
输入：
geosInfo::  几何信息，三角形、四面体、六面体的向量
nbf::       基函数数量
source::    激励源
返回：
V::         激励向量
"""
function getExcitationVector!(V, geosInfo::AbstractVector{VST}, source, bfType =  VSBFTypes.sbfType) where {VST <: SurfaceCellType}
    # 置零避免累加错误
    fill!(V, 0)
    if SimulationParams.ieT == :EFIE
        # 计算 RWG下 的 EFIE 激励向量
        excitationVectorEFIE!(V, source, geosInfo, bfType)
    elseif SimulationParams.ieT == :MFIE
        # 计算 RWG下 的 MFIE 激励向量
        excitationVectorMFIE!(V, source, geosInfo, bfType)
    elseif SimulationParams.ieT == :CFIE
        # 计算 RWG下 的 CFIE 激励向量
        excitationVectorCFIE!(V, source, geosInfo, bfType)
    end
    return V
end

"""
根据几何信息与基函数数量，计算激励向量
输入：
geosInfo::  几何信息，三角形、四面体、六面体的向量
nbf::       基函数数量
source::    激励源
返回：
V::         激励向量
"""
function getExcitationVector!(V, geosInfo::AbstractVector{VST}, source,  bfType =  VSBFTypes.vbfType) where {VST<:VolumeCellType}
    # 置零避免累加错误
    fill!(V, 0)
    # 计算 体积分的 激励向量
    excitationVectorEFIE!(V, source, geosInfo, bfType)
    return V
end

"""
根据几何信息与基函数数量，计算激励向量
输入：
geosInfo::  几何信息，三角形、四面体、六面体的向量
nbf::       基函数数量
source::    激励源
返回：
V::         激励向量
"""
function getExcitationVector!(V, geosInfo::AbstractVector{VT}, source) where {VT<:AbstractVector}
    # 置零避免累加错误
    fill!(V, 0)
    if SimulationParams.ieT == :EFIE
        # 计算 RWG下 的 EFIE 激励向量
        excitationVectorEFIE!(V, source, geosInfo)
    elseif SimulationParams.ieT == :MFIE
        # 计算 RWG下 的 MFIE 激励向量
        excitationVectorMFIE!(V, source, geosInfo)
    elseif SimulationParams.ieT == :CFIE
        # 计算 RWG下 的 CFIE 激励向量
        excitationVectorCFIE!(V, source, geosInfo)
    end
    return V
end
