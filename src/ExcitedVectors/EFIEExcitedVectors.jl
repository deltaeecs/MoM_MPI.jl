using MoM_Kernels:excitationVectorEFIE, excitationVectorEFIE!

"""
计算平面波在 RWG 基函数上的激励向量
输入：
source          ::ST, 波源
geosInfo        ::Vector{TriangleInfo{IT, FT}}，保存三角形信息的向量
nbf             ::Integer，基函数数目
"""
function MoM_Kernels.excitationVectorEFIE!(V::MPIVector, source::ST, geosInfo::SparseVector{GT, Int}, sbfType = VSBFTypes.sbfType) where{ST<:ExcitingSources, GT<:SurfaceCellType}
    
    nt  =   length(geosInfo.nzval)
    # V 在本rank的区间
    Vindices =  V.indices[1]
    pmeter  =   Progress(nt, "Calculating V...")
    lockV   =   SpinLock()
    # 开始对四面体形循环计算
    @threads for geo in geosInfo.nzval
        next!(pmeter)

        # geo.inBfsID 全不在本区间则跳过
        all(x -> !(x in Vindices), geo.inBfsID) && continue
        # 三角形相关激励向量
        Vtri    =   excitationVectorEFIE(source, geo, sbfType)
        # 写入结果
        for ni in eachindex(geo.inBfsID)
            n = geo.inBfsID[ni]
            # 跳过半基函数
            (n == 0) && continue
            lock(lockV)
            V[n] += Vtri[ni]
            unlock(lockV)
        end
    end # for geo

    nothing
end #for function

"""
计算源在 基函数 上的激励向量
输入：
source          ::ST, 平面波源
geosInfo      ::Vector{TetrahedraInfo{IT, FT, CT}}，保存四面体信息的向量
nbf             ::Integer，基函数数量  
"""
function MoM_Kernels.excitationVectorEFIE!(V::MPIVector, source::ST, geosInfo::AbstractVector{GT}, vbfType =   VSBFTypes.vbfType) where {ST<:ExcitingSources, GT<:VolumeCellType}

    nt  =   length(geosInfo.nzval)
    # V 在本rank的区间
    Vindices =  V.indices[1]
    pmeter  =   Progress(nt, "Calculating V...")
    lockV   =   SpinLock()
    # 开始对四面体形循环计算
    @threads for geo in geosInfo.nzval
        next!(pmeter)
        # geo.inBfsID 全不在本区间则跳过
        all(x -> !(x in Vindices), geo.inBfsID) && continue
        # 四面体相关激励向量
        Vtetra    =   excitationVectorEFIE(source, geo, vbfType)
        # 写入结果
        lock(lockV)
        for i in eachindex(geo.inBfsID)
            V[geo.inBfsID[i]] += Vtetra[i]
        end
        unlock(lockV)
    end # for geo

    # 返回
    return V
end # function

