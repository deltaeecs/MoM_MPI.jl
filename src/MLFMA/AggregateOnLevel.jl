include("AggOnBF/AggEFIE.jl")
include("AggOnBF/AggCFIE.jl")

"""
在所有分配叶层辐射或接收积分内存
"""
function allocatePatternOnLeaflevel(level::LT; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), 
    np = MPI.Comm_size(comm)) where {LT<:LevelInfoMPI}
    # 复数计算类型
    CT  =   Complex{typeof(level.cubeEdgel)}

    # 先将 cubes fetch到本地
    cubes   =   level.cubes
    # 采样多极子数量
    nPoles  =   length(level.poles.r̂sθsϕs)

    # 分配数据创建MPI数组
    aggSBF = allocatePatternCubes(cubes, nPoles; T = CT, comm = comm, rank = rank, np = np)
    disaggSBF = allocatePatternCubes(cubes, nPoles; T = CT, comm = comm, rank = rank, np = np)

    return aggSBF, disaggSBF

end


"""
在本进程上分配叶层辐射或接收积分内存
"""
function allocatePatternCubes(cubes::PartitionedVector, nPoles; T = Precision.CT, 
    comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm))

    # 本进程的盒子、邻盒子包含的基函数数量
    ngbfslw      =   sum(idc -> length(cubes[idc].bfInterval), cubes.indices; init = 0)
    # 分配内存
    ghostdata   =   zeros(T, nPoles, 2, ngbfslw)

    # 本地 vector 的区间
    indices         =  (1:nPoles, 1:2, first(cubes.data).bfInterval[1]:last(cubes.data).bfInterval[end])
    allindices      =   MPI.Allgather(indices, comm)
    rank2indices    =   Dict(zip(0:(np-1), allindices))
    datasize        =   last.(last(allindices))

    ghostindices    =   indices
    data            =   ghostdata
    
    # ghost rank to ghost indices (空)
    grank2gindices  = typeof(rank2indices)()
    # 最终获得远程进程所需数据在本进程存储的 indices 上的位置 (空)
    rrank2indices   = typeof(rank2indices)()
    aggSBF = MPIArray{T, typeof(indices), 3}(data, indices, OffsetArray(data, indices), comm, rank, datasize, rank2indices, ghostdata, ghostindices, grank2gindices, rrank2indices)

    sync!(aggSBF)

    MPI.Barrier(comm)
    
    return aggSBF

end

using MoM_Kernels:getAggSBFOnLevel

"""
根据积分方程类型计算基层聚合项
"""
function MoM_Kernels.getAggSBFOnLevel(level::LT, geosInfo::AbstractVector{ST}, 
        bfT = VSBFTypes.sbfType) where {LT<:LevelInfoMPI, ST<:SurfaceCellType}
    if SimulationParams.ieT == :EFIE
        return aggSBFOnLevelEFIE(level, geosInfo, bfT)
    elseif SimulationParams.ieT == :MFIE
        throw("暂未支持单独 MFIE 的分布式计算")
        # return aggSBFOnLevelMFIE(level, geosInfo, bfT)
    elseif SimulationParams.ieT == :CFIE
        return aggSBFOnLevelCFIE(level, geosInfo, bfT)
    end
end

"""
根据积分方程类型计算基层聚合项
"""
function MoM_Kernels.getAggSBFOnLevel(level::LT, geosInfo::AbstractVector{VT}, 
        bfT = VSBFTypes.vbfType) where {LT<:LevelInfoMPI, VT<:VolumeCellType}
    return aggSBFOnLevel(level, geosInfo, bfT)
end

"""
根据积分方程类型计算基层聚合项
"""
function MoM_Kernels.getAggSBFOnLevel(level::LT, geosInfoV::AbstractVector{VT}) where {LT<:LevelInfoMPI, VT<:AbstractVector}
    # 预分配内存
    aggSBF, disaggSBF = allocatePatternOnLeaflevel(level)

    if eltype(geosInfoV[1]) <: SurfaceCellType    
        # 面元、体元
        geoSs   =   geosInfoV[1]
        geoVs   =   geosInfoV[2]
        # 面基函数，体基函数
        bfST    =   VSBFTypes.sbfType
        bfVT    =   VSBFTypes.vbfType
        # 面部分
        if SimulationParams.ieT == :EFIE
            aggSBFOnLevelEFIE!(aggSBF, disaggSBF, level, geoSs, bfST)
        elseif SimulationParams.ieT == :MFIE
            aggSBFOnLevelMFIE!(aggSBF, disaggSBF, level, geoSs, bfST)
        elseif SimulationParams.ieT == :CFIE
            aggSBFOnLevelCFIE!(aggSBF, disaggSBF, level, geoSs, bfST)
        end
        # 体部分
        aggSBFOnLevelD!(aggSBF, disaggSBF, level, geoVs, bfVT)
    else
        for i in eachindex(geosInfoV)
            geosInfo = geosInfoV[i]
            aggSBFOnLevel!(aggSBF, disaggSBF, level, geosInfo, VSBFTypes.vbfType)
        end
    end

    return aggSBF, disaggSBF

end
