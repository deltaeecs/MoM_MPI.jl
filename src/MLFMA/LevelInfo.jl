using MoM_Kernels:AbstractLevel, CubeInfo, PolesInfo, InterpInfo, memoryAllocationOnLevels!

"""
层信息（MPI分布式）
ID          ::IT，层序号
L           ::IT, 本层截断项数
cubes       ::MPIVector{Vector{CubeInfo{IT, FT}}, IDCSCT} 包含每一个盒子信息的向量
cubeEdgel   ::FT，本层盒子的边长
poles       ::PolesInfo{IT, FT}, 多极子采样信息
interpWθϕ   ::InterpInfo{IT, FT}, 插值信息
aggS        ::MPIArray{Complex{FT}, 3}， 聚合项
disaggG     ::MPIArray{Complex{FT}, 3}， 解聚项
phaseShift2Kids  ::Array{Complex{FT}, 3}，本层盒子到子层盒子的相移因子 
αTrans      ::Array{Complex{FT}, 3}， 本层盒子远亲组之间的转移因子，根据相对位置共有 7^3 - 3^3 = 316 个
αTransIndex ::Array{IT, 2}, 远亲盒子的相对位置到其转移因子在所有转移因子数组的索引
"""
mutable struct LevelInfoMPI{IT<:Integer, FT<:Real} <: AbstractLevel
    ID          ::IT
    L           ::IT
    nCubes      ::IT
    cubes       ::PartitionedVector{CubeInfo{IT, FT}}
    cubeEdgel   ::FT
    poles       ::PolesInfo{FT}
    interpWθϕ   ::InterpInfo{IT, FT}
    aggS        ::MPIArray{Complex{FT}, IDCSAT, 3} where IDCSAT
    aggStransfer::PatternTransfer{Complex{FT}, IDCSATT} where IDCSATT
    disaggG     ::MPIArray{Complex{FT}, IDCSDT, 3} where IDCSDT
    disaggGtransfer     ::PatternTransfer{Complex{FT}, IDCSDTT} where IDCSDTT
    phaseShift2Kids     ::Matrix{Complex{FT}}
    phaseShiftFromKids  ::Matrix{Complex{FT}}
    αTrans      ::Matrix{Complex{FT}}
    αTransTransfer ::PatternTransfer{Complex{FT}, IDCSTTT} where IDCSTTT
    αTransIndex ::OffsetArray{IT, 3, Array{IT, 3}}
    LevelInfoMPI{IT, FT}() where {IT<:Integer, FT<:Real} = new{IT, FT}()
    LevelInfoMPI{IT, FT}(   ID, L, nCubes, cubes, cubeEdgel, poles, interpWθϕ, aggS, aggStransfer, disaggG,
                        disaggGtransfer, phaseShift2Kids, phaseShiftFromKids, αTrans, αTransTransfer,  αTransIndex) where {IT<:Integer, FT<:Real} = 
                new{IT, FT}(ID, L, nCubes, cubes, cubeEdgel, poles, interpWθϕ, aggS, aggStransfer, disaggG,
                        disaggGtransfer, phaseShift2Kids, phaseShiftFromKids, αTrans,  αTransIndex)
end

"""
    getFarNeighborCubeIDs(cubes, chunkIndice)

    获取 ckunkIndice 内的所有 cube 的 远亲盒子ID， 返回为 Tuple 形式以适应数组索引相关API

TBW
"""
function getFarNeighborCubeIDs(cubes, chunkIndice::Tuple)

    farneighborCubeIDs = reduce(vcat, cubes[i].farneighbors for i in chunkIndice[1])

    return (unique!(sort!(farneighborCubeIDs)), )

end


"""
    loadMPICubes(cubefn)

    载入 cubes 数组
"""
function loadCubes(cubefn)

    # Cubes
    cubes       = load(cubefn, "data")

    return cubes

end

function loadMPILevel!(level, fn; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm))

    # 除了 cubes 之外的信息
    leveltemp = load(fn, "data")
    for n in propertynames(leveltemp)
        try
            setfield!(level, n, getproperty(leveltemp, n))
        catch e
            e == UndefRefError() && continue
        end
    end

    # 多极子数
    sizePoles   =   length(level.poles.r̂sθsϕs)
    # 分区
    partitation = if level.nCubes > 3np
        (1, 1, np)
    else
        aggSize = (sizePoles, 2, level.nCubes)
        # 分区
        slicedim2mpi(aggSize, np)
    end

    cubes_part  =   rank ÷ partitation[1] + 1

    cubefn = split(fn, ".")[1]*"_Cubes_part_$(cubes_part).jld2"
    # cubes
    cubes = loadCubes(cubefn)
    level.cubes = cubes

    return level
end

function loadMPILevel(fn; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm))

    level = LevelInfoMPI{Int, Precision.FT}()
    loadMPILevel!(level, fn; comm = comm, rank = rank, np = np)

    return level
end


"""
预分配各层上的聚合项、解聚项
"""
function MoM_Kernels.memoryAllocationOnLevels!(nLevels::Integer, levels::Dict{IT, LV}; np = MPI.Comm_size(MPI.COMM_WORLD)) where{IT<:Integer, LV<:LevelInfoMPI}

    # 用到的浮点数类型
    FT = typeof(levels[nLevels].cubeEdgel) 

    for iLevel in nLevels:-1:2
        level   =   levels[iLevel]
        # 本层盒子数
        nCubes  =   level.nCubes
        # 多极子数
        sizePoles   =   length(level.poles.r̂sθsϕs)

        aggSize = (sizePoles, 2, nCubes)
        # 分区
        partitation = if level.nCubes > 3np
            (1, 1, np)
        else
            aggSize = (sizePoles, 2, level.nCubes)
            # 分区
            slicedim2mpi(aggSize, np)
        end
        # 开始预分配内存
        # 聚合项
        aggS    =   mpiarray(Complex{FT}, aggSize; partitation = partitation)
        # 解聚项
        disaggG =   mpiarray(Complex{FT}, aggSize; partitation = partitation)
        # 保存
        level.aggS  =   aggS
        level.disaggG = disaggG
        # 更新本进程的转移矩阵，只保留本进程用到的部分
        update_local_transInfo_onLevel(level)
    end

    # 对层循环进行插值矩阵本地化
    for iLevel in (nLevels - 1):-1:2
        # 本层
        tLevel  =   levels[iLevel]
        # 子层
        kLevel  =   levels[iLevel + 1]
        # 更新每个进程真正用到的本地化插值矩阵避免每次迭代重新算
        update_local_interpInfo_onLevel(tLevel, kLevel)
    end #iLevel

    return nothing

end

function update_local_transInfo_onLevel(level)
    # 本 level 盒子
    cubes = level.cubes
    # 本层的316个转移因子和其索引 OffsetArray 矩阵
    αTrans      =   level.αTrans

    # 局部 pole 的索引和数据
    poleIndices, iθϕ, cubeIndices = level.disaggG.indices

    # 更新数据
    level.αTrans = αTrans[poleIndices, :]

    
    ## 转移时要用到远亲组的数据，因此需要设置对应的数据转移算子
    # 首先，找出所有非本 rank (Other rank, or) 的远亲
    farNeis_or   =   Int[]
    for iCube in cubeIndices
        cube    =   cubes[iCube]
        farNeighborIDs = cube.farneighbors
        # 对远亲循环
        for iFarNei in farNeighborIDs
            !(iFarNei in cubeIndices) && push!(farNeis_or, iFarNei)
        end
    end
    unique!(sort!(farNeis_or))
    # 创建转移子
    level.αTransTransfer     =   PatternTransfer((poleIndices, iθϕ, farNeis_or), level.aggS)

    return nothing
    
end

"""
这四个函数用于寻找盒子的子盒子区间内的比较函数，多重分派以实现
"""
func4Cube1stkInterval(cube::CubeInfo) = first(cube.kidsInterval)
func4Cube1stkInterval(i::T) where T <: Integer = i
func4Cube1stkInterval(interval::T) where T <: UnitRange = first(interval)
func4CubelastkInterval(cube::CubeInfo) = last(cube.kidsInterval)
func4CubelastkInterval(i::T) where T <: Integer = i
func4CubelastkInterval(interval::T) where T <: UnitRange = last(interval)

function AllGatherintervals(cubes::PartitionedVector; comm = MPI.COMM_WORLD)
    ls = MPI.Allgather(length(cubes.data), comm)
    data = Vector{UnitRange{Int}}(undef, sum(ls))
    MPI.Allgatherv!([cube.kidsInterval for cube in cubes.data.parent], VBuffer(data, ls), comm)

    unique!(sort!(data, by = first))

    return data
end

"""
    update_local_interpInfo_onLevel(tlevel, kLevel)

    根据层间信息更新多极子、插值、转移矩阵等信息方便计算，同时构建 聚合、解聚项的 数据通信算子。
TBW
"""
function update_local_interpInfo_onLevel(tLevel, kLevel)
    # 子层插值矩阵
    interpWθϕ   =   kLevel.interpWθϕ
    θCSC    =   interpWθϕ.θCSC
    ϕCSC    =   interpWθϕ.ϕCSC
    θCSCT   =   interpWθϕ.θCSCT
    ϕCSCT   =   interpWθϕ.ϕCSCT

    # 本层盒子信息
    cubes   =   tLevel.cubes
    # 本层聚合项
    tAggS   =   tLevel.aggS
    # 从子层到本层的相移
    phaseShiftFromKids  =   tLevel.phaseShiftFromKids
    # 子层解聚项
    kDisAggG    =   kLevel.disaggG
    # 从本层到子层的相移
    phaseShift2Kids     =   tLevel.phaseShift2Kids

    # 子层局部的索引和数据
    kpoleIndices, kiθϕIndices, _ = kDisAggG.indices

    ### 先计算聚合相关项
    # 局部的索引和数据
    tpoleIndices, tiθϕIndices, tcubeIndices = tAggS.indices
    kcubeIndices = last(kDisAggG.indices)

    ## 局部插值矩阵
    # θ 方向
    θCSClwtemp  =   θCSC[tpoleIndices, :]
    # 用到的 ϕ 方向的 tpoleIndices
    tϕpoleIndices    =   filter(i -> !iszero(θCSClwtemp.colptr[i+1] - θCSClwtemp.colptr[i]), 1:θCSClwtemp.n)
    θCSClw      =   θCSClwtemp[:, tϕpoleIndices]

    ϕCSClwtemp  =   ϕCSC[tϕpoleIndices, :]
    # 用到的 ϕ 方向的 tpoleIndices
    tθϕpoleIndices   =   filter(i -> !iszero(ϕCSClwtemp.colptr[i+1] - ϕCSClwtemp.colptr[i]), 1:ϕCSClwtemp.n)
    ϕCSClw      =   ϕCSClwtemp[:, tθϕpoleIndices]
    ## 局部相移矩阵
    phaseShiftFromKidslw    =   phaseShiftFromKids[tpoleIndices,:]

    ## kAggS 的数据通信算子
    # 先找出所有的子盒子
    cubesKidsInterval   =   AllGatherintervals(cubes)
    kCubesInterval  =   first(cubesKidsInterval[tcubeIndices[1]]):last(cubesKidsInterval[tcubeIndices[end]])
    # 本进程用到的所有子盒子的辐射积分
    tθϕpoleIndicesUsed  =  (length(tθϕpoleIndices) == length(kpoleIndices)) ? kpoleIndices : tθϕpoleIndices
    kAggStransfer   =   PatternTransfer((tθϕpoleIndicesUsed, tiθϕIndices, kCubesInterval), kLevel.aggS)

    ### 再计算解聚相关项
    ## 局部插值矩阵
    # ϕ 方向
    ϕCSCTlwtemp     =   ϕCSCT[kpoleIndices, :]
    # 用到的 θ 方向的 kpoleIndices
    kθpoleIndices   =   filter(i -> !iszero(ϕCSCTlwtemp.colptr[i+1] - ϕCSCTlwtemp.colptr[i]), 1:ϕCSCTlwtemp.n)
    ϕCSCTlw         =   ϕCSCTlwtemp[:, kθpoleIndices]

    θCSCTlwtemp     =   θCSCT[kθpoleIndices, :]
    # 用到的 ϕ 方向的 kpoleIndices
    kθϕpoleIndices  =   filter(i -> !iszero(θCSCTlwtemp.colptr[i+1] - θCSCTlwtemp.colptr[i]), 1:θCSCTlwtemp.n)
    θCSCTlw         =   θCSCTlwtemp[:, kθϕpoleIndices]
    ## 局部相移矩阵
    phaseShift2Kidslw    =   phaseShift2Kids[kθϕpoleIndices,:]

    ## tDisaggG 的数据通信算子
    # 先找出所有的用到的父盒子索引
    # tCubesInterval  =   last(searchsorted(cubes, first(kcubeIndices); by = func4Cube1stkInterval)):first(searchsorted(cubes, last(kcubeIndices); by = func4CubelastkInterval))
    tCubesInterval  =   last(searchsorted(cubesKidsInterval, first(kcubeIndices); by = func4Cube1stkInterval)):first(searchsorted(cubesKidsInterval, last(kcubeIndices); by = func4CubelastkInterval))
    # 本进程用到的所有盒子的辐射积分
    kθϕpoleIndicesUsed  =  (length(kθϕpoleIndices) == (tpoleIndices)) ? tpoleIndices : kθϕpoleIndices
    tDisaggStransfer    =   PatternTransfer((kθϕpoleIndicesUsed, kiθϕIndices, tCubesInterval), tLevel.disaggG)

    ### 更新
    interpWθϕ.θCSC    =   θCSClw
    interpWθϕ.ϕCSC    =   ϕCSClw
    interpWθϕ.θCSCT   =   θCSCTlw
    interpWθϕ.ϕCSCT   =   ϕCSCTlw
    tLevel.phaseShiftFromKids   =   phaseShiftFromKidslw
    tLevel.phaseShift2Kids      =   phaseShift2Kidslw
    kLevel.aggStransfer         =   kAggStransfer
    tLevel.disaggGtransfer      =   tDisaggStransfer

    return nothing

end