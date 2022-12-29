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
    aggStransfer::ArrayTransfer{Complex{FT}, 3, IDCSATT} where IDCSATT
    disaggG     ::MPIArray{Complex{FT}, IDCSDT, 3} where IDCSDT
    disaggGtransfer     ::ArrayTransfer{Complex{FT}, 3, IDCSDTT} where IDCSDTT
    phaseShift2Kids     ::Matrix{Complex{FT}}
    phaseShiftFromKids  ::Matrix{Complex{FT}}
    αTrans      ::Matrix{Complex{FT}}
    αTransIndex ::OffsetArray{IT, 3, Array{IT, 3}}
    LevelInfoMPI{IT, FT}() where {IT<:Integer, FT<:Real} = new{IT, FT}()
    LevelInfoMPI{IT, FT}(   ID, L, nCubes, cubes, cubeEdgel, poles, interpWθϕ, aggS, aggStransfer, disaggG,
                        disaggGtransfer, phaseShift2Kids, phaseShiftFromKids, αTrans,  αTransIndex) where {IT<:Integer, FT<:Real} = 
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
    getNeiFarNeighborCubeIDs(cubes, chunkIndice::Tuple)


    getFarNeighborCubeIDs(cubes, chunkIndice)

    获取 ckunkIndice 内的所有 cube 的 远亲盒子ID， 返回为 Tuple 形式以适应数组索引相关API

TBW
"""
function getNeiFarNeighborCubeIDs(cubes, chunkIndice::Tuple)

    neighborCubeIDs     = reduce(vcat, cubes[i].neighbors for i in chunkIndice[1])
    farneighborCubeIDs  = reduce(vcat, cubes[i].farneighbors for i in chunkIndice[1])

    return (unique!(sort!(vcat(neighborCubeIDs, farneighborCubeIDs))), )

end

function saveCubes(cubes, nchunk = ParallelParams.nprocs; name, dir="")
	!ispath(dir) && mkpath(dir)
    # 对盒子按 盒子数 和块数分块
	cubes_ChunksIndices =   sizeChunks2idxs(length(cubes), nchunk)
    # 拿到各块的包含邻盒子的id
    cubesFarNeighbors_ChunksIndices    =   ThreadsX.mapi(chunkIndice -> getNeiFarNeighborCubeIDs(cubes, chunkIndice), cubes_ChunksIndices)
    # 分块后的索引区间
	indices = sizeChunks2idxs(length(cubes), nchunk)

    @floop for (i, indice) in enumerate(indices)
        data = OffsetVector(cubes[indice...], indice...)
        idcs = indice[1]
        ghostindices::Vector{Int} = setdiff(cubesFarNeighbors_ChunksIndices[i][1], indice[1])
        ghostdata = sparsevec(ghostindices, cubes[ghostindices])
        cubes_i = PartitionedVector{eltype(cubes)}(length(cubes), data, idcs, ghostdata, ghostindices)
		jldsave(joinpath(dir, "$(name)_part_$i.jld2"), data = cubes_i)
	end

    return nothing
    
end

function saveLevel(level, np = ParallelParams.nprocs; dir="")

    !ispath(dir) && mkpath(dir)

    # cube要单独处理
    cubes = level.cubes
    level.cubes = eltype(cubes)[]

    # 多极子数
    sizePoles   =   length(level.poles.r̂sθsϕs)

    aggSize = (sizePoles, 2, length(cubes))
    # 分区
    partitation = slicedim2mpi(aggSize, np)

    saveCubes(cubes, partitation[3]; name = "Level_$(level.ID)_Cubes", dir=dir)

    # 保存
    jldsave(joinpath(dir, "Level_$(level.ID).jld2"), data = level)

    level.cubes = cubes

    nothing

end

"""
    loadMPICubes(cubefn)

    载入 cubes 数组
"""
function loadCubes(cubefn)

    # Cubes
    cubes       = load(cubefn, "data")

    # dataloaded  = load(cubefn)
    # datasize    = dataloaded["size"]
    # ghostindices= dataloaded["indice"]

    # # indices
    # allindices   = sizeChunks2idxs(datasize, np)
    # rank2indices = Dict(zip(0:(np-1), allindices))
    # indices = rank2indices[rank]

    # # ghost rank to ghost indices
    # ghostranks      = indice2ranks(ghostindices, rank2indices)
    # grank2gindices  = grank2ghostindices(ghostranks, ghostindices, rank2indices; localrank = rank)

    # # remote rank to remote indices
    # rank2ghostindices = Dict{Int, typeof(ghostindices)}()
    # for rk in ghostranks
    #     rk == rank && continue
    #     rank2ghostindices[rk] =  load(replace(cubefn, "part_$(rank+1)" => "part_$(rk+1)"), "indice")
    # end
    # remoteranks     = indice2ranks(indices, rank2ghostindices)
    # rrank2indices   = remoterank2indices(remoteranks, indices, rank2ghostindices; localrank = rank)

    # # data
    # dataInGhostData = Tuple(map((i, gi) -> begin    st = searchsortedfirst(gi, i[1]); 
    #                                                 ed = st + length(i) - 1;
    #                                                 st:ed; end , indices, ghostindices))
    # data = view(ghostdata, dataInGhostData...)


    # cubes = MPIArray{eltype(ghostdata), typeof(indices), 1}(data, indices, OffsetArray(data, indices), comm, rank, datasize, rank2indices, ghostdata, ghostindices, grank2gindices, rrank2indices)
    # # sync!(cubes)

    # MPI.Barrier(comm)
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

    aggSize     =   (sizePoles, 2, level.nCubes)
    # 分区
    partitation =   slicedim2mpi(aggSize, np)

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
        partitation = slicedim2mpi(aggSize, np)

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

    # 本层的316个转移因子和其索引 OffsetArray 矩阵
    αTrans      =   level.αTrans

    # 局部 pole 的索引和数据
    poleIndices = level.disaggG.indices[1]

    # 更新数据
    level.αTrans = αTrans[poleIndices, :]

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

    ### 先计算聚合相关项
    # 局部的索引和数据
    tpoleIndices = first(tAggS.indices)
    tcubeIndices = last(tAggS.indices)
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
    kAggStransfer   =   ArrayTransfer((tθϕpoleIndices, 1:2, kCubesInterval), kLevel.aggS)

    ### 再计算解聚相关项
    # 局部的索引和数据
    kpoleIndices = kDisAggG.indices[1]
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
    tDisaggStransfer    =   ArrayTransfer((kθϕpoleIndices, 1:2, tCubesInterval), tLevel.disaggG)
    

    ### 更新
    interpWθϕ.θCSC    =   θCSClw
    interpWθϕ.ϕCSC    =   ϕCSClw
    interpWθϕ.θCSCT   =   θCSCTlw
    interpWθϕ.ϕCSCT   =   ϕCSCTlw
    tLevel.phaseShiftFromKids   =   phaseShiftFromKidslw
    tLevel.phaseShift2Kids      =   phaseShift2Kidslw
    kLevel.aggStransfer         =   kAggStransfer
    kLevel.disaggGtransfer      =   tDisaggStransfer

    return nothing

end