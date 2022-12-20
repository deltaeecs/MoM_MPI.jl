using MoM_Kernels:AbstractLevel, CubeInfo, PolesInfo, InterpInfo

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
    cubes       ::MPIVector{CubeInfo{IT, FT}, IDCSCT} where IDCSCT
    cubeEdgel   ::FT
    poles       ::PolesInfo{FT}
    interpWθϕ   ::InterpInfo{IT, FT}
    aggS        ::MPIArray{Complex{FT}, IDCSAT, 3} where IDCSAT
    disaggG     ::MPIArray{Complex{FT}, IDCSDT, 3} where IDCSDT
    phaseShift2Kids     ::Matrix{Complex{FT}}
    phaseShiftFromKids  ::Matrix{Complex{FT}}
    αTrans      ::Matrix{Complex{FT}}
    αTransIndex ::OffsetArray{IT, 3, Array{IT, 3}}
    LevelInfoMPI{IT, FT}() where {IT<:Integer, FT<:Real} = new{IT, FT}()
    LevelInfoMPI{IT, FT}(   ID, L, nCubes, cubes, cubeEdgel, poles, interpWθϕ, aggS, disaggG,
                        phaseShift2Kids, phaseShiftFromKids, αTrans,  αTransIndex) where {IT<:Integer, FT<:Real} = 
                new{IT, FT}(ID, L, nCubes, cubes, cubeEdgel, poles, interpWθϕ, aggS, disaggG,
                        phaseShift2Kids, phaseShiftFromKids, αTrans,  αTransIndex)
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

    # 对盒子按 盒子数 和块数分块
	cubes_ChunksIndices =   sizeChunks2idxs(length(cubes), nchunk)
    # 拿到各块的包含邻盒子的id
    cubesFarNeighbors_ChunksIndices    =   ThreadsX.mapi(chunkIndice -> getNeiFarNeighborCubeIDs(cubes, chunkIndice), cubes_ChunksIndices)
    # 保存
	saveVec2Chunks(cubes, name, cubesFarNeighbors_ChunksIndices; dir = dir)
    
end

function saveLevel(level, nchunk = ParallelParams.nprocs; dir="")

    !ispath(dir) && mkpath(dir)

    # cube要单独处理
    cubes = level.cubes
    level.cubes = eltype(cubes)[]

    saveCubes(cubes, nchunk; name = "Level_$(level.ID)_Cubes", dir=dir)

    # 保存
    jldsave(joinpath(dir, "Level_$(level.ID).jld2"), data = level)

    level.cubes = cubes

    nothing

end

"""
    loadMPICubes(cubefn; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm))

    载入 cubes 并封装为MPI数组
"""
function loadCubes(cubefn; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm))

    # Cubes
    dataloaded  = load(cubefn)
    ghostdata   = dataloaded["data"]
    datasize    = dataloaded["size"]
    ghostindices= dataloaded["indice"]

    # indices
    allindices   = sizeChunks2idxs(datasize, np)
    rank2indices = Dict(zip(0:(np-1), allindices))
    indices = rank2indices[rank]

    # ghost rank to ghost indices
    ghostranks      = indice2ranks(ghostindices, rank2indices)
    grank2gindices  = grank2ghostindices(ghostranks, ghostindices, rank2indices; localrank = rank)

    # remote rank to remote indices
    rank2ghostindices = Dict{Int, typeof(ghostindices)}()
    for rk in ghostranks
        rk == rank && continue
        rank2ghostindices[rk] =  load(replace(cubefn, "part_$(rank+1)" => "part_$(rk+1)"), "indice")
    end
    remoteranks     = indice2ranks(indices, rank2ghostindices)
    rrank2indices   = remoterank2indices(remoteranks, indices, rank2ghostindices; localrank = rank)

    # data
    dataInGhostData = Tuple(map((i, gi) -> begin    st = searchsortedfirst(gi, i[1]); 
                                                    ed = st + length(i) - 1;
                                                    st:ed; end , indices, ghostindices))
    data = view(ghostdata, dataInGhostData...)


    cubes = MPIArray{eltype(ghostdata), typeof(indices), 1}(data, indices, OffsetArray(data, indices), comm, rank, datasize, rank2indices, ghostdata, ghostindices, grank2gindices, rrank2indices)
    # sync!(cubes)

    MPI.Barrier(comm)
    return cubes

end

function loadMPILevel!(level, fn, cubefn; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm))

    # 除了 cubes 之外的信息
    leveltemp = load(fn, "data")
    for n in propertynames(leveltemp)
        try
            setfield!(level, n, getproperty(leveltemp, n))
        catch e
            e == UndefRefError() && continue
        end
    end

    # cubes
    cubes = loadCubes(cubefn; comm = comm, rank = rank, np = np)
    level.cubes = cubes

    return level
end

function loadMPILevel(fn, cubefn; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm))

    level = LevelInfoMPI{Int, Precision.FT}()
    loadMPILevel!(level, fn, cubefn; comm = comm, rank = rank, np = np)

    return level
end