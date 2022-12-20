using MoM_Kernels:ZNEARCHUNK, MatrixChunk

"""
创建近场矩阵结构体，所包含的数据为所有盒子内的近场矩阵元，保存的是分布式数据
```
m::Int，行数
n::Int，列数
nChunks::Int，矩阵块儿数
chunks::Vector{MatrixChunk{T}}，矩阵
lmul::Vector{T}，用于左乘其它矩阵、向量的临时数组，大小与列数相同
lmuld::Vector{T}，用于左乘其它矩阵、向量的临时分布式数组，大小与列数相同，默认不分配
rmul::Vector{T}，用于右乘其它矩阵、向量的临时数组，大小与行数相同
lmuld::Vector{T}，用于左乘其它矩阵、向量的临时分布式数组，大小与列数相同，默认不分配
```
"""
mutable struct ZnearChunksStructMPI{T<:Number} <:ZNEARCHUNK{T}
    m   ::Int
    n   ::Int
    nChunks ::Int
    chunks  ::MPIVector{MatrixChunk{T}, I} where I
    lmul    ::Vector{T}
    lmuld   ::MPIVector{T}
    rmul    ::Vector{T}
    rmuld   ::MPIVector{T}

    ZnearChunksStructMPI{T}() where {T} = new{T}()
    ZnearChunksStructMPI{T}(m, n, nChunks, chunks, lmul, lmuld, rmul, rmuld) where {T} = 
                        new{T}(m, n, nChunks, chunks, lmul, lmuld, rmul, rmuld)

end

"""
ZnearChunksStructMPI 类的初始化函数，将 lumld 和 rmuld 初始化为
"""

function ZnearChunksStructMPI{T}(chunks; m, n, comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm)) where {T<:Number}

    nChunks =   length(chunks)
    lmul    =   zeros(T, 0)
    lmuld   =   mpiarray(T, np; partitation = np, buffersize = 0)
    rmul    =   zeros(T, 0)
    rmuld   =   mpiarray(T, np; partitation = np, buffersize = 0)

    ZnearChunksStructMPI{T}(m, n, nChunks, chunks, lmul, lmuld, rmul, rmuld)
end

getGhostCubes(cubes) = sparsevec(cubes.ghostindices[1], cubes.ghostdata)

"""
根据八叉树盒子信息初始化 cube 对应的近场矩阵元块儿
"""
function initialZnearChunks(cube, cubes::MPIVector; CT = Complex{Precision.FT})

    # 确定近场元的总数、计算列起始位置指针
    rowIndices  =   cube.bfInterval

    # 本地数据
    cubeslw     =   getGhostCubes(cubes)

    # 对其邻盒循环
    colIndices  =   Int[]
    for jNearCube in cube.neighbors
        # 邻盒子信息
        nearCube    =   cubeslw[jNearCube]
        # 对邻盒子中测试基函数数量累加
        append!(colIndices, nearCube.bfInterval)
    end # jNearCube
    unique!(sort!(colIndices))

    return MatrixChunk{CT}(rowIndices, colIndices)
end

"""
根据八叉树盒子信息初始化 cube 对应的近场矩阵元块儿
"""
function initialZnearChunksMPI(level; nbf, CT = Complex{Precision.FT}, comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm))

    # 本进程的 ghostchunks (包含了在计算预条件时用到的远亲)
    cubes       =   level.cubes
    ghostcubes  =   getGhostCubes(cubes)

    # chunks 的 MPI 分结构与 cubes 类似, 因此可利用 cubes 的分布信息来构建
    indices     =   cubes.indices
    datasize    =   cubes.size
    rank2indices=   cubes.rank2indices

    # ghostdata 仅需要邻盒子部分
    ghostindices = getNeighborCubeIDs(cubes, indices)
    ghostdata   =  [initialZnearChunks(ghostcubes[i], cubes; CT = CT) for i in ghostindices[1]]

    # ghost rank to ghost indices
    ghostranks      = indice2ranks(ghostindices, rank2indices)
    grank2gindices  = grank2ghostindices(ghostranks, ghostindices, rank2indices; localrank = rank)

    # remote rank to ghost indices
    rank2ghostindices  =  get_rank2ghostindices(ghostranks, indices, ghostindices; comm = comm, rank = rank)
    # 最终获得远程进程所需数据在本进程存储的 indices 上的位置
    remoteranks     = indice2ranks(indices, rank2ghostindices)
    rrank2indices   = remoterank2indices(remoteranks, indices, rank2ghostindices; localrank = rank)
    dataInGhostData = Tuple(map((i, gi) -> begin    st = searchsortedfirst(gi, i[1]); 
                                                    ed = st + length(i) - 1;
                                                    st:ed; end , indices, ghostindices))
    data = view(ghostdata, dataInGhostData...)

    chunks  = MPIArray{eltype(ghostdata), typeof(indices), 1
                        }(  data, indices, OffsetArray(data, indices), comm, 
                            rank, datasize, rank2indices, ghostdata, ghostindices, grank2gindices, rrank2indices)

    Znear   = ZnearChunksStructMPI{CT}(chunks; m = nbf, n = nbf)
    initialZchunksMulV!(Znear, level)

    MPI.Barrier(comm)

    return Znear
end

"""
    MPIvecOnLevel(level; T = Precision.CT, comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm))


生成按层内盒子的划分分配的 MPI 向量, 该向量的 ghostdata 包括邻盒子部分，用于矩阵向量乘积计算。
"""
function MPIvecOnLevel(cubes::MPIVector{C, I}; T = Precision.CT, comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm)) where {C<:CubeInfo, I}

    # 本进程MPI盒子
    cubeslw = getGhostCubes(cubes)

    # 所有涉及的邻盒子 id
    nearCubesIndices = getNeighborCubeIDs(cubes, cubes.indices)

    # 本进程的盒子、邻盒子包含的基函数数量
    ngbfslw      =   sum(idc -> length(cubeslw[idc].bfInterval), nearCubesIndices[1]; init = 0)
    # 分配内存
    ghostdata   =   zeros(T, ngbfslw)

    # 本地 vector 的区间
    indices         =  (first(cubes.data).bfInterval[1]:last(cubes.data).bfInterval[end], )
    allindices      =   MPI.Allgather(indices, comm)
    rank2indices    =   Dict(zip(0:(np-1), allindices))
    datasize        =   last.(last(allindices))
    
    # vector + ghost data 的区间
    ghostindices    =   map(nearCubesIndice -> reduce(vcat, [cubeslw[idc].bfInterval for idc in nearCubesIndice]), nearCubesIndices)
    dataInGhostData =   Tuple(map((i, gi) -> begin  st = searchsortedfirst(gi, i[1]); 
                                                    ed = st + length(i) - 1;
                                                    st:ed; end , indices, ghostindices))
    data = view(ghostdata, dataInGhostData...)
    # ghost rank to ghost indices
    ghostranks      = indice2ranks(ghostindices, rank2indices)
    grank2gindices  = grank2ghostindices(ghostranks, ghostindices, rank2indices; localrank = rank)
    # remote rank to ghost indices
    rank2ghostindices  =  get_rank2ghostindices(ghostranks, indices, ghostindices; comm = comm, rank = rank)
    # 最终获得远程进程所需数据在本进程存储的 indices 上的位置
    remoteranks     = indice2ranks(indices, rank2ghostindices)
    rrank2indices   = remoterank2indices(remoteranks, indices, rank2ghostindices; localrank = rank)

    y = MPIArray{eltype(ghostdata), typeof(indices), 1}(data, indices, OffsetArray(data, indices), comm, rank, datasize, rank2indices, ghostdata, ghostindices, grank2gindices, rrank2indices)

    return y

end

"""
    MPIvecOnLevel(level; T = Precision.CT, comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm))


生成按层内盒子的划分分配的 MPI 向量, 该向量的 ghostdata 包括邻盒子部分，用于矩阵向量乘积计算。
"""
function MPIvecOnLevel(level::L; T = Precision.CT, comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm)) where L<:LevelInfoMPI

    return MPIvecOnLevel(level.cubes; T =T, comm = comm, rank = rank, np = np)

end


"""
初始化 阻抗矩阵 左乘 向量 乘积的 分布式数组
"""
function initialZchunksMulV!(Z::T, level) where{T<:ZnearChunksStructMPI}

    Z.rmuld = MPIvecOnLevel(level)
    nothing
end


