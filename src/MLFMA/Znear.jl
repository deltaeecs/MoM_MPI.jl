using MoM_Kernels:ZNEARCHUNK, MatrixChunk, CubeInfo

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
    chunks  ::MPIVector{MatrixChunk{T}, Tuple{UnitRange{Int64}}, 
                            SubArray{MatrixChunk{T}, 1, Vector{MatrixChunk{T}}, Tuple{UnitRange{Int64}}, true}, Tuple{UnitRange{Int64}}}
    lmul    ::Vector{T}
    lmuld   ::MPIVector{T, Tuple{UnitRange{Int64}}, SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}, Tuple{Vector{Int64}}}
    rmul    ::Vector{T}
    rmuld   ::MPIVector{T, Tuple{UnitRange{Int64}}, SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int64}}, true}, Tuple{Vector{Int64}}}

    ZnearChunksStructMPI{T}() where {T} = new{T}()
    ZnearChunksStructMPI{T}(m, n, nChunks, chunks, lmul, lmuld, rmul, rmuld) where {T} = 
                        new{T}(m, n, nChunks, chunks, lmul, lmuld, rmul, rmuld)

end

"""
ZnearChunksStructMPI 类的初始化函数，将 lumld 和 rmuld 初始化为
"""

function ZnearChunksStructMPI{T}(chunks; m, n, comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm)) where {T<:Number}

    Znear   =   ZnearChunksStructMPI{T}()
    nChunks =   length(chunks)

    Znear.m = m
    Znear.n = n
    Znear.nChunks = nChunks
    Znear.chunks = chunks

    Znear
end

"""
    getGhostICeoffVecs(y::MPIVector{T, I}) where {T, I}
    
	这里必须注明类型以稳定计算，仅限本包使用。
TBW
"""
function getGhostICeoffVecs(y::MPIVector{T, I, DT, IG}) where {T, I, DT, IG}
    sparsevec(y.ghostindices[1], y.ghostdata)::SparseVector{T, Int}
end

"""
    getGhostICeoffVecs(y::SubMPIVector{T, I, DT, IG, SI, L}) where {T, I, DT, IG, SI, L}
    
	这里必须注明类型以稳定计算，仅限本包使用。
TBW
"""
function getGhostICeoffVecs(y::SubMPIVector{T, I, DT, IG, SI, L}) where {T, I, DT, IG, SI, L}
    yp = y.parent
	sparsevec(yp.ghostindices[1], yp.ghostdata[:, y.indices[2]])::SparseVector{T, Int}
end

function mul_core!(Z, xghost)

    nthds   =   nthreads()
    BLAS.set_num_threads(1)
    cubeIndices = Z.chunks.indices[1]
    @inbounds for ii in cubeIndices
        Zchunk = Z.chunks[ii]
        mul!(Zchunk.rmul, Zchunk, xghost)
        setindex!(Z.rmuld, Zchunk.rmul, Zchunk.rowIndices)
    end
    BLAS.set_num_threads(nthds)

    nothing

end

"""
实现左乘其它向量
"""
function Base.:*(Z::T, x::MPIVector) where{T<:ZnearChunksStructMPI}
    # 同步向量
    sync!(x)
    # 获取本进程用到的部分
    xghost = getGhostMPIVecs(x)
    # 计算
    mul_core!(Z, xghost)
    MPI.Barrier(x.comm)
    # 输出向量同步
    sync!(Z.rmuld)

    return deepcopy(Z.rmuld)
end

"""
实现左乘其它向量，仅限本包使用。
"""
function Base.:*(Z::T, x::SubMPIVector) where{T<:ZnearChunksStructMPI}
    # 同步向量
    syncUnknownVectorView!(x)
    # 获取本进程用到的部分
    xghost = getGhostMPIVecs(x)
    # 计算
    mul_core!(Z, xghost)
    MPI.Barrier(x.comm)
    # 输出向量同步
    sync!(Z.rmuld)
    return deepcopy(Z.rmuld)
end

function LinearAlgebra.mul!(y::SubOrMPIVector, Z::T, x::MPIVector) where{T<:ZnearChunksStructMPI}
    sync!(x)
    xghost = getGhostICeoffVecs(x)    
    # initialZchunksMulV!(Z)
    mul_core!(Z, xghost)

    MPI.Barrier(x.comm)
    copyto!(getdata(y), Z.rmuld.data)
    
    return y
end

function LinearAlgebra.mul!(y::SubOrMPIVector, Z::T, x::SubMPIVector) where{T<:ZnearChunksStructMPI}
    syncUnknownVectorView!(x)
    xghost = getGhostICeoffVecs(x)
    # initialZchunksMulV!(Z)
    mul_core!(Z, xghost)

    MPI.Barrier(x.parent.comm)
    copyto!(getdata(y), Z.rmuld.data)

    return y
end


"""
根据八叉树盒子信息初始化 cube 对应的近场矩阵元块儿
"""
function initialZnearChunks(cube, cubes::PartitionedVector; CT = Complex{Precision.FT})

    # 确定近场元的总数、计算列起始位置指针
    rowIndices  =   cube.bfInterval

    # 对其邻盒循环
    colIndices  =   Int[]
    for jNearCube in cube.neighbors
        # 邻盒子信息
        nearCube    =   cubes[jNearCube]
        # 对邻盒子中测试基函数数量累加
        append!(colIndices, nearCube.bfInterval)
    end # jNearCube
    unique!(sort!(colIndices))

    return MatrixChunk{CT}(rowIndices, colIndices)
end

"""
根据八叉树盒子信息初始化 cube 对应的近场矩阵元块儿，不分配 ghost 数据因为会带来大约双倍内存需求
"""
function initialZnearChunksMPI(level; nbf, CT = Complex{Precision.FT}, comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm))

    @info "Initializing Znear on rank $rank..."
    # 本进程的 ghostchunks (包含了在计算预条件时用到的远亲)
    cubes       =   level.cubes

    # chunks 的 MPI 分结构与 cubes 类似, 因此可利用 cubes 的分布信息来构建
    indices     =   (cubes.indices, )
    datasize    =   (cubes.size, )
    allindices      =   MPI.Allgather((cubes.indices, ), comm)
    rank2indices    =   Dict(zip(0:(np-1), allindices))

    # ghostdata 仅需要邻盒子部分
    ghostindices = indices
    ghostdata   =  [initialZnearChunks(cubes[i], cubes; CT = CT) for i in indices[1]]

    # ghost rank to ghost indices (空)
    grank2gindices  = typeof(rank2indices)()
    # 最终获得远程进程所需数据在本进程存储的 indices 上的位置 (空)
    rrank2indices   = typeof(rank2indices)()
    dataInGhostData = Tuple(map((i, gi) -> begin    st = searchsortedfirst(gi, i[1]); 
                                                    ed = st + length(i) - 1;
                                                    st:ed; end , indices, ghostindices))
    data = view(ghostdata, dataInGhostData...)

    chunks  = MPIArray{eltype(ghostdata), typeof(indices), 1, typeof(data), typeof(ghostindices)}( 
                        data, indices, OffsetArray(data, indices), comm, 
                        rank, datasize, rank2indices, ghostdata, ghostindices, grank2gindices, rrank2indices)

    Znear   = ZnearChunksStructMPI{CT}(chunks; m = nbf, n = nbf)
    initialZchunksMulV!(Znear, level)
    @info "Znear initialized on rank $rank."
    MPI.Barrier(comm)

    return Znear
end

"""
    MPIvecOnLevel(level; T = Precision.CT, comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm))

生成按层内盒子的划分分配的 MPI 向量, 该向量的 ghostdata 包括邻盒子部分，用于矩阵向量乘积计算。
"""
function MPIvecOnLevel(cubes::PartitionedVector{C}; T = Precision.CT, comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm)) where {C<:CubeInfo}

    # 所有涉及的邻盒子 id
    nearCubesIndices = getNeighborCubeIDs(cubes, cubes.indices)

    # 本进程的盒子、邻盒子包含的基函数数量
    ngbfslw      =   sum(idc -> length(cubes[idc].bfInterval), nearCubesIndices[1]; init = 0)
    # 分配内存
    ghostdata   =   zeros(T, ngbfslw)

    # 本地 vector 的区间
    indices         =  (first(cubes.data).bfInterval[1]:last(cubes.data).bfInterval[end], )
    allindices      =   MPI.Allgather(indices, comm)
    rank2indices    =   Dict(zip(0:(np-1), allindices))
    datasize        =   last.(last(allindices))

    # vector + ghost data 的区间
    ghostindices    =   map(nearCubesIndice -> reduce(vcat, [cubes[idc].bfInterval for idc in nearCubesIndice]), nearCubesIndices)
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

    y = MPIArray{eltype(ghostdata), typeof(indices), 1, typeof(data), typeof(ghostindices)}(
                data, indices, OffsetArray(data, indices), comm, rank, datasize, rank2indices, ghostdata, ghostindices, grank2gindices, rrank2indices)

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


"""
根据积分方程类型选择相应计算函数
"""
function calZnearChunks!(cubes, geosInfo::AbstractVector{VSCellT}, 
    ZnearChunks, bfT::Type{BFT} = VSBFTypes.sbfType) where {BFT<:BasisFunctionType, VSCellT<:SurfaceCellType}
    if SimulationParams.ieT == :EFIE
        # 计算 RWG下 的 EFIE 阻抗矩阵
        calZnearChunksEFIE!(cubes, geosInfo, ZnearChunks.chunks, bfT)
    elseif SimulationParams.ieT == :MFIE
        # 计算 RWG下 的 MFIE 阻抗矩阵
        calZnearChunksMFIE!(cubes, geosInfo, ZnearChunks.chunks, bfT)
    elseif SimulationParams.ieT == :CFIE
        # 计算 RWG下 的 CFIE 阻抗矩阵
        calZnearChunksCFIE!(cubes, geosInfo, ZnearChunks.chunks, bfT)
    end
    return nothing
end

"""
根据积分方程类型选择相应计算函数
"""
function calZnearChunks!(cubes, geosInfo::AbstractVector{VSCellT}, 
    ZnearChunks, bfT::Type{BFT} = VSBFTypes.vbfType) where {BFT<:BasisFunctionType, VSCellT<:VolumeCellType}
    # 计算 SWG/PWC/RBF 下的 EFIE 阻抗矩阵
    calZnearChunksEFIE!(cubes, geosInfo, ZnearChunks.chunks, bfT)
    nothing
end

"""
根据积分方程类型选择相应计算函数
"""
function calZnearChunks!(cubes, geosInfo::AbstractVector{VT}, 
    ZnearChunks, bfT::Type{BFT} = VSBFTypes.vbfType) where {BFT<:BasisFunctionType, VT<:AbstractVector}
    # 计算 RWG + PWC/RBF 下的 EFIE 阻抗矩阵
    calZnearChunksEFIE!(cubes, geosInfo..., ZnearChunks.chunks, bfT)
    nothing
end

include("ZChunks/ZnearChunkEFIE.jl")
include("ZChunks/ZnearChunkMFIE.jl")
include("ZChunks/ZnearChunkCFIE.jl")
include("ZChunks/ZnearChunkEFIEVSIE.jl")