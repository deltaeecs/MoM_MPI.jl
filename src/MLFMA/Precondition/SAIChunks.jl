# 本文件提供计算稀疏近似逆预条件的函数
"""
计算稀疏近似逆 (Sparse Approximate Inverse (SAI)) 的函数
输入为近场阻抗矩阵CSC, 叶层信息 (也可以为非叶层, 但计算量更大) 
ZnearChunks::ZnearChunksStruct{CT}
level::LevelInfoMPI
该函数提供左预条件
"""
function MoM_Kernels.sparseApproximateInversePl(ZnearChunks::ZnearChunksStructMPI{CT}, level::LT; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm)) where {FT<:Real, CT<:Complex{FT}, LT<:LevelInfoMPI}
    sparseApproximateInversePl(ZnearChunks, level.cubes; comm = comm, rank = rank)
end

"""
计算稀疏近似逆 (Sparse Approximate Inverse (SAI)) 的函数
输入为近场阻抗矩阵CSC, 叶层信息 (也可以为非叶层, 但计算量更大) 
ZnearChunks::ZnearChunksStruct{CT}
cubes::AbstractVector{CubeInfo{IT, FT}}
该函数提供左预条件
"""
function MoM_Kernels.sparseApproximateInversePl(ZnearChunks::ZnearChunksStructMPI{CT}, cubes::AbstractVector{CBT}; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm)) where {FT<:Real, CT<:Complex{FT}, CBT<:CubeInfo}

    # 将本函数内的BLAS设为单线程
    nthds = nthreads()
    BLAS.set_num_threads(1)
    # 首先预分配结果, 采用与 ZnearCSC 相同的稀疏模式
    preM    =   deepcopy(ZnearChunks)

    # 所有盒子内的近场矩阵大小
    chunks      =   ZnearChunks.chunks
    # 获取所有盒子内的近场矩阵大小
    mat_sizes   =   gather_Independent_Vectors(map(size, chunks.data))
    # 获取所有近场元的行列索引
    allRowIndices   =   gather_Independent_Vectors(map(ck -> ck.rowIndices, chunks.data))
    allColIndices   =   gather_Independent_Vectors(map(ck -> ck.colIndices, chunks.data))
    MPI.Barrier(comm)

    # 本进程所有的盒子的 Indices
    indices    =   cubes.indices

    #=
    对 MPI 程序，这里要先将其他进程需要的数据采用非阻塞式通信发送，然后在循环内计算时逐个接收。
    =#
    # 本进程所有盒子的邻盒子的 id
    cubesNeiIDs     =   unique!(sort!(reduce(vcat, map(i -> cubes[i].neighbors, indices))))
    # 本进程所有盒子的邻盒子及其邻盒子的 id
    cubesNeisNeiIDs =   unique!(sort!(reduce(vcat, map(i -> cubes[i].neighbors, cubesNeiIDs))))

    # 本进程需要通信获取矩阵元数据的盒子
    ghostindices    =   (setdiff(cubesNeisNeiIDs, indices), )

    # 各个进程拥有的盒子区间
    ranksindices    =   MPI.Allgather((indices, ), comm)
    rank2indices    =   Dict([(i-1) => ranksindices[i] for i in eachindex(ranksindices)])

    # 本进程需要与其通信获取矩阵元的其它进程的 rank
    ghostranks      =   indice2ranks(ghostindices, rank2indices)
    # 所有进程需要与其通信获取矩阵元的其它进程的盒子 id
    rank2ghostindices  =  get_rank2ghostindices(ghostranks, (indices, ), ghostindices; comm = comm, rank = rank)
    
    # 最终获得远程进程所需数据在本进程存储的 indices 上的位置
    remoteranks     =   indice2ranks((indices, ), rank2ghostindices)
    rrank2indices   =   remoterank2indices(remoteranks, (indices, ), rank2ghostindices; localrank = rank)
    # 转换回全局坐标
    rrank2globalindices =   deepcopy(rrank2indices)
    for (_, v) in rrank2globalindices
        v[1] .= v[1] .+  (indices[1] - 1)
    end

    #  传输矩阵数据
    reqs = MPI.Request[]
    for (grank, cubeids) in rrank2globalindices
        append!(reqs, map(i -> MPI.Isend(chunks[i].mat, grank, i, comm), cubeids[1]))
    end

    ghostZmats =   Dict(map(i -> i => zeros(CT, mat_sizes[i]...), ghostindices[1]))
    indice2rrank(idc)   =  searchsortedfirst(ranksindices, idc, by = ris -> last(first(ris))) - 1
    append!(reqs, map(i -> MPI.Irecv!(ghostZmats[i], indice2rrank(i), i, comm), ghostindices[1]))

    MPI.Waitall(reqs)
    MPI.Barrier(comm)

    # # 传输矩阵行索引数据
    # reqs = MPI.Request[]
    # for (grank, cubeids) in rrank2globalindices
    #     append!(reqs, map(i -> MPI.Isend(chunks[i].rowIndices, grank, i, comm), cubeids[1]))
    # end
    # # 接收矩阵数据
    # ghostZmats =   Dict(map(i -> i => zeros(CT, mat_sizes[i]), ghostindices[1]))
    # indice2rrank(idc)   =  searchsortedfirst(ranksindices, idc, by = ris -> last(first(ris))) - 1
    # append!(reqs, map(i -> MPI.Irecv!(ghostZmats[i], indice2rrank(i), i, comm), ghostindices[1]))
    # MPI.Waitall(reqs)
    # MPI.Barrier(comm)


    # 进度条
    pmeter      =   Progress(length(indices); dt = 1, desc = "Pl (MPI) on rank $(rank)...", barglyphs=BarGlyphs("[=> ]"), color = :blue)
    
    # Znn ZnnH 预分配内存
    Znnpre       =   zeros(CT, 1)
    ZnnHZnnpre   =   zeros(CT, 1)
    

    # 对所有本进程盒子循环
    for iCube in indices
        # 本盒子与所有邻盒子id
        cube    =   cubes[iCube]
        ineiIDs =   cube.neighbors

        # 本盒子与所有邻盒子基函数的 id
        neibfs  = reduce(vcat, map(i -> cubes[i].bfInterval, ineiIDs))
        # 本盒子所有邻盒子基函数的数量
        nNeibfs = length(neibfs)

        # 本盒子与所有邻及其邻盒子 id
        iNeisNeiIDs  = reduce(vcat, map(i -> cubes[i].neighbors, ineiIDs))
        unique!(sort!(iNeisNeiIDs))

        # 本盒子与所有邻盒子及其邻盒子基函数的 id
        neisNeibfs  = reduce(vcat, map(i -> cubes[i].bfInterval, iNeisNeiIDs))
        # 本盒子与所有邻盒子及其邻盒子基函数的数量
        nNeisNeibfs = length(neisNeibfs)

        # 本盒子基函数
        cbfs        =   cube.bfInterval
        # 本盒子基函数起始点在 nneibfs 的位置
        cbfsInCnnei =   cbfs .+ (searchsortedfirst(neisNeibfs, cbfs.start) - cbfs.start)

        length(Znnpre) < nNeibfs*nNeisNeibfs && resize!(Znnpre, nNeibfs*nNeisNeibfs)
        fill!(Znnpre, 0)
        Znn     =   reshape(view(Znnpre, 1:nNeibfs*nNeisNeibfs), nNeibfs, nNeisNeibfs)
        # 提取对应的阻抗矩阵
        for cubeNeiNei in ineiIDs
            # 矩阵块儿行列索引
            rowIndices   =  allRowIndices[cubeNeiNei]
            colIndices   =  allColIndices[cubeNeiNei]
            # 判断数据在本地还是传输过来的
            mat = if cubeNeiNei in indices
                ZChunkNeiNei =  chunks[cubeNeiNei]
                ZChunkNeiNei.mat
            else
                ghostZmats[cubeNeiNei]
            end
            # 往 Znn 填充数据
            for (i, m) in enumerate(rowIndices)
                miZnn   =   searchsortedfirst(neibfs, m)
                for (j, n) in enumerate(colIndices)
                    njZnn   =   searchsortedfirst(neisNeibfs, n)
                    Znn[miZnn, njZnn] = mat[i, j]
                end
            end
        end

        ZnnH    =   Znn'
        # 对 Znn * ZnnH 预分配内存
        length(ZnnHZnnpre) < nNeibfs*nNeibfs && resize!(ZnnHZnnpre, nNeibfs*nNeibfs)
        ZnnHZnn     =   reshape(view(ZnnHZnnpre, 1:nNeibfs*nNeibfs), nNeibfs, nNeibfs)
        # Q
        Qi      =   inv(mul!(ZnnHZnn, Znn, ZnnH))

        # 计算并写入结果
        mul!(preM.chunks[iCube].mat, view(ZnnH, cbfsInCnnei, :), Qi)

        # 更新进度条
        next!(pmeter)

    end # iCube
    # 恢复BLAS默认线程以防影响其他多线程函数
    BLAS.set_num_threads(nthds)
    # 保存预条件类型
    open(SimulationParams.resultDir*"/InputArgs.txt", "a+")  do f
        write(f, "\npreT:\tSAI")
    end

    MPI.Barrier(comm)

    return SAIChunkPrec{CT}(preM)
end
