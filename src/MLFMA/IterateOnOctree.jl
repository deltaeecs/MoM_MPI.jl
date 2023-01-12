using MoM_Kernels:  aggOnBF!, agg2HighLevel!, agg2Level2!, 
                    transOnLevel!, transOnLevels!, 
                    disagg2KidLevel!, disagg2LeafLevel!, disaggOnBF!, calZfarI!

"""
本进程在叶层从基函数向盒子聚合
"""
function MoM_Kernels.aggOnBF!(level::LT, aggSBF, IVeclw::T) where {LT<:LevelInfoMPI, T<:AbstractVector}

    # 叶层盒子
    cubes   =   level.cubes
    # 叶层聚合项
    aggS    =   level.aggS
    # 置零避免累加错误
    fill!(aggS.ghostdata, 0)
    # 本进程的基函数聚合项
    aggSBFlw    =   aggSBF.dataOffset

    # 本地索引
    poleIndices, iθϕIndices, cubeIndices = aggS.indices
    # 本地数据
    aggSlc  =   aggS.dataOffset

    @threads for iCube in cubeIndices
        # 盒子信息
        cube    =   cubes[iCube]
        # 基函数区间
        bfInterval  =   cube.bfInterval
        # 盒子的聚合项
        aggSCube    =   @view aggSlc[:, :, iCube]

        # 往盒子中心聚合
        for n in bfInterval
            In = IVeclw[n]
            for dr in iθϕIndices, ipx in poleIndices
                aggSCube[ipx, dr]   +=  In*aggSBFlw[ipx, dr, n]
            end
        end #n
    end #iCube
    
    return nothing
end

function MoM_Kernels.aggOnBF!(level::LT, aggSBF, IVec::MPIVector{T, I, DT, IG}) where {LT<:LevelInfoMPI, T<:Number, I, DT, IG}
    ## 本 rank 的IVec
    IVeclw  =   IVec.dataOffset

    aggOnBF!(level, aggSBF, IVeclw)
    
    return nothing
end

function MoM_Kernels.aggOnBF!(level::LT, aggSBF, IVec::SubMPIVector) where {LT<:LevelInfoMPI}
    ## 本 rank 的IVec
    IVeclw  =   view(IVec.parent.dataOffset, :, IVec.indices[2])

    aggOnBF!(level, aggSBF, IVeclw)

    return nothing
end

"""
    setPatternData_form_localDataAndTransfer!(targetData, index, localdata, transfer)

TBW
"""
function setPatternData_form_localDataAndTransfer!(targetData, iseq_reqs_idcs, iCube, data, datatransfer)
    # 置零
    fill!(targetData, 0)
    
    if iseq_reqs_idcs[1] && iseq_reqs_idcs[2] #多极子方向完全匹配, θ ϕ 方向匹配 
        if  (iseq_reqs_idcs[3]) || (iCube in data.indices[3])  # 数据完全在 data 里面
            @views copyto!(targetData, data.dataOffset[:, :, iCube])
        else  # 数据完全在 datatransfer 里面
            spdata =  datatransfer.reqsDatas[iCube]
            setindex!(targetData, spdata.nzval, :, :)
        end
    else # 多极子方向不匹配 或 θ ϕ  方向不匹配
        # 创建 ArrayChunk
        targetDataChunk = ArrayChunk(targetData, datatransfer.reqsIndices[1], datatransfer.reqsIndices[2])
        ## 首先将 data 里的数据填充进 targetData
        if  (iseq_reqs_idcs[3]) || (iCube in data.indices[3])  # 部分数据在 data 里面
            dataOffset = data.dataOffset[:, :, iCube]
            for j in intersect(axes(dataOffset, 2), datatransfer.reqsIndices[2]), i in intersect(axes(dataOffset, 1), datatransfer.reqsIndices[1])
                targetDataChunk[i, j] = dataOffset[i, j]
            end
        end

        ## 其次将 datatransfer 里面数据的数据导出
        # 该数据仅位于 spdata
        if all(i -> sortedVecInUnitRange(datatransfer.reqsIndices[i], data.indices[i]), 1:2) && !(iCube in data.indices[3])
            spdata =  datatransfer.reqsDatas[iCube]
            setindex!(targetData, spdata.nzval, :, :)
        end
        # 该数据部分位于 spdata
        if any(i -> !sortedVecInUnitRange(datatransfer.reqsIndices[i], data.indices[i]), 1:2)
            
            # 首先判断
            spdata =  datatransfer.reqsDatas[iCube]
            for j in axes(spdata, 2)
                colptr = spdata.colptr
                is = view(spdata.rowval, colptr[j]:(colptr[j+1]-1))
                zs = view(spdata.nzval,  colptr[j]:(colptr[j+1]-1))
                for (i, z) in zip(is, zs)
                    targetDataChunk[i, j] = z
                end
            end
        end
    end

    nothing
end

"""
从子层聚合到本层
tLevel :: 本层
kLevel :: 子层
"""
function MoM_Kernels.agg2HighLevel!(tLevel::LT, kLevel::LT; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm)) where {LT<:LevelInfoMPI}
    # 本层信息
    cubes   =   tLevel.cubes
    tAggS   =   tLevel.aggS
    
    CT      =   eltype(tAggS)
    # 子层信息
    kAggS   =   kLevel.aggS
    # 从子层到本层的相移
    phaseShiftFromKids  =   tLevel.phaseShiftFromKids
    # 子层到本层的稀疏插值矩阵
    interpWθϕ   =   kLevel.interpWθϕ
    θCSC    =   interpWθϕ.θCSC
    ϕCSC    =   interpWθϕ.ϕCSC

    # 局部的索引
    cubeIndices = cubes.indices

    # 本进程用到的所有子盒子的辐射积分
    kAggIndices     =   kAggS.indices
    kAggSTransfer   =   kLevel.aggStransfer
    # 同步数据
    sync!(kAggSTransfer)
    # 本进程用到的所有子盒子的辐射积分的索引
    kreqsIndices    =   kAggSTransfer.reqsIndices
    # 检查需求索引与本进程提供索引是否相同
    iseq_reqs_idcs  =   map(isequal, kreqsIndices, kAggIndices)
    
    # 预分配内存以加速
    nthds   =   nthreads()
    aggSInterpedϕs  =   zeros(CT, size(ϕCSC, 1), size(kAggS, 2), nthds)
    aggSInterpeds   =   zeros(CT, size(θCSC, 1), size(kAggS, 2), nthds)
    
    # 本层待计算聚合项
    tAggSlc =   tAggS.ghostdata
    tCubeOffset =   first(tAggS.indices[3]) - 1
    ## 置零避免累加错误
    fill!(tAggS.ghostdata, 0)
    # 对盒子循环
    BLAS.set_num_threads(1)
    @threads for iCube in cubeIndices
        cube    =   cubes[iCube]
        # 本线程的临时变量
        tid     =   Threads.threadid()
        @views kAggSlc        =   zeros(CT, map(length, kreqsIndices[1:2]))
        @views aggSInterpedϕ  =   aggSInterpedϕs[:, :, tid]
        @views aggSInterped   =   aggSInterpeds[:, :, tid]

        # 对子盒子循环
        # 子盒子数
        nkCube  =   length(cube.kidsInterval)
        for jkCube in 1:nkCube
            # 子盒子id
            kCubeID =   cube.kidsInterval[jkCube]
            # 子盒子在8个子盒子中的id
            kIn8    =   cube.kidsIn8[jkCube]
            # 根据索引设置
            setPatternData_form_localDataAndTransfer!(kAggSlc, iseq_reqs_idcs, kCubeID, kAggS, kAggSTransfer)
            mul!(aggSInterpedϕ, ϕCSC, kAggSlc)
            mul!(aggSInterped, θCSC, aggSInterpedϕ)

            # 插值完毕进行子层到本层的相移，累加进父盒子聚合项
            iCubelc = iCube - tCubeOffset
            @views tAggSlc[:, :, iCubelc]   .+=   phaseShiftFromKids[:, kIn8] .* aggSInterped
        end # jkCube
    end #iCube
    MPI.Barrier(comm)

    BLAS.set_num_threads(nthds)
    nothing

end # function


"""
本进程层内转移
"""
function MoM_Kernels.transOnLevel!(level::LT) where {LT<:LevelInfoMPI}
    # 层信息
    cubes   =   level.cubes
    # aggS    =   Array(level.aggS)
    aggS    =   level.aggS
    disaggG =   level.disaggG
    # 本层的316个转移因子和其索引 OffsetArray 矩阵
    αTrans  =   level.αTrans
    αTransTransfer  =   level.αTransTransfer
    sync!(αTransTransfer)
    αTransIndex =   level.αTransIndex

    # 局部的索引和数据
    poleIndices, iθϕIndices, cubeIndices = disaggG.indices
    disaggGlc   =   disaggG.ghostdata
    cubeOffset  =   first(cubeIndices) - 1
    aggSlc      =   aggS.dataOffset
    # 置零避免累加错误
    fill!(disaggG.ghostdata, 0)

    # 对盒子循环
    @threads for iCube in cubeIndices
        cube    =   cubes[iCube]
        farNeighborIDs = cube.farneighbors
        # 本线程数据
        aggSi  =   zeros(eltype(aggS), length(poleIndices), length(iθϕIndices))
        # 对远亲循环
        for iFarNei in farNeighborIDs
            # 远亲盒子3DID
            farNeiCube3D    =   cubes[iFarNei].ID3D
            # 本盒子相对远亲盒子id
            relative3DID    =   cube.ID3D - farNeiCube3D
            # 相对 id
            i1d =   αTransIndex[relative3DID[1], relative3DID[2], relative3DID[3]]
            # 将数据转移到本地
            if iFarNei in cubeIndices
                @views copyto!(aggSi, aggSlc[:, :, iFarNei])
            else
                spdata =  αTransTransfer.reqsDatas[iFarNei]
                setindex!(aggSi, spdata.nzval, :, :)
            end
            # 转移
            @views disaggGlc[:, :, iCube - cubeOffset]    .+=   αTrans[:, i1d] .* aggSi
        end # iFarNei
    end #iCube


    nothing
end #function


"""
向低层解聚
tLevel :: 本层
kLevel :: 子层
"""
function MoM_Kernels.disagg2KidLevel!(tLevel::LT, kLevel::LT; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm)) where {LT <: LevelInfoMPI}

    # 本层信息
    cubes   =   tLevel.cubes
    tDisaggG    =   tLevel.disaggG
    tDisaggIndices     =   tDisaggG.indices
    tDisaggGtransfer    =   tLevel.disaggGtransfer
    # 同步数据
    sync!(tDisaggGtransfer)
    # 本进程用到的所有子盒子的辐射积分的索引
    treqsIndices    =   tDisaggGtransfer.reqsIndices

    # 检查需求索引与本进程提供索引是否相同
    iseq_reqs_idcs  =   map(isequal, treqsIndices, tDisaggIndices)
    # 子层信息
    kDisAggG    =   kLevel.disaggG
    CT          =   eltype(kDisAggG)
    # 从本层到子层的相移
    phaseShift2Kids =   tLevel.phaseShift2Kids
    # 本层到子层的稀疏反插值矩阵
    interpWθϕ   =   kLevel.interpWθϕ
    θCSCT    =   interpWθϕ.θCSCT
    ϕCSCT    =   interpWθϕ.ϕCSCT

    # 子层局部的索引和数据
    poleIndices, _, cubeIndices = kDisAggG.indices
    kDisaggGlw      =   kDisAggG.ghostdata
    kCubeOffset     =   first(cubeIndices) - 1
    
    ## 本地化本进程用到的子层辐射积分
    # 先找出所有的父盒子索引
    tCubesIntervallw    =   tDisaggGtransfer.reqsIndices[3]
    # 本进程用到的所有盒子的辐射积分
    tDisaggGlw      =   tDisaggG.ghostdata
    # 判断需求区间与拥有数据区间是否匹配
    iseq_reqs_idcs  =   map(isequal, treqsIndices, tDisaggIndices)

    # 分配内存
    nthds   =   nthreads()
    BLAS.set_num_threads(1)
    disGshifted2s   =   zeros(CT, map(length, treqsIndices[1:2])..., nthds)
    disGInterpedθ2s =   zeros(CT, size(θCSCT, 1), size(tDisaggGlw, 2), nthds)
    disGInterped2s  =   zeros(CT, size(ϕCSCT, 1), size(tDisaggGlw, 2), nthds)
    # 对盒子循环
    @threads for iCube in tCubesIntervallw
        cube    =   cubes[iCube]
        # 子盒子数
        nkCube  =   length(cube.kidsInterval)
        
        # 本线程数据
        tid =   threadid()
        tCubeDisaggG     =   zeros(CT, map(length, treqsIndices[1:2])...)
        @views disGshifted2     =   disGshifted2s[:, :, tid]
        @views disGInterpedθ2   =   disGInterpedθ2s[:, :, tid]
        @views disGInterped2    =   disGInterped2s[:, :, tid]
        # 复制数据到本地
        setPatternData_form_localDataAndTransfer!(tCubeDisaggG, iseq_reqs_idcs, iCube, tDisaggG, tDisaggGtransfer)

        # 进程 id
        # 对子盒子循环
        for jkCube in 1:nkCube
            # 子盒子id
            kCubeID =   cube.kidsInterval[jkCube]
            !(kCubeID in cubeIndices) && continue
            # 子盒子在8个子盒子中的id
            kIn8    =   cube.kidsIn8[jkCube]

            # 进行本层到子层的相移，累加进父盒子聚合项 
            # @views disGshifted2 .= phaseShift2Kids[:,kIn8] .* tCubeDisaggG
            for dr in axes(tCubeDisaggG, 2), ipx in axes(tCubeDisaggG, 1)
                disGshifted2[ipx, dr]   =  phaseShift2Kids[ipx, kIn8]*tCubeDisaggG[ipx, dr]
            end

            # disGInterpedθ2 .= θCSCT * disGshifted2
            # disGInterped2  .= ϕCSCT * disGInterpedθ2
            mul!(disGInterpedθ2, θCSCT, disGshifted2)
            mul!(disGInterped2, ϕCSCT, disGInterpedθ2)

            # @views kdisAggGAnterped     =   ϕCSCT*(θCSCT*(phaseShift2Kids[:,kIn8] .* tCubeDisaggG))
            @views kDisaggGlw[:,:,kCubeID - kCubeOffset] .+= disGInterped2
        end # jkCube
    end #iCube
    MPI.Barrier(comm)
    # 还原线程
    BLAS.set_num_threads(nthds)

    nothing
end #function


"""
在叶层往测试基函数解聚
"""
function MoM_Kernels.disaggOnBF!(level::LT, disaggSBF, ZID) where {LT <: LevelInfoMPI}
    CT = eltype(disaggSBF)
    
    # 叶层盒子
    cubes   =   level.cubes
    # 叶层解聚项
    disaggG =   level.disaggG

    # 常量
    JK_0η::CT   =   Params.JK_0*η_0
    
    # 本地索引
    poleIndices, _, cubeIndices = disaggG.indices
    # 本进程的基函数聚合项
    disaggSBFlc =   disaggSBF.dataOffset
    disaggGlc   =   disaggG.dataOffset
    ZIDlc       =   ZID.dataOffset

    # 置零避免累加错误
    # ZIDlc.parent    .=   0
    # 将本函数内的BLAS设为单线程
    nthds = nthreads()
    # 避免多线程内存分配问题
    ZInTemps = zeros(CT, nthds)
    # 对盒子循环计算
    @threads for iCube in cubeIndices
        tid     =   Threads.threadid()
        # 盒子信息
        cube    =   cubes[iCube]
        # 基函数区间
        bfInterval  =   cube.bfInterval
        # 该盒子解聚项
        @views disaggGCube =  disaggGlc[:, :, iCube]
        # 往基函数解聚
        for n in bfInterval
            ZInTemps[tid]   = 0
            ZInTemp     =   ZInTemps[tid]
            for idx in poleIndices
                ZInTemp += disaggSBFlc[idx, 1, n] * disaggGCube[idx, 1] + disaggSBFlc[idx, 2, n] * disaggGCube[idx, 2]
            end
            ZInTemp *=  JK_0η
            # 计算完毕写入数据
            ZIDlc[n]   +=  ZInTemp
        end #n
    end #iCube

    return nothing
end

"""
计算远区矩阵向量乘积
"""
function MoM_Kernels.calZfarI!(Zopt::MLMFAIterator{ZT, MT}, IVec::SubOrMPIVector; setzero = true) where {ZT, MT<:MPIVector}
    
    # 计算前置零
    setzero && fill!(Zopt.ZI, zero(ZT))

    # 基函数聚合到叶层
    aggOnBF!(Zopt.leafLevel, Zopt.aggSBF, IVec)
    # 聚合到2层
    agg2Level2!(Zopt.levels, Zopt.nLevels)
    # 层间转移
    transOnLevels!(Zopt.levels, Zopt.nLevels)
    # 解聚到叶层
    disagg2LeafLevel!(Zopt.levels, Zopt.nLevels)
    # 解聚到基函数
    disaggOnBF!(Zopt.leafLevel, Zopt.disaggSBF, Zopt.ZI)
    
    return Zopt.ZI
end
