"""
本进程在叶层从基函数向盒子聚合
"""
function MoM_Kernels.aggOnBF!(level::LT, aggSBF, IVec::MPIVector{T, I}) where {LT<:LevelInfoMPI, T<:Number, I}
    # 置零避免累加错误
    fill!(aggS.ghostdata, 0)
    
    # 叶层盒子
    cubes   =   getLocalDArgs(level.cubes)
    # 叶层聚合项
    aggS    =   level.aggS
    # 本进程的基函数聚合项
    aggSBFlw    =   aggSBF.dataview

    # 本地索引
    poleIndices, _, cubeIndices = aggS.indices
    # 本地数据
    aggSlc  =   aggS.dataview

    ## 本 rank 的IVec
    IVeclw  =   IVec.dataview

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
            for dr in 1:2, ipx in poleIndices
                aggSCube[ipx, dr]   +=  In*aggSBFlw[ipx, dr, n]
            end
            # @views aggS[:, :, iCube]   .+=   IVec[n] .* aggSBF[:, :, n]
        end #n
    end #iCube
    
    return nothing
end

"""
从子层聚合到本层
tLevel :: 本层
kLevel :: 子层
"""
function MoM_Kernels.agg2HighLevel!(tLevel::LT, kLevel::LT) where {LT<:LevelInfoMPI}
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

    # 局部的索引和数据
    poleIndices, _, cubeIndices = tAggS.indices
    
    ## 本地化本进程用到的子层辐射积分
    # 先找出所有的子盒子
    kCubesInterval  =   first(cubes[cubeIndices[1]].kidsInterval):last(cubes[cubeIndices[end]].kidsInterval)
    # 本进程用到的所有子盒子的辐射积分
    kAggSlw =   kAggS.ghostdata
    
    # 预分配内存以加速
    nthds   =   nthreads()
    aggSInterpedϕs   =   zeros(CT, size(ϕCSClw, 1), size(kAggS, 2), nthds)
    aggSInterpeds    =   zeros(CT, size(θCSClw, 1), size(kAggS, 2), nthds)
    
    tAggSlc =   tAggS.dataview
    ## 置零避免累加错误
    tAggSlc.parent .= 0
    # 对盒子循环
    BLAS.set_num_threads(1)#
    @threads for iCube in cubeIndices
        cube    =   cubes[iCube]
        # 本线程的临时变量
        tid     =   Threads.threadid()
        # kAggSlc =   zeros(CT, size(kAggS, 1), size(kAggS, 2))#kAggSlcs[:, :, tid]
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
            
            # @show typeof(kAggS), typeof(kAggS[:, :, kCubeID])
            @views kAggSlc =  kAggSlw[:, :, kCubeID - kCubesInterval[1] + 1]
            # kAggSlc .= kAggS[:, :, kCubeID]
            mul!(aggSInterpedϕ, ϕCSC, kAggSlc)
            mul!(aggSInterped, θCSC, aggSInterpedϕ)

            # 插值完毕进行子层到本层的相移，累加进父盒子聚合项
            @views tAggSlc[:,:,iCube]   .+=   phaseShiftFromKids[:, kIn8] .* aggSInterped

        end # jkCube
    end #iCube

    BLAS.set_num_threads(nthds)
    nothing

end # function



"""
从叶层聚合到第 '2' 层
"""
function MoM_Kernels.agg2Level2!(levels, nLevels)
    # 对层循环进行计算
    for iLevel in (nLevels - 1):-1:2
        # 本层
        tLevel  =   levels[iLevel]
        # 子层
        kLevel  =   levels[iLevel + 1]
        # 计算
        agg2HighLevel!(tLevel, kLevel)
    end #iLevel
end # function

"""
本进程层内转移
"""
function MoM_Kernels.transOnLevel!(level::LT) where {LT<:LevelInfoMPI}
    # 层信息
    cubes   =   getLocalDArgs(level.cubes)
    # aggS    =   Array(level.aggS)
    aggS    =   level.aggS
    disaggG =   level.disaggG
    # 本层的316个转移因子和其索引 OffsetArray 矩阵
    αTrans  =   getLocalDArgs(level.αTrans)
    αTransIndex =   getLocalDArgs(level.αTransIndex)

    # 局部的索引和数据
    poleIndices, _, cubeIndices = localindices(disaggG)
    disaggGlc   =   OffsetArray(localpart(disaggG), 0, 0, cubeIndices[1] - 1)
    aggSlc      =   OffsetArray(localpart(aggS), 0, 0, cubeIndices[1] - 1)
    # 置零避免累加错误
    disaggGlc  .=   0

    # 先 fetch 不是本 work 的 aggS
    # 首先，找出所有非本进程的远亲
    farNeisdw   =   Int[]
    for iCube in cubeIndices
        cube    =   cubes[iCube]
        farNeighborIDs = cube.farneighbors
        # 对远亲循环
        for iFarNei in farNeighborIDs
            !(iFarNei in cubeIndices) && push!(farNeisdw, iFarNei)
        end
    end
    unique!(sort!(farNeisdw))
    aggSdws     =   Array(aggS[poleIndices, :, farNeisdw])
    # @show cubeIndices, length(cubes),  length(cubeIndices), length(farNeisdw)
    # @show farNeisdw
    
    # 对盒子循环
    @threads for iCube in cubeIndices
        cube    =   cubes[iCube]
        farNeighborIDs = cube.farneighbors
        # 本线程数据
        aggSi  =   zeros(eltype(aggS), length(poleIndices), 2)#size(aggS, 1), 2)
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
                idx = first(searchsorted(farNeisdw, iFarNei))
                @views copyto!(aggSi, aggSdws[:, :, idx])
            end
            # 转移
            @views disaggGlc[:, :, iCube]    .+=   αTrans[poleIndices, i1d] .* aggSi
        end # iFarNei
    end #iCube


    nothing
end #function

"""
各层内转移
"""
function MoM_Kernels.transOnLevels!(levels::LT, nLevels) where {LT <: LevelInfoMPI}
    for iLevel in 2:nLevels
        # 层信息
        level   =   levels[iLevel]
        # 计算
        transOnLevel!(level)
    end #iLevel
end #function

"""
向低层解聚
tLevel :: 本层
kLevel :: 子层
"""
function MoM_Kernels.disagg2KidLevel!(tLevel, kLevel)

    # 本层信息
    cubes   =   getLocalDArgs(tLevel.cubes)
    tDisaggG    =   tLevel.disaggG
    # 子层信息
    kDisAggG    =   kLevel.disaggG
    CT          =   Complex{typeof(tLevel.cubeEdgel)}
    # 从本层到子层的相移
    phaseShift2Kids =   localpart(tLevel.phaseShift2Kids)[1]
    # 本层到子层的稀疏反插值矩阵
    interpWθϕ   =   localpart(kLevel.interpWθϕ)[1]
    θCSCT    =   interpWθϕ.θCSCT
    ϕCSCT    =   interpWθϕ.ϕCSCT

    # 局部的索引和数据
    poleIndices, _, cubeIndices = localindices(kDisAggG)
    kDisaggGSlcOff  =   OffsetArray(localpart(kDisAggG), 0, 0, cubeIndices[1] - 1)
    
    ## 局部插值矩阵
    # ϕ 方向
    ϕCSCTlwtemp     =   ϕCSCT[poleIndices, :]
    # 用到的 θ 方向的 poleIndices
    θpoleIndices    =   filter(i -> !iszero(ϕCSCTlwtemp.colptr[i+1] - ϕCSCTlwtemp.colptr[i]), 1:ϕCSCTlwtemp.n)
    ϕCSCTlw         =   ϕCSCTlwtemp[:, θpoleIndices]

    θCSCTlwtemp     =   θCSCT[θpoleIndices, :]
    # 用到的 ϕ 方向的 poleIndices
    θϕpoleIndices   =   filter(i -> !iszero(θCSCTlwtemp.colptr[i+1] - θCSCTlwtemp.colptr[i]), 1:θCSCTlwtemp.n)
    θCSCTlw         =   θCSCTlwtemp[:, θϕpoleIndices]
    ## 局部相移矩阵
    phaseShift2Kidslw    =   phaseShift2Kids[θϕpoleIndices,:]

    ## 本地化本进程用到的子层辐射积分
    # 先找出所有的父盒子索引
    tCubesIntervallw    =   last(searchsorted(cubes, first(cubeIndices); by = func4Cube1stkInterval)):first(searchsorted(cubes, last(cubeIndices); by = func4CubelastkInterval))
    # 本进程用到的所有盒子的辐射积分
    tDisaggGlw     =   Array(tDisaggG[θϕpoleIndices, :, tCubesIntervallw])

    # 分配内存
    nthds   =   nthreads()
    BLAS.set_num_threads(1)
    # tCubeDisaggGs   =   zeros(CT, size(tDisaggG, 1), size(tDisaggG, 2), nthds)
    disGshifted2s   =   zeros(CT, size(tDisaggGlw, 1), size(tDisaggGlw, 2), nthds)
    disGInterpedθ2s =   zeros(CT, size(θCSCTlw, 1), size(tDisaggGlw, 2), nthds)
    disGInterped2s  =   zeros(CT, size(ϕCSCTlw, 1), size(tDisaggGlw, 2), nthds)
    # 对盒子循环
    @threads for iCube in tCubesIntervallw
        cube    =   cubes[iCube]
        # 子盒子数
        nkCube  =   length(cube.kidsInterval)
        
        # 本线程数据
        tid =   threadid()
        # tCubeDisaggG    =   zeros(CT, size(tDisaggG, 1), size(tDisaggG, 2))#tCubeDisaggGs[:, :, tid]
        @views disGshifted2     =   disGshifted2s[:, :, tid]
        @views disGInterpedθ2   =   disGInterpedθ2s[:, :, tid]
        @views disGInterped2    =   disGInterped2s[:, :, tid]
        # 复制数据到本地
        @views tCubeDisaggG     =   tDisaggGlw[:,:,iCube - first(tCubesIntervallw) + 1]

        # 进程 id
        # 对子盒子循环
        for jkCube in 1:nkCube
            # 子盒子id
            kCubeID =   cube.kidsInterval[jkCube]
            !(kCubeID in cubeIndices) && continue
            # 子盒子在8个子盒子中的id
            kIn8    =   cube.kidsIn8[jkCube]

            # 进行本层到子层的相移，累加进父盒子聚合项
            @views disGshifted2 .= phaseShift2Kidslw[:,kIn8] .* tCubeDisaggG
            # disGInterpedθ2 .= θCSCT * disGshifted2
            # disGInterped2  .= ϕCSCT * disGInterpedθ2
            mul!(disGInterpedθ2, θCSCTlw, disGshifted2)
            mul!(disGInterped2, ϕCSCTlw, disGInterpedθ2)

            # @views kdisAggGAnterped     =   ϕCSCT*(θCSCT*(phaseShift2Kids[:,kIn8] .* tCubeDisaggG))
            kDisaggGSlcOff[:,:,kCubeID] .+= disGInterped2
        end # jkCube
    end #iCube
    # 还原线程
    BLAS.set_num_threads(nthds)

    nothing
end #function

"""
解聚到叶层
"""
function MoM_Kernels.disagg2LeafLevel!(levels, nLevels)

    # 对层循环进行计算
    for iLevel in 2:(nLevels - 1)
        # 本层
        tLevel  =   levels[iLevel]
        # 子层
        kLevel  =   levels[iLevel + 1]
        # 计算
        disagg2KidLevelD!(tLevel, kLevel)
    end #iLevel

end #function


"""
在叶层往测试基函数解聚
"""
function MoM_Kernels.disaggOnBF!(level, disaggSBF, ZID)
    CT = Complex{eltype(level.cubeEdgel)}
    
    # 叶层盒子
    cubes   =   getLocalDArgs(level.cubes)
    # 叶层解聚项
    disaggG =   level.disaggG
    # 多极子数
    nPoles  =   size(disaggG, 1)
    # 常量
    JK_0η::CT   =   Params.JK_0*η_0
    
    # 本地索引
    _, _, cubeIndices = localindices(disaggG)
    # 本进程的基函数聚合项
    disaggSBFlc =   OffsetArray(localpart(disaggSBF), 0, 0, first(localindices(disaggSBF)[3]) - 1)
    disaggGlc   =   OffsetArray(localpart(disaggG), 0, 0, cubeIndices[1] - 1)
    ZIDlc       =   OffsetVector(localpart(ZID), first(localindices(ZID)[1]) - 1)

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
            for idx in 1:nPoles
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
function calZfarI!(Zopt::MLMFAIterator{ZT, MT}, IVec::MPIVector{T, I}; setzero = true) where {ZT, MT<:MPIVector, T<:Number, I}
    
    # 计算前置零
    setzero && fill!(Zopt.ZI, zero(T))

    # 基函数聚合到叶层
    aggOnBFD!(Zopt.leafLevel, Zopt.aggSBF, IVec)
    # 聚合到2层
    agg2Level2D!(Zopt.levels, Zopt.nLevels)
    # 层间转移
    transOnLevelsD!(Zopt.levels, Zopt.nLevels)
    # 解聚到叶层
    disagg2LeafLevelD!(Zopt.levels, Zopt.nLevels)
    # 解聚到基函数
    disaggOnBFD!(Zopt.leafLevel, Zopt.disaggSBF, Zopt.ZI)
    
    return Zopt.ZI
end
