"""
计算某层聚合项, 输入为三角形信息和 RWG 基函数信息
"""
function aggSBFOnLevelCFIE(level, trianglesInfo::AbstractVector{TriangleInfo{IT, FT}}, 
    bfT) where {IT<:Integer, FT<:Real}
    CT  =   Complex{FT}
    aggSBF, disaggSBF = allocatePatternOnLeaflevel(level)
    # 计算
    aggSBFOnLevelCFIE!(aggSBF, disaggSBF, level, trianglesInfo, bfT)

    return aggSBF, disaggSBF
end

"""
计算某层聚合项, 输入为三角形信息和 RWG 基函数信息
"""
function MoM_Kernels.aggSBFOnLevelCFIE!(aggSBF::MPIArray, disaggSBF::MPIArray, level, trianglesInfo::AbstractVector{TriangleInfo{IT, FT}}, 
    ::Type{BFT}) where {IT<:Integer, FT<:Real, BFT<:RWG}
    CT  =   Complex{FT}

    # 本层盒子信息
    cubes   =   level.cubes
    # 本进程分配到的盒子id     
    cubeIndices::UnitRange{Int} =   level.cubes.indices[1]
    # 本进程分配到的 pattern 数据
    aggSBFlw    =   OffsetArray(aggSBF.ghostdata, aggSBF.indices)
    disaggSBFlw =   OffsetArray(disaggSBF.ghostdata, disaggSBF.indices)
    # 层采样点
    polesr̂sθsϕs =   level.poles.r̂sθsϕs
    # CFIE 混合系数
    α   =   Params.CFIEα

    # poles索引
    polesIndices    =   eachindex(polesr̂sθsϕs)

    # 三角形高斯求积权重
    weightTridiv2   =   TriGQInfo.weight / 2
    # 常数
    JK_0 = Params.JK_0
    ntri = length(trianglesInfo)

    # 进度条
    pmeter  =   Progress(length(cubeIndices), "Agg on rank $(aggSBF.myrank) RWG (CFIE)...")
    @threads for iCube in cubeIndices 

        # 盒子
        cube    =   cubes[iCube]
        # 盒子中心
        cubeCenter  =   cube.center

        # 盒子里的基函数区间
        cubeBFinterval  =   cube.bfInterval
        # 找出对应的三角形id
        cubeTriID       =   cube.geoIDs
        # 排序并剔除冗余元素
        # unique!(sort!(cubeTriID))
        # 对盒子内三角形循环
        for iTri in eachindex(cubeTriID)
            it = cubeTriID[iTri]
            # 超出区间跳过
            it > ntri && continue
            # 三角形
            tri =   trianglesInfo[it]
            # 面外法向量
            n̂   =   tri.facen̂
            # 高斯求积点
            rgs =   getGQPTri(tri)
            # 盒子中心到求积点向量
            cubeC2rgs   =   zero(rgs)
            for gi in 1:GQPNTri
                cubeC2rgs[:, gi]   .=   view(rgs, :, gi) .- cubeCenter
            end
            # 预分配
            ρcn̂ck̂   =   zero(MVec3D{FT})
            # 对三角形上的基函数循环
            for ni in 1:3
                # 基函数编号
                n   =   tri.inBfsID[ni]
                # 基函数不在该盒子的基函数区间则跳过
                !(n in cubeBFinterval) && continue
                # ln
                ln  =   tri.edgel[ni]
                # ρs
                ρs  =   zero(rgs)
                @views for gi in 1:GQPNTri
                    ρs[:, gi]   .=   rgs[:, gi] .- tri.vertices[:,ni]
                end

                # 对多极子循环计算
                for iPole in polesIndices
                    # 该多极子
                    poler̂θϕ =   polesr̂sθsϕs[iPole]
                    # 聚合项初始化
                    aggSθ   =   zero(CT)
                    aggSϕ   =   zero(CT)
                    disaggSθ   =   zero(CT)
                    disaggSϕ   =   zero(CT)
                    # 对高斯求积点循环
                    for gi in 1:GQPNTri
                        ρi   =   ρs[:, gi]
                        # 公用的 指数项和权重边长
                        expWlntemp  =   exp(JK_0*(poler̂θϕ.r̂ ⋅ cubeC2rgs[:,gi]))*(weightTridiv2[gi]*ln)
                        # 在 θϕ 方向累加
                        @views aggSθ += (poler̂θϕ.θhat ⋅ ρi)*expWlntemp
                        @views aggSϕ += (poler̂θϕ.ϕhat ⋅ ρi)*expWlntemp
                        # ρi + ρi × n̂ × k̂
                        ρcn̂ck̂  .=   α * ρi
                        ρcn̂ck̂ .+=   (1 - α) * acrossbcrossc(ρi, n̂, poler̂θϕ.r̂)
                        @views disaggSθ += (poler̂θϕ.θhat ⋅ ρcn̂ck̂)*expWlntemp'
                        @views disaggSϕ += (poler̂θϕ.ϕhat ⋅ ρcn̂ck̂)*expWlntemp'
                    end # gi 
                    # 将结果写入目标数组
                    aggSBFlw[iPole, 1, n]    +=  aggSθ
                    aggSBFlw[iPole, 2, n]    +=  aggSϕ
                    disaggSBFlw[iPole, 1, n]     +=  disaggSθ
                    disaggSBFlw[iPole, 2, n]     +=  disaggSϕ
                end # iPole
            end # ni
        end #iTri
        # 更新进度条
        next!(pmeter)
    end #iCube
    return nothing
end
