"""
利用多重派发实现
"""
EFIEOnGeoPWCSepPV(geo::T) where {T<:TetrahedraInfo}             =   EFIEOnTetraPWCSepPV(geo)
EFIEOnGeoPWCSepPV(geo::T) where {T<:HexahedraInfo}              =   EFIEOnHexaPWCSepPV(geo)
EFIEOnGeosPWC(geot::T, geos::T) where {T<:TetrahedraInfo}       =   EFIEOnTetrasPWC(geot, geos)
EFIEOnGeosPWC(geot::T, geos::T) where {T<:HexahedraInfo}        =   EFIEOnHexasPWC(geot, geos)
EFIEOnNearGeosPWC(geot::T, geos::T) where {T<:TetrahedraInfo}   =   EFIEOnNearTetrasPWC(geot, geos)
EFIEOnNearGeosPWC(geot::T, geos::T) where {T<:HexahedraInfo}    =   EFIEOnNearHexasPWC(geot, geos)
EFIEOnGeosPWC(geot::T1, geos::T2) where {T1<:TetrahedraInfo, T2<:HexahedraInfo}     =   EFIEOnHexaTetraPWC(geot, geos)
EFIEOnGeosPWC(geot::T1, geos::T2) where {T1<:HexahedraInfo, T2<:TetrahedraInfo}     =   EFIEOnHexaTetraPWC(geot, geos)
EFIEOnNearGeosPWC(geot::T1, geos::T2) where {T1<:TetrahedraInfo, T2<:HexahedraInfo} =   EFIEOnNearHexaTetraPWC(geot, geos)
EFIEOnNearGeosPWC(geot::T1, geos::T2) where {T1<:HexahedraInfo, T2<:TetrahedraInfo} =   EFIEOnNearHexaTetraPWC(geot, geos)


"""
采用 RWG + PWC 基函数计算 三角形 + 四面体/六面体 EFIE 的体积分（VIE）阻抗矩阵近场元并将结果放在Znear稀疏矩阵中
"""
function calZnearChunkEFIEonCube!(iCube::Int, cubes, 
    geos1Info::AbstractVector{TriangleInfo{IT, FT}}, geos2Info::AbstractVector{VT},
    Znear, ::Type{BFT}) where {IT<:Integer, FT<:Real,  VT<:VolumeCellType, BFT<:PWC}
    
    
    # 三角形、四面体数
    ntri    =   length(geos1Info)
    # ngoeV   =   length(geos2Info)
    geos1Interval   =   1:ntri

    # 本盒子信息
    cube    =   cubes[iCube]
    # 常数
    Rsglr       =   Params.Rsglr

    # 盒子里的基函数区间
    cubeBFinterval  =   cube.bfInterval
    # 找出对应的三角形id
    cubeGeoID       =   cube.geoIDs

    # 邻盒子的 geo id
    nearCubesGeoID  =   getGeoIDsInNearCubes(cube, cubes)
    nearCubeBFindices = ZnearChunk.colIndices
    # 对场盒子内三角形循环
    @inbounds for tid in cubeGeoID
        
        # 局域的场四面体
        # tid     =   nearCubesGeoID[iGeo]
        # 场只找三角形
        # tid     >   ntri && continue

        # 场分geo1 geo2
        tin1    =   (tid in geos1Interval)
        tin1 ? begin
            geot1    =   geos1Info[tid]
            # 测试网格元包含的四个测试基函数是否在所有邻盒子（测试盒子）的基函数（测试基函数）区间内
            msInInterval    =   [!isempty(searchsorted(nearCubeBFindices, m)) for m in geot1.inBfsID]
            # 局部判断奇异性距离
            Rsglrlc =   Rsglr/sqrt(norm(geot1.ε)/ε_0)
            # 中心
            centert =   geot1.center
        end : begin
            geot2    =   geos2Info[tid]
            # 测试网格元包含的四个测试基函数是否在所有邻盒子（测试盒子）的基函数（测试基函数）区间内
            msInInterval    =   [!isempty(searchsorted(nearCubeBFindices, m)) for m in geot2.inBfsID]
            # 局部判断奇异性距离
            Rsglrlc =   Rsglr/sqrt(norm(geot2.ε)/ε_0)
            # 中心
            centert =   geot2.center
        end
        # 对源四面体循环
        for sid in nearCubesGeoID
            
            # 源亦分geo1 geo2
            sin1    =   (sid in geos1Interval)
            sin1 ? begin
                geos1    =   geos1Info[sid]
                # 源网格元包含的四个源基函数是否在所有邻盒子（测试盒子）的基函数（测试基函数）区间内
                nsInInterval    =   [n in cubeBFinterval for n in geos1.inBfsID]
                # 局部判断奇异性距离
                Rsglrlc =   Rsglr/sqrt(norm(geos1.ε)/ε_0)
                # 中心
                centers =   geos1.center
            end : begin
                geos2    =   geos2Info[sid]
                # 源四面体包含的四个源基函数是否在所有邻盒子（测试盒子）的基函数（测试基函数）区间内
                nsInInterval    =   [n in cubeBFinterval for n in geos2.inBfsID]
                # 局部判断奇异性距离
                Rsglrlc =   Rsglr/sqrt(norm(geos2.ε)/ε_0)
                # 中心
                centers =   geos2.center
            end

            # 场源距离
            Rts     =   dist(centert, centers)

            if tid == sid
                if tin1
                    # 计算三角形相关的(3*3)个矩阵元的结果
                    Zts  =  EFIEOnTris(geot1)
                    # 写入数据
                    for ni in 1:3, mi in 1:3
                        # 基函数id
                        m = trit.inBfsID[mi]
                        n = tris.inBfsID[ni]
                        # 判断边是不是基函数（边缘不构建半基函数时适用）
                        (m == 0 || n == 0) && continue
                        # 往矩阵填充结果
                        # 判断是不是在源盒子、场盒子包含的区间内
                        ((msInInterval[mi] && nsInInterval[ni])) && begin
                            Znear[m, n] += Zts[mi, ni]
                        end
                    end
                else
                    # 重合
                    Zts, ZtsPV  =   EFIEOnGeoPWCSepPV(geot2)
                    for ni in 1:3
                        n = geos.inBfsID[ni]
                        for mi in 1:3
                            # 基函数id
                            m = geot.inBfsID[mi]
                            # 写入
                            if discreteJ
                                Znear[m, n]  =   Zts[mi, ni]
                            else
                                Znear[m, n]  =   Zts[mi, ni]*κₜ
                            end
                        end
                        if discreteJ
                            Znear[n, n] += ZtsPV/(geot2.ε - ε_0)
                        else
                            Znear[n, n] += ZtsPV/geot2.ε
                        end
                    end
                end
            else
                if tin1
                    if sin1
                        # 需要进行近奇异性处理的场源三角形
                        Zts    =   Rts < Rsglrlc ? EFIEOnNearTris(geot1, geos1) : EFIEOnTris(geot1, geos1)
                        # 写入数据
                        for ni in 1:3, mi in 1:3
                            # 基函数id
                            m = geot1.inBfsID[mi]
                            n = geos1.inBfsID[ni]

                            # 判断边是不是基函数（边缘不算）
                            (m == 0 || n == 0) && continue
                            ## 分布式避免数据通信不再利用对称性填充
                            # 避免线程锁的矩阵元循环方式下产生的条件
                            # (tid > sid) && (m in cubeBFinterval) && continue
                            # 判断是不是在源盒子、场盒子包含的区间内
                            ((msInInterval[mi] & nsInInterval[ni])) && begin
                                Znear[m, n] += Zts[mi, ni]
                            end
                        end
                    else
                        # 需要进行近奇异性处理的场源四面体
                        Zts, _    =   Rts < Rsglrlc ? EFIEOnNearRWGPWC(geot1, geos2) : EFIEOnRWGPWC(geot1, geos2)
                        # 写入数据
                        for ni in 1:3, mi in 1:3
                            # 基函数id
                            m = geot1.inBfsID[mi]
                            n = geos2.inBfsID[ni]
                            # RWG基函数不设半基函数因此跳过
                            (m == 0) && continue
                            # 判断是不是在源盒子、场盒子包含的区间内
                            ((msInInterval[mi] && nsInInterval[ni])) && begin
                                if discreteJ
                                    Znear[m, n]  +=  Zts[mi, ni]
                                else
                                    Znear[m, n]  +=  Zts[mi, ni] * geos2.κ
                                end
                                # Znear[n, m] += Zst[ni, mi]
                            end
                        end
                    end
                else
                    if sin1
                        # 需要进行近奇异性处理的场源四面体
                        Zts, _    =   Rts < Rsglrlc ? EFIEOnNearRWGPWC(geot2, geos1) : EFIEOnRWGPWC(geot2, geos1)
                        # 写入数据
                        for ni in 1:3, mi in 1:3
                            # 基函数id
                            m = geot2.inBfsID[mi]
                            n = geos1.inBfsID[ni]
                            # RWG基函数不设半基函数因此跳过
                            (n == 0) && continue
                            # 判断是不是在源盒子、场盒子包含的区间内
                            ((msInInterval[mi] && nsInInterval[ni])) && begin
                                Znear[m, n]  +=  Zts[mi, ni]
                                # Znear[n, m] += Zst[ni, mi]
                            end
                        end
                    else
                        Zts, _    =   Rts < Rsglrlc ? EFIEOnNearGeosPWC(geot2, geos2) : EFIEOnGeosPWC(geot2, geos2)
                                            # 写入数据
                        for ni in 1:3, mi in 1:3
                            # 基函数id
                            m = geot2.inBfsID[mi]
                            n = geos2.inBfsID[ni]
                            # 写入
                            if discreteJ
                                Znear[m, n]  =   Zts[mi, ni]
                            else
                                Znear[m, n]  =   Zts[mi, ni]*geos2.κ
                            end
                        end
                    end

                end
                
            end # if

        end # sid
    end # tid
    return nothing
end


"""
采用 基函数计算指定层内 EFIE 阻抗矩阵近场元并将结果放在 ZnearChunk 中 (分布式)
"""
function calZnearChunksCFIE!(cubes, geosInfo1::AbstractVector{T1}, geosInfo2::AbstractVector{T2},
    ZnearChunks, bfT::Type{BFT}) where {T1 <: VSCellType, T2 <: VSCellType, BFT<:LinearBasisFunction}
    
    # 本进程索引
    idcs    =   ZnearChunks.indices[1]
    # 本地数据
    cubeslw         =   getGhostGeos(cubes)
    ZnearChunkslc   =   getGhostGeos(ZnearChunks)
    # 进度条
    cond = true
    if cond
        pmeter = Progress(length(idcs); desc = "Cal Z on rank $(ZnearChunks)...", dt = 1, color = :origin)
    end
    # 计算
    @threads for i in idcs
        calZnearChunkEFIEonCube!(i, cubeslw, geosInfo1, geosInfo2, ZnearChunkslc[i], bfT)
        cond && next!(pmeter)
    end
    MPI.Barrier(ZnearChunks.comm)
    nothing

end # function

#! TODO
"""
采用 RWG + PWC + PWC 基函数计算 三角形 + 四面体 + 六面体 EFIE 的体积分（VIE）阻抗矩阵近场元并将结果放在Znear稀疏矩阵中
"""
function calZnearChunkEFIEonCube!(iCube::Int, cubes, 
    geos1Info::AbstractVector{TriangleInfo{IT, FT}}, geos2Info::AbstractVector{VT2}, geos3Info::AbstractVector{VT3},
    Znear, ::Type{BFT}) where {IT<:Integer, FT<:Real,  VT2<:VolumeCellType, VT3<:VolumeCellType, BFT<:PWC}
    
    
    # 三角形、四面体数
    ntri    =   length(geos1Info)
    ngoeV1  =   length(geos2Info)
    geos1Interval   =   1:ntri
    # 是否为偏置数组（用于混合网格）
    isoffset    =   isa(geos2Info, OffsetVector)
    geo2Interval =   begin
        isoffset ? begin
            st  =   (eachindex(geos2Info).offset) + 1
            st:(st - 1 + ngoeV1)
        end : begin
            1:ngoeV1
        end
    end
    ngoeV2  =   length(geos2Info)
    # 是否为偏置数组（用于混合网格）
    isoffset    =   isa(geos3Info, OffsetVector)
    geo3Interval =   begin
        isoffset ? begin
            st  =   (eachindex(geos3Info).offset) + 1
            st:(st - 1 + ngoeV2)
        end : begin
            1:ngoeV2
        end
    end

    # 本盒子信息
    cube    =   cubes[iCube]
    # 常数
    Rsglr       =   Params.Rsglr

    # 盒子里的基函数区间
    cubeBFinterval  =   cube.bfInterval
    # 找出对应的三角形id
    cubeGeoID       =   cube.geoIDs

    # 邻盒子的 geo id
    nearCubesGeoID  =   getGeoIDsInNearCubes(cube, cubes)
    nearCubeBFindices = ZnearChunk.colIndices
    # 排序并剔除冗余元素
    unique!(sort!(nearCubesGeoID))
    # 对场盒子内三角形循环
    @inbounds for tid in cubeGeoID
        
        # 局域的场四面体
        # tid     =   nearCubesGeoID[iGeo]
        # 场只找三角形
        # tid     >   ntri && continue

        # 场分geo1 geo2
        tin1    =   (tid in geos1Interval)
        tin1 ? begin
            geot1    =   geos1Info[tid]
            # 测试网格元包含的四个测试基函数是否在所有邻盒子（测试盒子）的基函数（测试基函数）区间内
            msInInterval    =   [!isempty(searchsorted(nearCubeBFindices, m)) for m in geot1.inBfsID]
            # 局部判断奇异性距离
            Rsglrlc =   Rsglr/sqrt(norm(geot1.ε)/ε_0)
            # 中心
            centert =   geot1.center
        end : begin
            geot2    =   geos2Info[tid]
            # 测试网格元包含的四个测试基函数是否在所有邻盒子（测试盒子）的基函数（测试基函数）区间内
            msInInterval    =   [!isempty(searchsorted(nearCubeBFindices, m)) for m in geot2.inBfsID]
            # 局部判断奇异性距离
            Rsglrlc =   Rsglr/sqrt(norm(geot2.ε)/ε_0)
            # 中心
            centert =   geot2.center
        end
        # 对源四面体循环
        for sid in nearCubesGeoID
            
            # 源亦分geo1 geo2
            sin1    =   (sid in geos1Interval)
            sin1 ? begin
                geos1    =   geos1Info[sid]
                # 源网格元包含的四个源基函数是否在所有邻盒子（测试盒子）的基函数（测试基函数）区间内
                nsInInterval    =   [n in cubeBFinterval for n in geos1.inBfsID]
                # 局部判断奇异性距离
                Rsglrlc =   Rsglr/sqrt(norm(geos1.ε)/ε_0)
                # 中心
                centers =   geos1.center
            end : begin
                geos2    =   geos2Info[sid]
                # 源四面体包含的四个源基函数是否在所有邻盒子（测试盒子）的基函数（测试基函数）区间内
                nsInInterval    =   [n in cubeBFinterval for n in geos2.inBfsID]
                # 局部判断奇异性距离
                Rsglrlc =   Rsglr/sqrt(norm(geos2.ε)/ε_0)
                # 中心
                centers =   geos2.center
            end

            # 场源距离
            Rts     =   dist(centert, centers)

            if tid == sid
                if tin1
                    # 计算三角形相关的(3*3)个矩阵元的结果
                    Zts  =  EFIEOnTris(geot1)
                    # 写入数据
                    for ni in 1:3, mi in 1:3
                        # 基函数id
                        m = trit.inBfsID[mi]
                        n = tris.inBfsID[ni]
                        # 判断边是不是基函数（边缘不构建半基函数时适用）
                        (m == 0 || n == 0) && continue
                        # 往矩阵填充结果
                        # 判断是不是在源盒子、场盒子包含的区间内
                        ((msInInterval[mi] && nsInInterval[ni])) && begin
                            Znear[m, n] += Zts[mi, ni]
                        end
                    end
                else
                    # 重合
                    Zts, ZtsPV  =   EFIEOnGeoPWCSepPV(geot2)
                    for ni in 1:3
                        n = geos.inBfsID[ni]
                        for mi in 1:3
                            # 基函数id
                            m = geot.inBfsID[mi]
                            # 写入
                            if discreteJ
                                Znear[m, n]  =   Zts[mi, ni]
                            else
                                Znear[m, n]  =   Zts[mi, ni]*κₜ
                            end
                        end
                        if discreteJ
                            Znear[n, n] += ZtsPV/(geot2.ε - ε_0)
                        else
                            Znear[n, n] += ZtsPV/geot2.ε
                        end
                    end
                end
            else
                if tin1
                    if sin1
                        # 需要进行近奇异性处理的场源三角形
                        Zts    =   Rts < Rsglrlc ? EFIEOnNearTris(geot1, geos1) : EFIEOnTris(geot1, geos1)
                        # 写入数据
                        for ni in 1:3, mi in 1:3
                            # 基函数id
                            m = geot1.inBfsID[mi]
                            n = geos1.inBfsID[ni]

                            # 判断边是不是基函数（边缘不算）
                            (m == 0 || n == 0) && continue
                            ## 分布式避免数据通信不再利用对称性填充
                            # 避免线程锁的矩阵元循环方式下产生的条件
                            # (tid > sid) && (m in cubeBFinterval) && continue
                            # 判断是不是在源盒子、场盒子包含的区间内
                            ((msInInterval[mi] & nsInInterval[ni])) && begin
                                Znear[m, n] += Zts[mi, ni]
                            end
                        end
                    else
                        # 需要进行近奇异性处理的场源四面体
                        Zts, _    =   Rts < Rsglrlc ? EFIEOnNearRWGPWC(geot1, geos2) : EFIEOnRWGPWC(geot1, geos2)
                        # 写入数据
                        for ni in 1:3, mi in 1:3
                            # 基函数id
                            m = geot1.inBfsID[mi]
                            n = geos2.inBfsID[ni]
                            # RWG基函数不设半基函数因此跳过
                            (m == 0) && continue
                            # 判断是不是在源盒子、场盒子包含的区间内
                            ((msInInterval[mi] && nsInInterval[ni])) && begin
                                if discreteJ
                                    Znear[m, n]  +=  Zts[mi, ni]
                                else
                                    Znear[m, n]  +=  Zts[mi, ni] * geos2.κ
                                end
                                # Znear[n, m] += Zst[ni, mi]
                            end
                        end
                    end
                else
                    if sin1
                        # 需要进行近奇异性处理的场源四面体
                        Zts, _    =   Rts < Rsglrlc ? EFIEOnNearRWGPWC(geot2, geos1) : EFIEOnRWGPWC(geot2, geos1)
                        # 写入数据
                        for ni in 1:3, mi in 1:3
                            # 基函数id
                            m = geot2.inBfsID[mi]
                            n = geos1.inBfsID[ni]
                            # RWG基函数不设半基函数因此跳过
                            (n == 0) && continue
                            # 判断是不是在源盒子、场盒子包含的区间内
                            ((msInInterval[mi] && nsInInterval[ni])) && begin
                                Znear[m, n]  +=  Zts[mi, ni]
                                # Znear[n, m] += Zst[ni, mi]
                            end
                        end
                    else
                        Zts, _    =   Rts < Rsglrlc ? EFIEOnNearGeosPWC(geot2, geos2) : EFIEOnGeosPWC(geot2, geos2)
                                            # 写入数据
                        for ni in 1:3, mi in 1:3
                            # 基函数id
                            m = geot2.inBfsID[mi]
                            n = geos2.inBfsID[ni]
                            # 写入
                            if discreteJ
                                Znear[m, n]  =   Zts[mi, ni]
                            else
                                Znear[m, n]  =   Zts[mi, ni]*geos2.κ
                            end
                        end
                    end

                end
                
            end # if

        end # sid
    end # tid
    return nothing
end

