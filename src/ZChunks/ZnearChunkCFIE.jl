using MoM_Kernels:CFIEOnTris, CFIEOnNearTris

"""
采用 RWG 基函数计算指定盒子内 CFIE 面积分（SIE）阻抗矩阵近场元并将结果放在 ZnearChunk 中
"""
function calZnearChunkCFIEonCube!(iCube::Int, cubes, 
    geosInfo::AbstractVector{TriangleInfo{IT, FT}},
    ZnearChunk, ::Type{BFT}) where {IT<:Integer, FT<:Real, BFT<:LinearBasisFunction}
    
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
    # 本地数据
    nearCubeBFindices = ZnearChunk.colIndices
    # 对盒子内三角形循环
    # @inbounds for iGeo in eachindex(view(geosInfo, cubeGeoID))
    @inbounds for tid in cubeGeoID
        # 局域的场三角形
        # tid   =   cubeGeoID[iGeo]
        geot  =   geosInfo[tid]
        #= 场三角形与源三角形在不在一个盒子？因为程序利用了PEC目标的CFIE矩阵的对称性
        进行对称位置阻抗矩阵元的计算，要避免对同一个盒子内阻抗矩阵元的重复计算 =#
        # tins  =   nearCubesGeoID[iGeo] in cubeGeoID
        # tins  =   !isempty(searchsorted(cubeGeoID, nearCubesGeoID[iGeo]))
        # 测试三角形包含的三个测试基函数是否在所有邻盒子（测试盒子）的基函数（测试基函数）区间内
        msInInterval    =   [m in cubeBFinterval for m in geot.inBfsID]
        # @show msInInterval
        # for jGeo in eachindex(view(geosInfo, nearCubesGeoID))
        for sid in nearCubesGeoID
            # sid =   nearCubesGeoID[jGeo]
            # 源三角形
            geos    =   geosInfo[sid]
            # 场源距离
            Rts     =   dist(geot.center, geos.center)
            # 源三角形包含的三个源基函数是否在所有邻盒子（测试盒子）的基函数（测试基函数）区间内
            nsInInterval    =   [!isempty(searchsorted(nearCubeBFindices, n)) for n in geos.inBfsID]
            # 判断二者远近，调用不同精度的矩阵元处理函数
            if tid == sid
                # 计算三角形相关的(3*3)个矩阵元的结果
                Zts  =  CFIEOnTris(geot)
                # 写入数据
                for ni in 1:3, mi in 1:3
                    # 基函数id
                    m = geot.inBfsID[mi]
                    n = geos.inBfsID[ni]
                    # 判断边是不是基函数（边缘不构建半基函数时适用）
                    (m == 0 || n == 0) && continue
                    # 往矩阵填充结果
                    # 判断是不是在源盒子、场盒子包含的区间内
                    ((msInInterval[mi] && nsInInterval[ni])) && begin
                        ZnearChunk[m, n] += Zts[mi, ni]
                    end
                end
            elseif Rts < Rsglr
                # 需要进行近奇异性处理的场源三角形
                Zts, _    =   CFIEOnNearTris(geot, geos)
                # 写入数据
                for ni in 1:3, mi in 1:3
                    # 基函数id
                    m = geot.inBfsID[mi]
                    n = geos.inBfsID[ni]

                    # 判断边是不是基函数（边缘不算）
                    (m == 0 || n == 0) && continue
                    ## 分布式避免数据通信不再利用对称性填充
                    # 避免线程锁的矩阵元循环方式下产生的条件
                    # (tid > sid) && (m in cubeBFinterval) && continue
                    # 判断是不是在源盒子、场盒子包含的区间内
                    ((msInInterval[mi] & nsInInterval[ni])) && begin
                        ZnearChunk[m, n] += Zts[mi, ni]
                    end
                end
            else
                # 正常高斯求积
                # 计算三角形相关的(3*3)个矩阵元的结果
                Zts, _    =   CFIEOnTris(geot, geos)
                # 写入数据
                for ni in 1:3, mi in 1:3
                    # 基函数id
                    m = geot.inBfsID[mi]
                    n = geos.inBfsID[ni]

                    # 判断边是不是基函数（边缘不算）
                    (m == 0 || n == 0) && continue
                    ## 分布式避免数据通信不再利用对称性填充
                    # 避免线程锁的矩阵元循环方式下产生的条件
                    # (tid > sid) && (m in cubeBFinterval) && continue

                    # 判断是不是在源盒子、场盒子包含的区间内
                    # @show ((msInInterval[mi] & nsInInterval[ni]))
                    ((msInInterval[mi] & nsInInterval[ni])) && begin
                        ZnearChunk[m, n] += Zts[mi, ni]
                    end
                end
                
            end # if
        end #jGeo
    end #iGeo

    return nothing
end


# """
# 采用 RWG 基函数计算指定层内 CFIE 面积分（SIE）阻抗矩阵近场元并将结果放在 ZnearChunk 中
# """
# function calZnearChunksCFIEonLevel!(cubes, geosInfo::Vector{GeoangleInfo{IT, FT}},
#     ZnearChunk, bfT::Type{BFT}) where {IT<:Integer, FT<:Real, BFT<:LinearBasisFunction}
    
#     # ZnearCksLocal = localpart(ZnearChunks)
#     for iCube in localindices(ZnearChunks)[1]
#         calZnearChunkCFIEonCube!(iCube, cubes, geosInfo, ZnearChunks[iCube], bfT)
#     end # for

# end # function

"""
采用 RWG 基函数计算指定层内 CFIE 阻抗矩阵近场元并将结果放在 ZnearChunk 中 (分布式)
"""
function calZnearChunksCFIE!(cubes, geosInfo::AbstractVector{GT},
    ZnearChunks::MPIVector, bfT::Type{BFT}) where {GT<:VSCellType, BFT<:BasisFunctionType}
    
    # 本进程索引
    idcs    =   ZnearChunks.indices[1]
    # 本地数据
    ZnearChunkslc   =   getGhostMPIVecs(ZnearChunks)
    # 进度条
    cond = true

    pmeter = Progress(length(idcs); desc = "Z on rank $(ZnearChunks.myrank)...", dt = 1, barglyphs=BarGlyphs("[=> ]"), color = :blue, enabled = cond)

    # 计算
    for i in idcs
        calZnearChunkCFIEonCube!(i, cubes, geosInfo, ZnearChunkslc[i], bfT)
        next!(pmeter)
    end
    MPI.Barrier(ZnearChunks.comm)
    nothing

end # function