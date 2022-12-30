## 本文件的函数用于构建迭代算子
include("AggregateOnLevel.jl")

# MPI 迭代计算相关函数
include("IterateOnOctree.jl")


"""
实现矩阵向量乘积，并封装为线性算子
"""
function MoM_Kernels.MLMFAIterator(Znear, octree::OctreeInfo{FT, LT}, 
    geosInfo::AbstractVector) where {FT<:Real, LT<:LevelInfoMPI}
    
    # 叶层ID
    nLevels     =   octree.nLevels
    # 层信息
    levels      =   octree.levels
    # 叶层
    leafLevel   =   octree.levels[nLevels]
    # 基函数数量
    nbf         =   size(Znear, 2)
    
    # 预先计算叶层聚合项，内存占用为该层采样点数 nPoles × Nbf， 因此仅在内存充足时使用
    aggSBF, disaggSBF   =   getAggSBFOnLevel(leafLevel, geosInfo)
    
    # 给各层的聚合项、解聚项预分配内存
    memoryAllocationOnLevels!(nLevels, levels)
    # 给矩阵向量乘积预分配内存
    ZI      =   MPIvecOnLevel(leafLevel; T = Complex{FT})

    Zopt    =   MLMFAIterator{Complex{FT}, MPIVector}(octree, Znear, geosInfo, Int[], aggSBF, disaggSBF, ZI)

    return Zopt

end