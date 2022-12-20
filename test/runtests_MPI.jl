# 包的载入与相关参数设置
include("mom_input.jl")
include("mpi_initial.jl")

# 几何信息与八叉树信息
geosInfo = getGeosInfo("temp/GeosInfo/geosInfo_part_$(comm_rank + 1).jld2")
octree = loadOctree("temp/OctreeInfo/Octree.jld2", np = ParallelParams.nprocs)
MPI.Barrier(comm)
@info "Data loaded!"

# 叶层
nLevels    =   octree.nLevels
leafLevel  =   octree.levels[nLevels];
leafCubes  =   leafLevel.cubes;
# 创建系数向量 MPI 数组
ICoeff = MPIvecOnLevel(leafLevel);
fill!(ICoeff, comm_rank)

sync!(ICoeff)

@info norm(ICoeff)

for i in 1:ParallelParams.nprocs
    i == (comm_rank + 1) && begin
        @info comm_rank "ghost" ICoeff.grank2ghostindices
        @info comm_rank "remote" ICoeff.rrank2localindices
        for (rk, idc) in ICoeff.grank2ghostindices
            @info comm_rank, rk, sum(ICoeff.ghostdata[idc...]) / length(ICoeff.ghostdata[idc...])
        end
        for (rk, idc) in ICoeff.rrank2localindices
            @info comm_rank, rk, sum(ICoeff.data[idc...]) / length(ICoeff.data[idc...])
        end
    end
    MPI.Barrier(comm)
end

# 计算矩阵近场元
nbf = length(ICoeff)
ZnearChunksMPI = initialZnearChunksMPI(leafLevel; nbf = nbf)
# calZnearChunks!(leafCubes, geosInfo, ZnearChunksMPI)

