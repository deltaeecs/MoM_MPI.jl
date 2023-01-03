module MoM_MPI

using MoM_Basics, MoM_Kernels
using MPI, MPIArray4MoMs
using OffsetArrays, SparseArrays
using LinearAlgebra
using .Threads, ThreadsX, FLoops, FoldsThreads
using UnPack, JLD2, ProgressMeter, Printf
using IterativeSolvers

export  ParallelParams, set_nprocs!,
        getMeshDataSaveGeosInterval, set_geosInterval!,
        MPIvecOnLevel,
        initialZnearChunksMPI, calZnearChunks!,
        getGeoIDsInCubeChunk, getNeighborCubeIDs, saveGeosInfoChunks, getGeosInfo,
        LevelInfoMPI, getFarNeighborCubeIDs, saveCubes, saveLevel, 
        loadCubes, loadMPILevel!, loadMPILevel,
        saveOctree, loadOctree,
        getExcitationVector!

# 参数设置相关
include("ParallelParams.jl")

# 通用接口
include("Common_Utlis.jl")

# 涉及几何信息 IO 等
include("GeosInfo.jl")

# 涉及 MPI 数据的八叉树、层的 IO, MLFMA计算 等
include("MLFMA.jl")

# 涉及阻抗矩阵
include("Znear.jl")

# 激励向量
include("ExcitedVectors.jl")

# 涉及 求解器
include("Solver.jl")

end
