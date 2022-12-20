using MoM_Basics, MoM_Kernels, MoM_MPI
using Test

include("mom_input.jl")
include("GenerateOctreeAndSave.jl")

@testset "MoM_MPI.jl" begin
    # Write your tests here.
    meshData, εᵣs   =  getMeshData(filename; meshUnit=:mm);
    @test true

    @testset "Host" begin

        ngeo, nbf, geosInfo, bfsInfo =  getBFsFromMeshData(meshData, vbfT = vbfT)
        @test true

        setGeosPermittivity!(geosInfo, 2 + 0.001im)
        @test true

        genetare_octree_and_save(geosInfo, bfsInfo)

        # ! 主节点工作到此为止，其他工作交给 MPI 进程
        # 运行$(ParallelParams.nprocs)
        mpiexec(cmd -> run(`$cmd -n $(ParallelParams.nprocs) $(Base.julia_cmd()) -t 1 --project=. runtests_MPI.jl`))
    end

end
