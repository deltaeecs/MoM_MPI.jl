using Test, MPI
using MoM_Kernels

include("GenerateOctreeAndSave.jl")
include("PostProcessing.jl")

@testset "MoM_MPI.jl" begin

    @testset "Triangle, RWG" begin
        for IE in [:EFIE, :MFIE, :CFIE]
            @testset "S_$IE" begin
                include(joinpath(@__DIR__, "inputs", "tri_$IE.jl"))
                # meshes
                meshData, εᵣs   =  getMeshDataSaveGeosInterval(joinpath(meshesdir, filename); meshUnit=:mm, dir = tempdir);
                @test true
                # geos bfs
                ngeo, nbf, geosInfo, bfsInfo =  getBFsFromMeshData(meshData; sbfT = sbfT)
                @test true

                genetare_octree_and_save(geosInfo, bfsInfo; tempdir = tempdir)

                # ! 主节点工作到此为止，其他工作交给 MPI 进程
                # 运行$(ParallelParams.nprocs)
                mpiexec(cmd -> run(`$cmd -n $(ParallelParams.nprocs) $(Base.julia_cmd()) -t 1 
                                    --project=. $(joinpath(proj_path, "test/inputs", "tri_$(IE)_MPI.jl"))`))
        
                # 后处理
                ICoeff = loadCurrent(joinpath(tempdir, "results/ICurrent.jld2"))
                test_postprocessing(ICoeff, geosInfo)
                rm(SimulationParams.resultDir; force = true, recursive = true)
                rm(tempdir; force = true, recursive = true)
            end
        end
    end
    @testset "Tetrahedra, SWG + PWC" begin
        for vbfT in [:SWG, :PWC]
            include(joinpath(proj_path, "test/inputs", "tetra_$vbfT.jl"))
            # meshes
            meshData, εᵣs   =  getMeshDataSaveGeosInterval(joinpath(meshesdir, filename); meshUnit=:mm, dir = tempdir);
            @test true
            # geos bfs
            ngeo, nbf, geosInfo, bfsInfo =  getBFsFromMeshData(meshData; sbfT = sbfT, vbfT = vbfT)
            @test true

            setGeosPermittivity!(geosInfo, 2(1 - 0.001im))
            @test true

            genetare_octree_and_save(geosInfo, bfsInfo; tempdir = tempdir)

            # ! 主节点工作到此为止，其他工作交给 MPI 进程
            # 运行$(ParallelParams.nprocs)
            mpiexec(cmd -> run(`$cmd -n $(ParallelParams.nprocs) $(Base.julia_cmd()) -t 1 
                                --project=. $(joinpath(proj_path, "test/inputs", "tetra_$(vbfT)_MPI.jl"))`))

            # 后处理
            ICoeff = loadCurrent(joinpath(tempdir, "results/ICurrent.jld2"))
            test_postprocessing(ICoeff, geosInfo)
            rm(SimulationParams.resultDir; force = true, recursive = true)
            rm(tempdir; force = true, recursive = true)
        end
    end

    @testset "Hexahedra, RBF + PWC" begin
        for vbfT in [:RBF, :PWC]
            include(joinpath(proj_path, "test/inputs", "hexa_$vbfT.jl"))
            # meshes
            meshData, εᵣs   =  getMeshDataSaveGeosInterval(joinpath(meshesdir, filename); meshUnit=:mm, dir = tempdir);
            @test true
            # geos bfs
            ngeo, nbf, geosInfo, bfsInfo =  getBFsFromMeshData(meshData; sbfT = sbfT, vbfT = vbfT)
            @test true

            setGeosPermittivity!(geosInfo, 2(1 - 0.001im))
            @test true

            genetare_octree_and_save(geosInfo, bfsInfo; tempdir = tempdir)

            # ! 主节点工作到此为止，其他工作交给 MPI 进程
            # 运行$(ParallelParams.nprocs)
            mpiexec(cmd -> run(`$cmd -n $(ParallelParams.nprocs) $(Base.julia_cmd()) -t 1 
                                --project=. $(joinpath(proj_path, "test/inputs", "hexa_$(vbfT)_MPI.jl"))`))

            # 后处理
            ICoeff = loadCurrent(joinpath(tempdir, "results/ICurrent.jld2"))
            test_postprocessing(ICoeff, geosInfo)
            rm(SimulationParams.resultDir; force = true, recursive = true)
            rm(tempdir; force = true, recursive = true)
        end
    end

    @test true

end
