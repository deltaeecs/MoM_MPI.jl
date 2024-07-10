using Test

@testset "MPI_Solving" begin
    # 几何信息与八叉树信息
    set_geosInterval!(joinpath(tempdir, "geoInterval.jld2"))
    geosInfo = getGeosInfo(joinpath(tempdir, "GeosInfo/geosInfo_part_$(rank + 1).jld2"))
    octree = loadOctree(joinpath(tempdir, "OctreeInfo/Octree.jld2"), np = ParallelParams.nprocs)
    MPI.Barrier(comm)
    @info "Data loaded!"
    @test true

    # 叶层
    nLevels    =   octree.nLevels
    leafLevel  =   octree.levels[nLevels];
    leafCubes  =   leafLevel.cubes;
    # 创建系数向量 MPI 数组
    ICoeff = MPIvecOnLevel(leafLevel);
    nbf = length(ICoeff)

    sleep(2)
    fill!(ICoeff, 1)

    @test norm(ICoeff) ≈ sqrt(nbf)

    fill!(ICoeff, rank)
    sync!(ICoeff)
    

    for i in 1:ParallelParams.nprocs
        i == (rank + 1) && begin
            for (rk, idc) in ICoeff.grank2ghostindices
                @test rk ≈ real(sum(ICoeff.ghostdata[idc...]) / length(ICoeff.ghostdata[idc...]))
            end
            for (rk, idc) in ICoeff.rrank2localindices
                @test rank ≈ real(sum(ICoeff.data[idc...]) / length(ICoeff.data[idc...]))
            end
        end
        MPI.Barrier(comm)
    end

    # 计算矩阵近场元
    ZnearChunksMPI = initialZnearChunksMPI(leafLevel; nbf = nbf)
    calZnearChunks!(leafCubes, geosInfo, ZnearChunksMPI)
    @test true

    # 构建矩阵向量乘积算子
    Zopt  =   MLFMAIterator(ZnearChunksMPI, octree, geosInfo; np = ParallelParams.nprocs);
    @test true

    fill!(ICoeff, 1)
    y = ZnearChunksMPI * ICoeff
    mul!(y, ZnearChunksMPI, ICoeff)
    @info "Znear*I" norm(y)
    @test true

    z = Zopt * ICoeff
    @test true
    mul!(z, Zopt, ICoeff)
    @info "Zopt*I" norm(z)
    @test true

    zc = deepcopy(z)
    zc.data .-= y.data
    MPI.Barrier(comm)
    @info "Zopt*I - Znear*I" norm(zc)
    @test true

    source  =   PlaneWave(π, 0, 0f0, 1f0)
    V = deepcopy(ICoeff)
    getExcitationVector!(V, geosInfo, source);
    sync!(V)
    @info "V" norm(V)
    @test true

    fill!(ICoeff, 0)
    ICoeff, ch   =   solve!(Zopt, ICoeff, V; dir = joinpath(tempdir, "results"));
    @test true
    
end

MPI.Finalize()