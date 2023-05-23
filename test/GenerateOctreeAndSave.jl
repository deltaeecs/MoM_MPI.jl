
function genetare_octree_and_save(geosInfo, bfsInfo; tempdir = "temp")
    nLevels, octree     =   getOctreeAndReOrderBFs!(geosInfo, bfsInfo);
    #* 保存用于 MPI 进程调用
    leafLevel = octree.levels[nLevels];
    # geosInfo
    saveGeosInfoChunks(geosInfo, leafLevel.cubes, "geosInfo", ParallelParams.nprocs; dir = joinpath(tempdir, "GeosInfo"))
    @test true
    # octree
    saveOctree(octree; dir = joinpath(tempdir, "OctreeInfo"))
    @test true

    nothing
end
