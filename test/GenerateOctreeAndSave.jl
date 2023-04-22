
function genetare_octree_and_save(geosInfo, bfsInfo)
    nLevels, octree     =   getOctreeAndReOrderBFs!(geosInfo, bfsInfo)
    #* 保存用于 MPI 进程调用
    leafLevel = octree.levels[nLevels];
    # geosInfo
    saveGeosInfoChunks(geosInfo, leafLevel.cubes, "geosInfo", ParallelParams.nprocs; dir = "temp/GeosInfo")
    @test true
    # octree
    saveOctree(octree; dir = "temp/OctreeInfo")
    @test true

    nothing
end
