using MoM_Kernels:OctreeInfo


function loadOctree(fn; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), np = MPI.Comm_size(comm))
    data = load(fn, "data")
    @unpack nLevels, leafCubeEdgel, 
        bigCubeLowerCoor, levelsname = data
    levels = Dict{Int, LevelInfoMPI}()
    for i in 1:nLevels
        levels[i] = loadMPILevel(levelsname*"_$i.jld2"; comm=comm, rank=rank, np=np)
    end
    return OctreeInfo(nLevels, leafCubeEdgel, bigCubeLowerCoor, levels)
end