"""
    getGeoIDsInCubeChunk(cubes, ckunkIndice)

获取 ckunkIndice 内的所有 cube 的 geo ID ， 返回为 Tuple 形式以适应数组索引相关API

"""
function getGeoIDsInCubeChunk(cubes, chunkIndice::Tuple)

    geoIDs = reduce(vcat, cubes[i].geoIDs for i in chunkIndice[1])

    return (unique!(sort!(geoIDs)), )

end

"""
    getNeighborCubeIDs(cubes, chunkIndice)

    获取 ckunkIndice 内的所有 cube 的 邻盒子ID， 返回为 Tuple 形式以适应数组索引相关API

TBW
"""
function getNeighborCubeIDs(cubes, chunkIndice::Tuple)

    neighborCubeIDs = reduce(vcat, cubes[i].neighbors for i in chunkIndice[1])

    return (unique!(sort!(neighborCubeIDs)), )

end

function getNeighborCubeIDs(cubes, chunkIndice::AbstractVector)

    neighborCubeIDs = reduce(vcat, cubes[i].neighbors for i in chunkIndice)

    return (unique!(sort!(neighborCubeIDs)), )

end


"""
    saveGeosInfoChunks(geos::AbstractVector, cubes, name::AbstractString, nchunk::Int; dir = "")
    
    将几何信息保存在
TBW
"""
function saveGeosInfoChunks(geos::AbstractVector, cubes, name::AbstractString, nchunk::Int; dir = "")
    # 对盒子按 盒子数 和块数分块
	cubes_ChunksIndices =   sizeChunks2idxs(length(cubes), nchunk)
    # 拿到各块的包含邻盒子的id
    cubesNeighbors_ChunksIndices    =   ThreadsX.mapi(chunkIndice -> getNeighborCubeIDs(cubes, chunkIndice), cubes_ChunksIndices)
    # 拿到包含邻盒子内的该块的所有几何信息 id
    geoInfo_chunks_indices  =   ThreadsX.mapi(chunkIndice -> getGeoIDsInCubeChunk(cubes, chunkIndice), cubesNeighbors_ChunksIndices)
    # 保存
	saveVec2Chunks(geos, name, geoInfo_chunks_indices; dir = dir)

	nothing

end

function getGeosInfo(fn)

    @unpack data, size, indice = load(fn)

    geosInfolw = sparsevec(indice[1], data, size[1])

    return geosInfolw

end