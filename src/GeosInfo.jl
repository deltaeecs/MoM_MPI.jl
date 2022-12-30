"""
    getGeoIDsInCubeChunk(cubes, ckunkIndice)

获取 ckunkIndice 内的所有 cube 的 geo ID ， 返回为 Tuple 形式以适应数组索引相关API

"""
function getGeoIDsInCubeChunk(cubes, chunkIndice::Tuple)

    geoIDs = reduce(vcat, cubes[i].geoIDs for i in chunkIndice[1])

    return (unique!(sort!(geoIDs)), )

end

"""
    getGeoIDsInCubeChunk(cubes, ckunkIndice)

获取 ckunkIndice 内的所有 cube 的 geo ID ， 返回为 Tuple 形式以适应数组索引相关API

"""
function getGeoIDsInCubeChunk(cubes, chunkIndice::UnitRange)

    geoIDs = reduce(vcat, cubes[i].geoIDs for i in chunkIndice)

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
function saveGeosInfoChunks(geos::AbstractVector, cubes, name::AbstractString, nchunk::Int; dir = "", cubes_ChunksIndices =   sizeChunks2idxs(length(cubes), nchunk))
    # 拿到各块的包含邻盒子的id
    cubesNeighbors_ChunksIndices    =   ThreadsX.mapi(chunkIndice -> getNeighborCubeIDs(cubes, chunkIndice), cubes_ChunksIndices)
    # 拿到包含邻盒子内的该块的所有几何信息 id
    geoInfo_chunks_indices  =   ThreadsX.mapi(chunkIndice -> getGeoIDsInCubeChunk(cubes, chunkIndice), cubesNeighbors_ChunksIndices)
    # 保存
	saveVec2Chunks(geos, name, geoInfo_chunks_indices; dir = dir)

	nothing

end

function getMeshDataSaveGeosInterval(filename; meshUnit=:mm, dir = "temp/GeosInfo");
    meshData, εᵣs   =  getMeshData(filename; meshUnit=meshUnit);
    saveGeoInterval(meshData; dir = dir)
    return meshData, εᵣs   
end

function saveGeoInterval(meshData; dir = "temp/GeosInfo")
    !ispath(dir) && mkpath(dir)
    data = (tri = 1:meshData.trinum, tetra = (meshData.trinum + 1):(meshData.trinum + meshData.tetranum),
            hexa = (meshData.trinum + meshData.tetranum + 1):meshData.geonum,)
    jldsave(joinpath(dir, "geoInterval.jld2"), data = data)
    nothing
end

function loadGeoInterval(fn)
    load(fn, "data")
end


function getGeosInfo(fn)

    @unpack data, size, indice = load(fn)

    geosInfolw = sparsevec(indice[1], data, size[1])

    return geosInfolw

end

using MoM_Kernels:getGeosInterval

"""
获取几何信息数组的区间，针对 AbstractVector{TriangleInfo} 派发
"""
function MoM_Kernels.getGeosInterval(::AbstractVector{T})::UnitRange{Int} where {T<:TriangleInfo}
    return GeosInterval.tri
end

"""
获取几何信息数组的区间，针对 AbstractVector{TetrahedraInfo} 派发
"""
function MoM_Kernels.getGeosInterval(::AbstractVector{T})::UnitRange{Int} where {T<:TetrahedraInfo}
    return GeosInterval.tetra
end

"""
获取几何信息数组的区间，针对 AbstractVector{HexahedraInfo} 派发
"""
function MoM_Kernels.getGeosInterval(::AbstractVector{T})::UnitRange{Int} where {T<:HexahedraInfo}
    return GeosInterval.hexa
end


