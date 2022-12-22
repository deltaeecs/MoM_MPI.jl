
function saveVec2Chunks(y::AbstractVector, name::AbstractString, nchunk::Int; dir = "")

	indices = sizeChunks2idxs(length(y), nchunk)

	saveVec2Chunks(y, name, indices; dir = dir)

	nothing

end


function saveVec2Chunks(y::AbstractVector, name::AbstractString, indices; dir = "")

	!ispath(dir) && mkpath(dir)

	@floop for (i, indice) in enumerate(indices)
		jldsave(joinpath(dir, "$(name)_part_$i.jld2"), data = y[indice...], size = (length(y), ), indice = indice)
	end

	nothing

end

"""
	getGhostMPIVecs(y::MPIVector{T, I}) where {T, I}
   
	这里必须注明类型以稳定计算。
TBW
"""
function getGhostMPIVecs(y::MPIVector{T, I}) where {T, I}
	sparsevec(y.ghostindices[1], y.ghostdata)::SparseVector{T, Int}
end

Base.copyto!(y::MPIVector, x::MPIVector) = copyto!(y.data, x.data)