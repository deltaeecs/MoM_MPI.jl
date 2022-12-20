
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