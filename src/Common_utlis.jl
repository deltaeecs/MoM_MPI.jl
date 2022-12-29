
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

"""
struct PartitionedVector{T} <: AbstractVector{T}
    length::Int
    data::OffsetVector{T, Vector{T}}
    indices::UnitRange{Int}
    ghostdata::SparseVector{T, Int}
    ghostindices::Vector{T, Int}
end

用于保存向量块，同时在块内保存一些其他块的数据。

"""
struct PartitionedVector{T} <: AbstractVector{T}
    size::Int # 原始 Vector 的大小
    data::OffsetVector{T, Vector{T}} # 本地保存的数据
    indices::UnitRange{Int} # 本地保存数据的索引区间
    ghostdata::SparseVector{T, Int} # 用到的其它数据
    ghostindices::Vector{Int}    # 用到的其它数据的索引区间
end

Base.size(A::T) where{T<:PartitionedVector}  = (A.size, )
Base.eltype(::PartitionedVector{T}) where{T} = T
Base.length(A::T) where{T<:PartitionedVector} = A.size


"""
    Base.getindex(A::PartitionedVector, i::I) where {I<:Integer}

    重载 getindex.
TBW
"""
function Base.getindex(A::PartitionedVector, i::I) where {I<:Integer}
    if i in A.indices
        return getindex(A.data, i)
    elseif !isempty(searchsorted(A.ghostindices, i))
        return getindex(A.ghostdata, i)
    else
        @warn "$i is out of indices of Array"
		return nothing
    end
end