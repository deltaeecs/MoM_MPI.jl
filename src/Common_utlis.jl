"""
	getGhostMPIVecs(y::MPIVector{T, I, DT, IG}) where {T, I, DT, IG}
	
	这里必须注明类型以稳定计算。
TBW
"""
function getGhostMPIVecs(y::MPIVector{T, I, DT, IG}) where {T, I, DT, IG}
	sparsevec(y.ghostindices[1], y.ghostdata)::SparseVector{T, Int}
end


function sortedVecInUnitRange(y::AbstractVector, ur::UnitRange)
	!issorted(y) && sort!(y)
	if (first(y) >= first(ur)) && (last(y) <= last(ur))
		return true
	else
		return false
	end
end


"""
	创建宏用于 MPI 测速
	mpitime(nloop, message, ex)

	nloop 测试循环数
	message 展示信息
	ex 测试表达式

"""
macro mpitime(nloop, message, ex, to = TimerOutput())
    esc(
        quote
			local comm = MPI.COMM_WORLD
            for _ in 1:$nloop
				@timeit to $message $ex
				MPI.Barrier(comm)
			end
			to
        end
    )
end


"""
    gather_Independent_Vectors(vecs; comm = MPI.COMM, rank = rank)

	获取 MPI 进程里的分段向量

TBW
"""
function gather_Independent_Vectors(vecs::AbstractVector; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm))

	# 各个 rank 的 vec 的长度
	rvecls	=	MPI.Allgather(length(vecs), comm)
	# 所有 vec 的长度
	vecls	=	zeros(eltype(first(vecs)), sum(rvecls))
	MPI.Allgatherv!(map(length, vecs), VBuffer(vecls, rvecls), comm)
	MPI.Barrier(comm)
	# 每个 vec 的区间
	intervals 	= 	zeros(Int, length(vecls)+1)
	cumsum!(view(intervals, 2:(length(vecls)+1)), vecls)

	# 收集 vecs
	ls		=	MPI.Allgather(sum(length, vecs), comm)
	data	=	zeros(eltype(first(vecs)), sum(ls))
	MPI.Allgatherv!(reduce(vcat, vecs), VBuffer(data, ls), comm)
	
	# 分割
	indep_data 	=	map(i -> data[(intervals[i]+1):intervals[i+1]], eachindex(vecls))

	return indep_data

end

"""
    gather_Independent_Vectors(vecs; comm = MPI.COMM, rank = rank)

	获取 MPI 进程里的分段向量

TBW
"""
function gather_Independent_Vectors(vecs::Vector{T}; comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm)) where{T<:UnitRange}
	# 每个rank上的大小
	ls  	=   MPI.Allgather(length(vecs), comm)
	# 收集 vecs
	data 	= 	Vector{UnitRange{eltype(first(vecs))}}(undef, sum(ls))
	MPI.Allgatherv!(vecs, VBuffer(data, ls), comm)
	
	return data

end

