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