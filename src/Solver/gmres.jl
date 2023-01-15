### 为 MPIArray 扩展 gemres solver.
using IterativeSolvers: Adivtype, ArnoldiDecomp, Residual, init!, init_residual!, expand!,
						reserve!, setconv, converged, shrink!, gmres_iterable!, nextiter!, update_solution!,
						GMRESIterable, zerox, OrthogonalizationMethod, ModifiedGramSchmidt

IterativeSolvers.zerox(A::AbstractArray, b::MPIVector) = deepcopy(b)

function get_Orthonormal_basis_vectorsz(b, restart::Int)

	T = eltype(b)
	restartp1 = restart + 1
	indices 		= 	(b.indices..., 1:restartp1)
	ghostindices 	= 	(b.ghostindices..., 1:restartp1)
	ghostdata 		= 	zeros(T, size(b.ghostdata, 1), restartp1)
	data 			=	view(ghostdata, b.data.indices..., 1:restartp1)
	dataOffset 		= 	OffsetArray(data, b.dataOffset.offsets..., 0)
	Vsize 			=	(b.size..., restartp1)

	rank2indices 	=	Dict{Int, Tuple{typeof(first(values(b.rank2indices))[1]), UnitRange{Int}}}()

	for (k, v) in b.rank2indices
		rank2indices[k] =	(v..., 1:restartp1)
	end

	grank2ghostindices	=	Dict{Int, Tuple{typeof(first(values(b.grank2ghostindices))[1]), UnitRange{Int}}}()
	for (k, v) in b.grank2ghostindices
		grank2ghostindices[k] = (v..., 1:restartp1)
	end

	rrank2localindices	=	Dict{Int, Tuple{typeof(first(values(b.rrank2localindices))[1]), UnitRange{Int}}}()
	for (k, v) in b.rrank2localindices
		rrank2localindices[k] = (v..., 1:restartp1)
	end

	return MPIMatrix{T, typeof(indices), typeof(data), typeof(ghostindices)}(	data, indices, dataOffset, b.comm, b.myrank, Vsize, 
											rank2indices, ghostdata, ghostindices, grank2ghostindices, rrank2localindices)

end


"""
	IterativeSolvers.gmres_iterable!(x::MPIVector, A, b::MPIVector;
							Pl = Identity(),
							Pr = Identity(),
							abstol::Real = zero(real(eltype(b))),
							reltol::Real = sqrt(eps(real(eltype(b)))),
							restart::Int = min(20, size(A, 2)),
							maxiter::Int = size(A, 2),
							initially_zero::Bool = false,
							orth_meth::OrthogonalizationMethod = ModifiedGramSchmidt())

	重载 gmres_iterable!

"""
function IterativeSolvers.gmres_iterable!(x::MPIVector, A, b::MPIVector;
							Pl = Identity(),
							Pr = Identity(),
							abstol::Real = zero(real(eltype(b))),
							reltol::Real = sqrt(eps(real(eltype(b)))),
							restart::Int = min(20, size(A, 2)),
							maxiter::Int = size(A, 2),
							initially_zero::Bool = false,
							orth_meth::OrthogonalizationMethod = ModifiedGramSchmidt())
	T = Adivtype(A, b)

	V = get_Orthonormal_basis_vectorsz(b, restart)

	# Approximate solution
	arnoldi = ArnoldiDecomp(A, restart, T, V)
	residual = Residual(restart, T)
	mv_products = initially_zero ? 1 : 0

	# Workspace vector to reduce the # allocs.
	Ax = similar(x)
	residual.current = init!(arnoldi, x, b, Pl, Ax, initially_zero = initially_zero)
	init_residual!(residual, residual.current)

	tolerance = max(reltol * residual.current, abstol)

	GMRESIterable(Pl, Pr, x, b, Ax,
					arnoldi, residual,
					mv_products, restart, 1, maxiter, tolerance, residual.current,
					orth_meth)
end


function IterativeSolvers.init!(arnoldi::ArnoldiDecomp{T}, x, b::MPIArray, Pl, Ax; initially_zero::Bool = false) where {T}
    # Initialize the Krylov subspace with the initial residual vector
    # This basically does V[1] = Pl \ (b - A * x) and then normalize

    first_col = view(arnoldi.V, :, 1)

    copyto!(first_col, b)

    # Potentially save one MV product
    if !initially_zero
        mul!(Ax, arnoldi.A, x)
        # first_col .-= Ax
		axpy!(-1, Ax, first_col)

    end

    ldiv!(Pl, first_col)

    # Normalize
    β = norm(first_col)
    # first_col .*= inv(β)
    rmul!(first_col, inv(β))
    
    β
end

"""
	sync!(A::SubMPIVector)

	Synchronize data in `A` between MPI ranks.
    为求解器设计，不做它用。

TBW
"""
function syncUnknownVectorView!(V::SubMPIVector; comm = V.parent.comm, rank = V.parent.myrank, np = MPI.Comm_size(comm))

	Vp 	= V.parent
	k 	= V.indices[2]

	# begin sync
	req_all = MPI.Request[]
	begin
		for (ghostrank, indices) in Vp.grank2ghostindices
			req = MPI.Irecv!(view(Vp.ghostdata, indices[1], k), ghostrank, ghostrank*np + rank, Vp.comm)
			push!(req_all, req)
		end
		for (remoterank, indices) in Vp.rrank2localindices
			req = MPI.Isend(Vp.data[indices[1], k], remoterank, rank*np + remoterank, Vp.comm)
			push!(req_all, req)
		end
	end
	MPI.Waitall(MPI.RequestSet(req_all), MPI.Status)

	MPI.Barrier(Vp.comm)

	nothing

end

import IterativeSolvers:gmres!, update_solution!

function gmres!(x::MPIVector, A, b::MPIVector;
				Pl = Identity(),
				Pr = Identity(),
				abstol::Real = zero(real(eltype(b))),
				reltol::Real = sqrt(eps(real(eltype(b)))),
				restart::Int = min(20, size(A, 2)),
				maxiter::Int = size(A, 2),
				log::Bool = false,
				initially_zero::Bool = false,
				verbose::Bool = false,
				orth_meth::OrthogonalizationMethod = ModifiedGramSchmidt(),
				root::Int = 0, comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm))
	history = ConvergenceHistory(partial = !log, restart = restart)
	history[:abstol] = abstol
	history[:reltol] = reltol
	log && reserve!(history, :resnorm, maxiter)

	iterable = gmres_iterable!(x, A, b; Pl = Pl, Pr = Pr,
					abstol = abstol, reltol = reltol, maxiter = maxiter,
					restart = restart, initially_zero = initially_zero,
					orth_meth = orth_meth)

	if (rank == root) && verbose
		@printf("================= gmres ==================\n%4s\t%4s\t%9s\t%9s\n","rest", "iter", "resnorm", "relresn")
	end

	resnorm0 = iterable.residual.current
	for (iteration, residual) = enumerate(iterable)
		if log
			nextiter!(history)
			history.mvps = iterable.mv_products
			push!(history, :resnorm, residual)
		end
		if (rank == root) && verbose
			@printf("%4d\t%4d\t%1.4e\t%1.4e\n", 1 + div(iteration - 1, restart), 1 + mod(iteration - 1, restart), residual, residual/resnorm0)
		end
	end

	(rank == root) && verbose && println()
	setconv(history, converged(iterable))
	log && shrink!(history)

	log ? (x, history) : x
end


function update_solution!(x::MPIVector, y, arnoldi::ArnoldiDecomp{T}, Pr::Identity, k::Int, Ax) where {T}
    # Update x ← x + V * y
    mul!(x, view(arnoldi.V, :, 1 : k - 1), y, one(T), one(T))
end


function update_solution!(x::MPIVector, y, arnoldi::ArnoldiDecomp{T}, Pr::Identity, k::Int, Ax) where {T}
    # Computing x ← x + Pr \ (V * y) and use Ax as a work space
    mul!(Ax, view(arnoldi.V, :, 1 : k - 1), y)
    ldiv!(Pr, Ax)
    axpy!(1, Ax, x)
end