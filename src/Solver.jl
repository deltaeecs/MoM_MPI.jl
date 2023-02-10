include("Solver/gmres.jl")

import MoM_Kernels:solve!, saveCurrent
using MoM_Kernels:LinearMapType, iterSolverSet, saveCurrent, convergencePlot
using IterativeSolvers:Identity

"""
保存电流系数
"""
function MoM_Kernels.saveCurrent(ICurrent::MPIVector; dir = "temp/results", str = "", root = 0, rank = ICurrent.myrank)
    ICurrentRoot = gather(ICurrent)
    !ispath(dir) && mkpath(dir)
    rank == root && jldsave(joinpath(dir, "ICurrent$str.jld2"); ICurrent = ICurrentRoot)
end

"""
矩阵方程
Ax=b
复合求解函数
输入值：
A::Matrix{T}, b::Vector{T}
solverT::Symbol  求解器类型
"""
function solve!(A::LinearMapType{T}, x::MPIVector, b::MPIVector; 
    solverT::Symbol = :gmres!, Pl = Identity(), Pr = Identity(), rtol = 1e-3, 
    maxiter = 1000, restart = 200, verbose = true, str = "", dir = "temp/results",
    comm = MPI.COMM_WORLD, rank = MPI.Comm_rank(comm), root  = 0) where{T<:Number}

    FT = real(T)
    # 迭代求解器
    solver      =   iterSolverSet(solverT)
    # 残差阈值
    resnorm0    =   FT(norm(ldiv!(deepcopy(b), Pl, b)))
    resnormtol  =   FT(rtol*resnorm0)

    # 迭代求解
    rank == root && println("Solving matrix function with iterate solver $solverT, with initial resnorm $resnorm0. ")
    # try 
    ICoeff, ch       =   solver(x, A, b; restart = restart, abstol = resnormtol, Pl = Pl, Pr = Pr,  log = true, verbose=verbose, maxiter = maxiter)
    saveCurrent(ICoeff; str = str, dir = dir)

    # 相对残差结果
    relresnorm  =   ch.data[:resnorm] / resnorm0

    # 命令行绘图
    SimulationParams.SHOWIMAGE  &&  convergencePlot(relresnorm)

    # 将相对残差写入文件
    !ispath(SimulationParams.resultDir) && mkpath(SimulationParams.resultDir)
    open(joinpath(SimulationParams.resultDir, "$(solverT)_ch$str.txt"), "w") do io
        for resi in relresnorm
            write(io, "$resi\n" )
        end
    end

    return ICoeff, ch

end