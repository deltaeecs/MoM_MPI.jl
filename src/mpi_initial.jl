# * 导入计算包
using MPI, MoM_MPI
using TimerOutputs, JLD2
using LinearAlgebra
using .Threads


# * 初始化MPI包
!MPI.Initialized() && MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
np   = MPI.Comm_size(comm)

root = 0

@info "Rank $rank successfully started the program!"