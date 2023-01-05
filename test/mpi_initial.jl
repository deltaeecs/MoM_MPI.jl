# * 导入计算包
using MPI, MPIArray4MoMs, MoM_MPI
using TimerOutputs, JLD2
using LinearAlgebra
using .Threads, MKL, MKLSparse


# * 初始化MPI包
!MPI.Initialized() && MPI.Init()

comm = MPI.COMM_WORLD
comm_rank = MPI.Comm_rank(comm)
np   = MPI.Comm_size(comm)

root = 0

# ! 所有参数请在本文件设置
# 设置精度，是否运行时出图等，推荐不出图以防在没有图形界面的服务器报错
setPrecision!(Float64)
SimulationParams.SHOWIMAGE = false

@info "Rank $comm_rank successfully started the program!"