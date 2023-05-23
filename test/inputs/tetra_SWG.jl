# ! 所有 MoM 参数请在本文件设置, MPI参数在 "mpi_initial.jl" 中设置

# 导入包和基本参数
include(joinpath(@__DIR__, "../mom_input.jl"))

# 设置精度，是否运行时出图等，推荐不出图以防在没有图形界面的服务器报错
setPrecision!(Float64)
SimulationParams.SHOWIMAGE = true

# 网格文件和基函数类型
filename = "Tetra.nas"
sbfT  =  :nothing
vbfT  =  :SWG

## 设置输入频率从而修改内部参数
inputParameters(;frequency = 20e8, ieT = :EFIE)

# ! set the number of nprocs uesed in MPI
set_nprocs!(np = 4)
