using MoM_Basics, MoM_Kernels

# 网格文件夹
meshesdir = normpath(joinpath(pathof(MoM_Kernels), "../../meshfiles"))
# project path
proj_path = normpath(joinpath(@__DIR__, ".."))
# 临时文件夹
tempdir   = normpath(joinpath(@__DIR__, "../temp"))
!ispath(tempdir) && mkdir(tempdir)
