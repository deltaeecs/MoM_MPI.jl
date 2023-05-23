using MoM_Basics, MoM_Kernels

# 网格文件夹
meshesdir = joinpath(pathof(MoM_Kernels), "../../meshfiles")
# project path
proj_path = joinpath(@__DIR__, "..")
# 临时文件夹
tempdir   = joinpath(@__DIR__, "../temp")
!ispath(tempdir) && mkdir(tempdir)
