## 激活环境
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

## 安装包
pkgs = ["https://gitee.com/deltaeecs/IterativeSolvers.jl.git",
        "https://gitee.com/deltaeecs/MPIArray4MoMs.jl.git"]

map(pkgs) do pkg
    try
        Pkg.add(url = pkg)
    catch
        nothing
    end
end

## 初始化
Pkg.instantiate()
