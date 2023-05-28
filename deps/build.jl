## 激活环境
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

## 安装包
pkgs = ["https://gitee.com/deltaeecs/IterativeSolvers.jl.git",]

map(pkgs) do pkg
    try
        Pkg.add(url = pkg)
    catch
        nothing
    end
end

## 初始化
Pkg.instantiate()
