![MoM](docs/src/assets/logo.png)
# MoM_MPI

![star](https://img.shields.io/github/stars/deltaeecs/MoM_MPI.jl?style=social)

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://deltaeecs.github.io/MoM_MPI.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://deltaeecs.github.io/MoM_MPI.jl/dev/)
[![Build Status](https://github.com/deltaeecs/MoM_MPI.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/deltaeecs/MoM_MPI.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/deltaeecs/MoM_MPI.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/deltaeecs/MoM_MPI.jl)

![Size](https://img.shields.io/github/repo-size/deltaeecs/MoM_MPI.jl
)
![Downloads](https://img.shields.io/github/downloads/deltaeecs/MoM_MPI.jl/total)
![License](https://img.shields.io/github/license/deltaeecs/MoM_MPI.jl)

## 介绍

提供 CEM\_MoMs ([![github](https://img.shields.io/badge/github-blue.svg)](https://github.com/deltaeecs/CEM_MoMs.jl), [![gitee](https://img.shields.io/badge/gitee-red.svg)](https://gitee.com/deltaeecs/CEM_MoMs.jl)) 包的 MPI 并行扩展算法，包括各类基函数的矩阵元计算、多层快速多极子（MLFMA）各种类型和函数、求解器等功能。 CEM_MoMs 本身被拆分为几个独立的包以方便开发时快速编译，同时避免在无图形化界面使用时调入绘图相关包而导致报错。

## 安装与测试

### 安装

可直接在 Julia REPL 的包管理模式中安装：

```julia
julia> Pkg.add("MoM_MPI")
```

或

```julia
pkg> add MoM_MPI
```

### 测试

同样可直接在 Julia REPL 的包管理模式中测试包：

```julia
julia> Pkg.test("MoM_MPI")
```

或

```julia
pkg> add MoM_MPI
```