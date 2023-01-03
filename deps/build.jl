using Pkg

Pkg.activate("..")

# Pkg.add(url = "https://gitee.com/deltaeecs/mom_basics.git")
Pkg.add(path="../../MoM_Basics")
Pkg.add(path="../../MoM_Kernels")
Pkg.add(path="../../MPIArray4MoMs")
Pkg.add(path="../../IterativeSolvers/")

using MPIPreferences

MPIPreferences.use_jll_binary()