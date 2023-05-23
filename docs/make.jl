using MoM_Basics, MoM_Kernels, MoM_MPI
using Documenter

DocMeta.setdocmeta!(MoM_MPI, :DocTestSetup, :(using MoM_MPI); recursive=true)

makedocs(;
    modules=[MoM_Basics, MoM_Kernels, MoM_MPI],
    authors="deltaeecs <1225385871@qq.com> and contributors",
    repo="https://github.com/deltaeecs/MoM_MPI.jl/blob/{commit}{path}#{line}",
    sitename="MoM_Kernels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://deltaeecs.github.io/MoM_MPI.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/deltaeecs/MoM_MPI.jl",
)