Base.@kwdef mutable struct ParallelParamsType
    nprocs::Int
end

ParallelParams = ParallelParamsType(0)

function set_nprocs!(;nprocs=1, np=nprocs)
    ParallelParams.nprocs = np
    nothing
end