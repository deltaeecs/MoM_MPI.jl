Base.@kwdef mutable struct ParallelParamsType
    nprocs::Int
end

ParallelParams = ParallelParamsType(0)

function set_nprocs!(;nprocs=1, np=nprocs)
    ParallelParams.nprocs = np
    nothing
end

Base.@kwdef mutable struct GeosIntervalType{T}
    tri::T
    tetra::T
    hexa::T
end

GeosInterval = GeosIntervalType{UnitRange{Int}}(0:0, 0:0, 0:0)

function set_geosInterval!(fn)
    data = loadGeoInterval(fn)
    GeosInterval.tri = data.tri
    GeosInterval.tetra = data.tetra
    GeosInterval.hexa = data.hexa
    nothing
end
