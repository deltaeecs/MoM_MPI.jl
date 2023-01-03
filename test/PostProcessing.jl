function test_postprocessing(ICoeff, geosInfo)

    ## 观测角度
    θs_obs  =   LinRange{Precision.FT}(  -π/2, π/2,  1441)
    ϕs_obs  =   LinRange{Precision.FT}(  0, π/2,  3 )

    # RCS
    radarCrossSection(θs_obs, ϕs_obs, ICoeff, geosInfo) 

    @test true
    nothing
end