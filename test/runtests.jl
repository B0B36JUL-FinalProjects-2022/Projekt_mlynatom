using Projekt_mlynatom
using Test

@testset "Projekt_mlynatom.jl" begin
    x = 5
    y = 6
    res = x + y

    @test my_function(x,y) == res+1
end
