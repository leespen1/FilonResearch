using Test
using FilonResearch
using ControlledOperators

# Umbrella-level (integration) tests run in a lean environment that excludes the
# plotting stack (see test/Project.toml), so CI stays fast. Add tests here that
# span the lib packages or exercise the DrWatson glue in src/.
#
# If a test needs DrWatson path helpers (srcdir/projectdir), add DrWatson to
# test/Project.toml — it is light and pulls no plotting deps. Do NOT call
# `@quickactivate "FilonExperiments"` here: that re-activates the heavy root
# environment and defeats the purpose of this lean setup.

println("Starting tests")
ti = time()

@testset "FilonExperiments" begin
    @testset "library packages load" begin
        @test FilonResearch isa Module
        @test ControlledOperators isa Module
    end
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti / 60, digits = 3), " minutes")
