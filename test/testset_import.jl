

@testset "Import Step" verbose = true begin

    # fn = joinpath(pkgdir(@__MODULE__), "assets", "plate_with_ribs_2.step")
    fn = "../assets/plate_with_ribs.step"
    meshdata = FD.load_step_mesh_as_surface(fn)
    @test FD.is_watertight(meshdata.mesh.faces)

end

@testset "STL import" verbose = true begin
    fn = "test_assets/cube.stl"
    meshdata = FD.meshdata(fn)
    @test hasproperty(meshdata, :tlas)
end