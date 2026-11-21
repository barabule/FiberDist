

@testset "Import Step" verbose = true begin

    # fn = joinpath(pkgdir(@__MODULE__), "assets", "plate_with_ribs_2.step")
    fn = "../assets/plate_with_ribs.step"
    meshdata = FD.load_step_mesh_as_surface(fn)
    @test FD.is_valid(meshdata.mesh)

end

@testset "STL import" verbose = true begin
    fn = "test_assets/cube.stl"
    meshdata = FD.meshdata(fn)
    @test FD.is_valid(meshdata.mesh)
    @test hasproperty(meshdata, :tlas)
end

@testset "Voxelization" verbose = true begin

    fn = "../assets/plate_with_ribs.step"
    meshdata = FD.load_step_mesh_as_surface(fn)
    voxeldata = FD.VoxelData(meshdata)
    @test size(voxeldata.grid)[1] == 100

    gates = [FD.ValveGate(position = FD.Point3f(0, 25, 0))]
    @test true

    gate_voxels = FD.gates_to_voxels(gates, voxeldata)
    @test true

    tfield = FD.compute_squared_thickness_u16(voxeldata)
    @test size(tfield.thick_r_sq) == size(voxeldata.grid)

    fill_time = FD.solve_eikonal_fim_ka(
                            voxeldata.grid,
                            tfield.thick_r_sq,
                            gate_voxels,
    voxeldata.voxel_spacing)
    @test true

end