

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

@testset "Voxelization Solve and Project" verbose = true begin

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

    actual_fill_time = fill_time[isfinite.(fill_time)]
    (tmin, tmax) = extrema(actual_fill_time)
    @info "tmin = $tmin, tmax = $tmax"
    @test length(actual_fill_time) == sum(voxeldata.grid)

    meshproj = FD.project_voxel_to_mesh_triangles(
                    meshdata,
                    voxeldata,
                    fill_time,
                    tfield)


    @test size(meshproj.centroids) == size(meshdata.mesh.faces)
    @test size(meshproj.normals) == size(meshdata.mesh.faces)


    layered_fot = FD.compute_triangle_layered_fot(meshproj)

    @test size(layered_fot.layers.a11)[1] == length(meshdata.mesh.faces)

end