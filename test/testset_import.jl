

@testset "Import Step" verbose = true begin

    # fn = joinpath(pkgdir(@__MODULE__), "assets", "plate_with_ribs_2.step")
    fn = "../assets/plate_with_ribs.step"
    meshdata = FD.load_step_mesh_as_surface(fn)
    @test FD.is_valid(meshdata.mesh)

end

@testset "STL import" verbose = true begin
    fn = "test_assets/cube.stl"
    meshdata = FD.MeshData(fn)
    @test FD.is_valid(meshdata.mesh)
    @test hasproperty(meshdata, :tlas)
end

@testset "MeshData" verbose = true begin
    mesh = FD.normal_mesh(FD.Tesselation(FD.Sphere(FD.Point3f(0), 1.0f0), 3))
    meshdata = FD.MeshData(mesh)
    @test true
end



@testset "Voxelization Solve and Project" verbose = true begin

    fn = "../assets/plate_with_ribs.step"
    meshdata = FD.load_step_mesh_as_surface(fn)
    voxeldata = FD.VoxelData(meshdata; fullres = true)
    @test size(voxeldata.grid) == size(voxeldata.thickness)
    # voxeldata = FD.VoxelData(meshdata; fullres = false)
    # @test size(voxeldata.grid) == size(voxeldata.thickness)
    @test size(voxeldata.grid)[1] == 100

    wt = FD.wallthickness(voxeldata, 25)
    wt2 = FD.wallthickness(voxeldata, 12, 4, 5)

    gates = [FD.ValveGate(position = FD.Point3f(0, 25, 0), radius = 2.0) ]
    @test true

    
    fill_time = FD.solve_eikonal_fim_ka(voxeldata, gates)
    @info "fill_time", length(isfinite.(fill_time))
    @test true

    actual_fill_time = fill_time[isfinite.(fill_time)]
    (tmin, tmax) = extrema(actual_fill_time)
    @info "tmin = $tmin, tmax = $tmax"
    @test length(actual_fill_time) == sum(voxeldata.grid)

    meshproj = FD.project_voxel_to_mesh_triangles(
                    meshdata,
                    voxeldata,
                    fill_time)


    @test size(meshproj.centroids) == size(meshdata.mesh.faces)
    @test size(meshproj.normals) == size(meshdata.mesh.faces)


    layered_fot = FD.compute_triangle_layered_fot(meshproj)

    @test size(layered_fot.layers.a11)[1] == length(meshdata.mesh.faces)

end


@testset "Extrapolate" verbose = true begin


end