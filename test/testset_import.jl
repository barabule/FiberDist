

@testset "Import Step" verbose = true begin

    # fn = joinpath(pkgdir(@__MODULE__), "assets", "plate_with_ribs_2.step")
    fn = "../assets/plate_with_ribs.step"
    meshdata = FD.load_step_mesh_as_surface(fn)
    @test FD.is_valid(meshdata.mesh.position, meshdata.mesh.faces)
    @test FD.is_valid(meshdata.mesh)
end

@testset "STL import" verbose = true begin
    fn = "test_assets/cube.stl"
    meshdata = FD.MeshData(fn)
    @test FD.is_valid(meshdata.mesh.position, meshdata.mesh.faces)
    @test FD.is_valid(meshdata.mesh)
    @test hasproperty(meshdata, :tlas)
    @test hasproperty(meshdata, :adj)
end

@testset "MeshData" verbose = true begin
    mesh = FD.normal_mesh(FD.Tesselation(FD.Sphere(FD.Point3f(0), 1.0f0), 3))
    meshdata = FD.MeshData(mesh)
    @test meshdata isa FD.MeshData
    @test !FD.is_valid(mesh)

    tet_pts = FD.Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    tet_faces = FD.TriangleFace{Int}[(1, 3, 2), (1, 2, 4), (1, 4, 3), (2, 3, 4)]
    tet_mesh = FD.normal_mesh(tet_pts, tet_faces)
    @test FD.is_valid(tet_mesh)
end



@testset "Voxelization Solve and Project" verbose = true begin

    fn = "../assets/plate_with_ribs.step"
    meshdata = FD.load_step_mesh_as_surface(fn)
    voxeldata = FD.VoxelData(meshdata; fullres = true)
    @test size(voxeldata.grid) == size(voxeldata.thickness)
    # voxeldata = FD.VoxelData(meshdata; fullres = false)
    # @test size(voxeldata.grid) == size(voxeldata.thickness)
    @test size(voxeldata.grid)[1] >= 100

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

    # @test size(layered_fot.layers.a11)[1] == length(meshdata.mesh.faces)
    @test size(layered_fot.layers)[2] ==length(meshdata.mesh.faces)
    #half res
    voxeldata = FD.VoxelData(meshdata; fullres = false)
    @test size(voxeldata.grid) == size(voxeldata.thickness)
    fill_time = FD.solve_eikonal_fim_ka(voxeldata, gates)
    actual_fill_time = fill_time[isfinite.(fill_time)]
    (tmin, tmax) = extrema(actual_fill_time)
    @info "tmin = $tmin, tmax = $tmax"
    @test length(actual_fill_time) == sum(voxeldata.grid)


end

@testset "Stepped Plate Fill Time, Projection & FOT" verbose = true begin
    # 1. Build a 50x50 stepped plate in Gmsh:
    #    - Thin half:  x ∈ [-25, 0], y ∈ [0, 50], z ∈ [0, 2]  (thickness = 2.0)
    #    - Thick half: x ∈ [0, 25],  y ∈ [0, 50], z ∈ [0, 4]  (thickness = 4.0)
    pts   = FD.Point3f[]
    faces = FD.TriangleFace{Int}[]

    gmsh = FD.Gmsh.gmsh
    did_init = gmsh.isInitialized() == 0
    did_init && gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("stepped_plate")

        occ = gmsh.model.occ
        b_thin  = occ.addBox(-25.0, 0.0, 0.0, 25.0, 50.0, 2.0)
        b_thick = occ.addBox(  0.0, 0.0, 0.0, 25.0, 50.0, 4.0)

        # Fuse so the shared interface at x = 0 is conformal and watertight
        occ.fuse([(3, b_thin)], [(3, b_thick)])
        occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", 1.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 2.5)
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        tag_to_idx = Dict{UInt64, Int}(t => i for (i, t) in enumerate(node_tags))
        resize!(pts, length(node_tags))
        for i in eachindex(node_tags)
            pts[i] = FD.Point3f(node_coords[3i - 2], node_coords[3i - 1], node_coords[3i])
        end

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2)
        for (etype, enodes) in zip(elem_types, elem_node_tags)
            etype == 2 || continue
            for k in 1:3:length(enodes)
                push!(faces, FD.TriangleFace{Int}(
                    tag_to_idx[enodes[k]],
                    tag_to_idx[enodes[k + 1]],
                    tag_to_idx[enodes[k + 2]]
                ))
            end
        end
        gmsh.model.remove()
    finally
        did_init && gmsh.finalize()
    end

    mesh     = FD.normal_mesh(pts, faces)
    meshdata = FD.MeshData(mesh)
    n_faces  = length(meshdata.mesh.faces)

    # Precompute CSR adjacency once and reuse across fullres = true/false runs
    adj = FD.MeshAdjacency(meshdata.mesh)

    function world_to_voxel(vd, p)
        dx, dy, dz = vd.voxel_spacing
        cx, cy, cz = vd.corner
        ix = clamp(round(Int, (p[1] - cx) / dx + 0.5), 1, size(vd.grid, 1))
        iy = clamp(round(Int, (p[2] - cy) / dy + 0.5), 1, size(vd.grid, 2))
        iz = clamp(round(Int, (p[3] - cz) / dz + 0.5), 1, size(vd.grid, 3))
        return (ix, iy, iz)
    end

    nearest_tri(centroids, target) = argmin([sum(abs2, c - target) for c in centroids])

    @testset "Resolution (fullres = $fullres)" for fullres in (true, false)
        voxeldata = FD.VoxelData(meshdata; fullres = fullres)
        @test size(voxeldata.grid) == size(voxeldata.thickness)

        # Gate at the middle of the 50x50 plate (x = 0, y = 25, z = 1.0)
        gates = [FD.ValveGate(position = FD.Point3f(0.0f0, 25.0f0, 1.0f0), radius = 2.0f0)]

        # 2. Verify interior voxel wall thickness on thin vs. thick sides
        ix_thin,  iy_mid, iz_thin  = world_to_voxel(voxeldata, (-15.0f0, 25.0f0, 1.0f0))
        ix_thick, _,      iz_thick = world_to_voxel(voxeldata, ( 15.0f0, 25.0f0, 2.0f0))

        @test voxeldata.grid[ix_thin, iy_mid, iz_thin]
        @test voxeldata.grid[ix_thick, iy_mid, iz_thick]

        wt_thin  = FD.wallthickness(voxeldata, ix_thin,  iy_mid, iz_thin)
        wt_thick = FD.wallthickness(voxeldata, ix_thick, iy_mid, iz_thick)
        @info "fullres=$fullres | voxel wt_thin = $wt_thin (~2.0), wt_thick = $wt_thick (~4.0)"

        @test wt_thick > 1.5 * wt_thin
        @test isapprox(wt_thin,  2.0; rtol = 0.25)
        @test isapprox(wt_thick, 4.0; rtol = 0.25)

        # 3. Solve Eikonal fill time and check interior corner arrival times
        fill_time = FD.solve_eikonal_fim_ka(voxeldata, gates)
        actual_fill_time = fill_time[isfinite.(fill_time)]
        @test length(actual_fill_time) == sum(voxeldata.grid)

        c_thin  = world_to_voxel(voxeldata, (-23.0f0, 48.0f0, 1.0f0))
        c_thick = world_to_voxel(voxeldata, ( 23.0f0, 48.0f0, 2.0f0))

        t_vox_thin  = fill_time[c_thin...]
        t_vox_thick = fill_time[c_thick...]
        @info "fullres=$fullres | voxel t_thin = $t_vox_thin, t_thick = $t_vox_thick (ratio = $(t_vox_thin / t_vox_thick))"

        @test isfinite(t_vox_thin) && isfinite(t_vox_thick)
        @test t_vox_thick < t_vox_thin
        @test (t_vox_thin / t_vox_thick) > 2.5

        # 4. Project voxel fields onto surface mesh triangles
        meshproj = FD.project_voxel_to_mesh_triangles(
            meshdata,
            voxeldata,
            fill_time;
        )

        @test length(meshproj.centroids) == n_faces
        @test length(meshproj.normals) == n_faces
        @test all(isfinite, meshproj.fill_time)

        # Query triangles on the bottom face (z = 0) near the symmetric far corners
        tri_corner_thin  = nearest_tri(meshproj.centroids, FD.SVector{3, Float32}(-23.5f0, 48.5f0, 0.0f0))
        tri_corner_thick = nearest_tri(meshproj.centroids, FD.SVector{3, Float32}( 23.5f0, 48.5f0, 0.0f0))

        t_tri_thin  = meshproj.fill_time[tri_corner_thin]
        t_tri_thick = meshproj.fill_time[tri_corner_thick]
        h_tri_thin  = meshproj.thickness[tri_corner_thin]
        h_tri_thick = meshproj.thickness[tri_corner_thick]
        @info "fullres=$fullres | tri t_thin = $t_tri_thin, t_thick = $t_tri_thick (ratio = $(t_tri_thin / t_tri_thick))"

        @test isfinite(t_tri_thin) && isfinite(t_tri_thick)
        @test t_tri_thick < t_tri_thin
        @test (t_tri_thin / t_tri_thick) > 2.5
        @test h_tri_thick > 1.5 * h_tri_thin

        # Flow vectors on the bottom plate should lie in the XY plane (|u_z| ≈ 0)
        u_thin  = meshproj.flow_dir[tri_corner_thin]
        u_thick = meshproj.flow_dir[tri_corner_thick]
        @test isapprox(FD.norm(u_thin),  1.0f0; atol = 1f-4)
        @test isapprox(FD.norm(u_thick), 1.0f0; atol = 1f-4)
        @test abs(u_thin[3])  < 1f-3
        @test abs(u_thick[3]) < 1f-3

        # 5. Compute N-layer through-thickness 5-component FOT and verify invariants
        n_layers = 10
        layered_fot = FD.compute_triangle_layered_fot(meshproj; n_layers = n_layers)

        @test size(layered_fot.layers) == (n_layers, n_faces)
        @test length(layered_fot.shell_avg) == n_faces

        for idx in (tri_corner_thin, tri_corner_thick)
            # a11_avg = layered_fot.shell_avg.a11[idx]
            # a22_avg = layered_fot.shell_avg.a22[idx]
            # a33_avg = FD.FD.get_a33(a11_avg, a22_avg)
            fot_avg = layered_fot.shell_avg[idx]
            a11_avg, a22_avg, a33_avg = FD.get_a11(fot_avg), FD.get_a22(fot_avg), FD.get_a33(fot_avg)
            @test isapprox(a11_avg + a22_avg + a33_avg, 1.0f0; atol = 1f-5)
            @test a33_avg < 0.12f0

            e1 = meshproj.flow_dir[idx]
          
            A_skin = layered_fot.layers[end, idx]
            A_skin_11, A_skin_22, A_skin_12 = FD.get_a11(A_skin), FD.get_a22(A_skin), FD.get_a12(A_skin)
            a_flow_skin = A_skin_11 * e1[1]^2 + A_skin_22 * e1[2]^2 + 2.0f0 * A_skin_12 * e1[1] * e1[2]
            @test a_flow_skin > 0.75f0
        end
    end
end


@testset "Extrapolate" verbose = true begin
    # Watertight, non-degenerate icosahedron (12 vertices, 20 equilateral faces)
    ϕ = (1.0f0 + sqrt(5.0f0)) / 2.0f0
    verts = FD.Point3f[
        (-1,  ϕ,  0), ( 1,  ϕ,  0), (-1, -ϕ,  0), ( 1, -ϕ,  0),
        ( 0, -1,  ϕ), ( 0,  1,  ϕ), ( 0, -1, -ϕ), ( 0,  1, -ϕ),
        ( ϕ,  0, -1), ( ϕ,  0,  1), (-ϕ,  0, -1), (-ϕ,  0,  1)
    ]
    faces = FD.TriangleFace{Int}[
        (1, 12,  6), (1,  6,  2), (1,  2,  8), (1,  8, 11), (1, 11, 12),
        (2,  6, 10), (6, 12,  5), (12, 11, 3), (11, 8,  7), (8,  2,  9),
        (4, 10,  5), (4,  5,  3), (4,  3,  7), (4,  7,  9), (4,  9, 10),
        (5, 10,  6), (3,  5, 12), (7,  3, 11), (9,  7,  8), (10, 9,  2)
    ]
    mesh = FD.normal_mesh(verts, faces)
    meshdata = FD.MeshData(mesh)

    interp_f32 = FD.AngleWeightInterpolator(meshdata, Float32)
    interp_f64 = FD.AngleWeightInterpolator(meshdata, Float64)
    n_faces = length(mesh.faces)
    n_verts = length(mesh.position)

    # 1. Partition of unity & scalar types (Float32 and Float64)
    @testset "Scalars ($FT)" for (interp, FT) in ((interp_f32, Float32), (interp_f64, Float64))
        const_faces = fill(FT(3.5), n_faces)
        const_verts = interp(const_faces)
        @test const_verts isa Vector{FT}
        @test size(const_verts) == (n_verts,)
        @test all(isfinite, const_verts)
        @test isapprox(const_verts, fill(FT(3.5), n_verts); rtol = FT(1e-5))

        facedata = rand(FT, n_faces)
        vertdata = interp(facedata)
        buffer = similar(vertdata)
        vertdata2 = interp(buffer, facedata)
        @test vertdata2 === buffer
        @test eltype(vertdata2) == FT
        @test vertdata == vertdata2
    end

    # 2. StaticVectors (3D flow directions & 5-component compact FOT)
    @testset "StaticVectors" begin
        # SVector{3, Float32}
        v3_faces = [FD.SVector{3, Float32}(1.0f0, -2.0f0, 0.5f0) for _ in 1:n_faces]
        v3_verts = interp_f32(v3_faces)
        @test v3_verts isa Vector{FD.SVector{3, Float32}}
        @test all(v -> isapprox(v, FD.SVector{3, Float32}(1.0f0, -2.0f0, 0.5f0); rtol = 1f-5), v3_verts)

        v3_rand = [rand(FD.SVector{3, Float32}) for _ in 1:n_faces]
        v3_buf = similar(v3_verts)
        @test interp_f32(v3_buf, v3_rand) === v3_buf
        @test v3_buf == interp_f32(v3_rand)

        # SVector{5, Float32} (5-component symmetric FOT: a11, a22, a12, a13, a23)
        fot5_const = FD.SVector{5, Float32}(0.70f0, 0.20f0, 0.05f0, -0.02f0, 0.01f0)
        fot5_faces = fill(fot5_const, n_faces)
        fot5_verts = interp_f32(fot5_faces)
        @test fot5_verts isa Vector{FD.SVector{5, Float32}}
        @test all(v -> isapprox(v, fot5_const; rtol = 1f-5), fot5_verts)
    end

    # 3. GeometryBasics types (Vec3f / Point3f)
    @testset "GeometryBasics Vec3f & Point3f" begin
        vec_const = FD.Vec3f(0.25f0, -0.5f0, 1.0f0)
        vec_verts = interp_f32(fill(vec_const, n_faces))
        @test vec_verts isa Vector{FD.Vec3f}
        @test all(v -> isapprox(v, vec_const; rtol = 1f-5), vec_verts)

        pt_const = FD.Point3f(2.0f0, 4.0f0, -1.0f0)
        pt_verts = interp_f32(fill(pt_const, n_faces))
        @test pt_verts isa Vector{FD.Point3f}
        @test all(p -> isapprox(p, pt_const; rtol = 1f-5), pt_verts)
    end

    # 4. Small StaticMatrices (full 3×3 Fiber Orientation Tensor)
    @testset "SMatrix{3, 3}" begin
        A_const = FD.SMatrix{3, 3, Float32, 9}(
            0.72f0, 0.04f0, -0.01f0,
            0.04f0, 0.20f0,  0.02f0,
           -0.01f0, 0.02f0,  0.08f0
        )
        mat_faces = fill(A_const, n_faces)
        mat_verts = interp_f32(mat_faces)
        @test mat_verts isa Vector{FD.SMatrix{3, 3, Float32, 9}}
        @test all(M -> isapprox(M, A_const; rtol = 1f-5), mat_verts)

        mat_rand = [rand(FD.SMatrix{3, 3, Float32, 9}) for _ in 1:n_faces]
        mat_buf = similar(mat_verts)
        @test interp_f32(mat_buf, mat_rand) === mat_buf
        @test mat_buf == interp_f32(mat_rand)
    end
end


@testset "Mesh Loading" verbose = true begin
    state = FD.init_app_state()
    FD.setup_3d_scene!(state)

    # Downstream observables in AppState must stay wired to the struct's Observables
    @test !isempty(state.meshdata.listeners)
    @test !isempty(state.voxeldata.listeners)
    @test !isempty(state.stats.listeners)
    @test !isempty(state.vertex_itp_f32.listeners)

    init_vd_size = size(state.voxeldata[].grid)
    init_n_faces = length(state.stats[].fill_time)
    init_n_verts = length(state.fill_time[])
    @test init_n_faces == length(state.meshdata[].mesh.faces)
    @test init_n_verts == length(state.meshdata[].mesh.position)

    # 1. Load a different geometry (cube.stl) with voxel_confirmed = false
    cube_fn = joinpath(@__DIR__, "test_assets", "cube.stl")
    cube_md = FD.MeshData(cube_fn)
    state.is_cad_file[] = false
    state.voxel_confirmed[] = false
    state.meshdata[] = cube_md

    cube_vd = state.voxeldata[]
    @test size(cube_vd.grid) != init_vd_size
    @test size(cube_vd.grid, 1) == size(cube_vd.grid, 2) == size(cube_vd.grid, 3)
    @test length(state.fill_time[]) == length(cube_md.mesh.position)

    # 2. Change voxelization resolution and verify it voxelizes the new mesh (not the old one)
    new_res = 50
    state.res[] = new_res
    cube_vd_50 = state.voxeldata[]
    @test size(cube_vd_50.grid) != size(cube_vd.grid)
    @test size(cube_vd_50.grid, 1) == size(cube_vd_50.grid, 2) == size(cube_vd_50.grid, 3)
    @test (new_res + 2) <= size(cube_vd_50.grid, 1) <= (new_res + 4)

    # 3. Confirm voxelization -> trigger solve & surface projection on the new mesh
    state.voxel_confirmed[] = true
    @test length(state.stats[].fill_time) == length(cube_md.mesh.faces)
    @test length(state.fill_time[]) == length(cube_md.mesh.position)
    @test all(isfinite, state.stats[].fill_time)
    @test all(isfinite, state.fill_time[])

    # 4. Remesh / load a STEP geometry again and verify sync from STL -> STEP
    step_md = FD.load_step_mesh_as_surface(
        state.current_filepath[];
        target_element_size = 3.0,
        min_curvature_size = 2.0,
        min_circle_points = 10,
    )
    state.is_cad_file[] = true
    state.voxel_confirmed[] = false
    state.meshdata[] = step_md

    step_vd = state.voxeldata[]
    @test size(step_vd.grid) != size(cube_vd_50.grid)
    @test !(size(step_vd.grid, 1) == size(step_vd.grid, 2) == size(step_vd.grid, 3))
    @test length(state.fill_time[]) == length(step_md.mesh.position)

    state.voxel_confirmed[] = true
    @test length(state.stats[].fill_time) == length(step_md.mesh.faces)
    @test length(state.fill_time[]) == length(step_md.mesh.position)
    @test all(isfinite, state.stats[].fill_time)
end


@testset "Save & Load" verbose = true begin


    # 1. Construct a minimal 4-vertex, 4-triangle tetrahedron MeshData (no GUI/CAD)
    pts = Point3f[
        Point3f(0.0f0,  0.0f0,  0.0f0),
        Point3f(10.0f0, 0.0f0,  0.0f0),
        Point3f(5.0f0,  8.66f0, 0.0f0),
        Point3f(5.0f0,  2.89f0, 8.16f0),
    ]
    tris = GeometryBasics.TriangleFace{Int}[
        GeometryBasics.TriangleFace(1, 3, 2),
        GeometryBasics.TriangleFace(1, 2, 4),
        GeometryBasics.TriangleFace(2, 3, 4),
        GeometryBasics.TriangleFace(3, 1, 4),
    ]
    tet_md = FD.MeshData(GeometryBasics.mesh(pts, tris))

    gates = [
        FD.ValveGate(position = SVector{3, Float32}(0.0f0, 0.0f0, 0.0f0), radius = 1.2f0, open_time = 0.0f0),
        FD.ValveGate(position = SVector{3, Float32}(10.0f0, 0.0f0, 0.0f0), radius = 2.5f0, open_time = 0.35f0),
    ]

    src = (
        current_filepath    = "dummy_tet.step",
        is_cad_file         = true,
        cad_elem_size       = 1.5,
        cad_min_curv        = 0.8,
        cad_circle_pts      = 20,
        meshdata            = tet_md,
        res                 = 48,
        gates               = gates,
        polymer             = :pa66,
        fiber               = :cf,
        fiber_wt_fraction   = 0.40f0,
        num_half_thk_layers = 9,
    )

    # 2. Verify Observable-wrapped fields produce the exact same Dict as plain values
    src_obs = NamedTuple{keys(src)}(FD.Observable.(values(src)))
    @test FD.project_to_dict(src) == FD.project_to_dict(src_obs)

    # 3. Roundtrip through JLD2 on disk
    mktempdir() do tmpdir
        path = joinpath(tmpdir, "tet_project.jld2")
        FD.save_project_dict(path, FD.project_to_dict(src))
        @test isfile(path)

        restored = FD.dict_to_project(FD.load_project_dict(path))

        @test restored.current_filepath    == src.current_filepath
        @test restored.is_cad_file         == src.is_cad_file
        @test restored.cad_elem_size       ≈  src.cad_elem_size
        @test restored.cad_min_curv        ≈  src.cad_min_curv
        @test restored.cad_circle_pts      == src.cad_circle_pts
        @test restored.res                 == src.res
        @test restored.polymer             == src.polymer
        @test restored.fiber               == src.fiber
        @test restored.fiber_wt_fraction   ≈  src.fiber_wt_fraction
        @test restored.num_half_thk_layers == src.num_half_thk_layers

        @test GeometryBasics.coordinates(restored.meshdata.mesh) == GeometryBasics.coordinates(tet_md.mesh)
        @test GeometryBasics.faces(restored.meshdata.mesh)       == GeometryBasics.faces(tet_md.mesh)

        @test length(restored.gates) == length(src.gates)
        for (g_out, g_in) in zip(restored.gates, src.gates)
            @test g_out.position  ≈ g_in.position
            @test g_out.radius    ≈ g_in.radius
            @test g_out.open_time ≈ g_in.open_time
        end
    end
end

