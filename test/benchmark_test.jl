
const RUN_BENCHMARKS = get(ENV, "FIBERDIST_BENCH", "false") == "true" || ("--bench" in ARGS)

if RUN_BENCHMARKS
    @testset "Performance Benchmarks (plate_with_ribs.step)" verbose = true begin
        @info "Running performance benchmarks on $(Threads.nthreads()) thread(s)..."

        fn = joinpath(@__DIR__, "..", "assets", "plate_with_ribs.step")
        gates = [FD.ValveGate(position = FD.Point3f(0.0f0, 25.0f0, 0.0f0), radius = 2.0f0)]

        # Warmup pass (compiles all kernels and functions before timing)
        md_warm = FD.load_step_mesh_as_surface(fn)
        vd_warm = FD.VoxelData(md_warm; fullres = false)
        ft_warm = FD.solve_eikonal_fim_ka(vd_warm, gates)
        mp_warm = FD.project_voxel_to_mesh_triangles(md_warm, vd_warm, ft_warm)
        FD.compute_triangle_layered_fot(mp_warm)

        # Helper to measure best-of-N runs without BenchmarkTools overhead
        function bench_stage(f, n_runs = 3)
            best_t = Inf
            best_bytes = 0
            local last_res
            for _ in 1:n_runs
                GC.gc(false)
                stats = @timed f()
                if stats.time < best_t
                    best_t = stats.time
                    best_bytes = stats.bytes
                end
                last_res = stats.value
            end
            return last_res, best_t, best_bytes
        end

        meshdata, t_cad, b_cad = bench_stage(() -> FD.load_step_mesh_as_surface(fn), 2)
        n_tris = length(meshdata.mesh.faces)
        fullres = false
        for r in (100, 200, 300)
            voxeldata, t_vox, b_vox = bench_stage(() -> FD.VoxelData(meshdata; res = r), 3)
            n_active = sum(voxeldata.grid)
            grid_dims = size(voxeldata.grid)

            fill_time, t_solve, b_solve = bench_stage(() -> FD.solve_eikonal_fim_ka(voxeldata, gates), 5)
            meshproj, t_proj, b_proj = bench_stage(
                () -> FD.project_voxel_to_mesh_triangles(meshdata, voxeldata, fill_time), 5
            )
            layered_fot, t_fot, b_fot = bench_stage(
                () -> FD.compute_triangle_layered_fot(meshproj; n_layers = 10), 5
            )

            t_total = t_vox + t_solve + t_proj + t_fot
            mvox_per_sec = (n_active / 1e6) / t_solve

            @info """
            === Benchmark Summary (fullres = $fullres | threads = $(Threads.nthreads())) ===
            Mesh triangles      : $n_tris
            Voxel grid / active : $grid_dims / $n_active voxels
            1. CAD Import/Mesh  : $(round(t_cad * 1000; digits = 2)) ms  ($(round(b_cad / 1024^2; digits = 2)) MiB)
            2. Voxelization     : $(round(t_vox * 1000; digits = 2)) ms  ($(round(b_vox / 1024^2; digits = 2)) MiB)
            3. Eikonal FIM      : $(round(t_solve * 1000; digits = 2)) ms  ($(round(b_solve / 1024^2; digits = 2)) MiB) [$(round(mvox_per_sec; digits = 2)) MVox/s]
            4. Surface Project  : $(round(t_proj * 1000; digits = 2)) ms  ($(round(b_proj / 1024^2; digits = 2)) MiB)
            5. Layered FOT (10) : $(round(t_fot * 1000; digits = 2)) ms  ($(round(b_fot / 1024^2; digits = 2)) MiB)
            ---------------------------------------------------------
            Pipeline (Stages 2-5): $(round(t_total * 1000; digits = 2)) ms
            """

            @test t_total > 0
            @test length(layered_fot.shell_avg.a11) == n_tris
        end
    
    end
end