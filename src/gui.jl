function main()

    fig = Figure()

    ax = Axis3(fig[1,1], aspect = :data)


    BOTTOMBAR = GridLayout(fig[2,1], tellwidth = false)


    fn_step = Observable(joinpath(pkgdir(@__MODULE__), "assets", "plate_with_ribs_2.step"))
    # meshdata = Observable(load_step_mesh_as_surface(fn_step[]))
    meshdata = lift(fn_step) do fn
        load_step_mesh_as_surface(fn)
    end
    res = Observable(100)
    # voxeldata = Observable(voxelize_mesh(meshdata[]; res = res[]))
    voxeldata = lift(meshdata, res) do md, r
        voxelize_mesh(md; res = r)
    end

    # interior_points = Observable(mask_points(voxeldata[].grid, voxeldata[].voxel_spacing, voxeldata[].corner))
    interior_points = lift(voxeldata) do vd
        mask_points(vd.grid, vd.voxel_spacing, vd.corner)
    end


    p1 = Point3f(0, 25, 0)
    p2 = Point3f(100, 25,0)
    t2 = Observable(0.0)


    gates = lift(t2) do t
        [
            ValveGate(p1, 0.0),
            ValveGate(p2, t)
        ]
    end

    thickness_field = lift(voxeldata) do vd
        compute_squared_thickness_u16(vd)
    end

    fill_time = lift(voxeldata, thickness_field, gates) do vd, tf, gt
        solve_eikonal_voxel(vd, tf, gt)
    end

    stats = lift(fill_time, voxeldata, thickness_field) do ft, vd, tf
        analyze_eikonal_field(ft, vd, tf)
    end


    mesh!(ax, meshdata[].mesh, color = (:lightgray, 0.25), transparency = true)

    # scatter!(ax, interior_points, color = :black)

    interior_flow_dirs = lift(fill_time, stats) do ft,st
        st.flow_dirs[isfinite.(ft)]
    end


    lengthscale = Observable(0.5)
    arrows3d!(ax, interior_points, interior_flow_dirs, color = (:black, 0.5), lengthscale = lengthscale)



    tb_res = Textbox(fig, placeholder = "100", validator = Int)
    tb_lenscale = Textbox(fig, placeholder=  "0.5", validator = Float64)
    tb_t2 = Textbox(fig, placeholder = "0.0", validator = Float64)
    BOTTOMBAR[1,1] = vgrid!(
        hgrid!(Label(fig, "Resolution"), tb_res, Label(fig, "Lengthscale"), tb_lenscale, Label(fig, "Gate 2 time"), tb_t2),
    )

    on(tb_res.stored_string) do s   
        new_res = tryparse(Int, s)
        if new_res>1
            res[] = new_res
        end
    end
    on(tb_lenscale.stored_string) do s
        ls = tryparse(Float64, s)
        lengthscale[] = ls
    end
    on(tb_t2.stored_string) do s
        t = tryparse(Float64, s)
        t2[] = clamp(t, 0.0, Inf)
    end

    return fig


end