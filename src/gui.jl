function main()

    fig = Figure()

    ax = Axis3(fig[1,1], aspect = :data, protrusions = 0)
    

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


    vcolor = lift(fill_time, voxeldata, meshdata) do ft, vd, md
        project_voxel_to_mesh_vertex(md, vd, ft)
    end

    num_color_bands = Observable(16)
    cmap_name = Observable(:turbo)

    # mesh!(ax, meshdata[].mesh, color = (:lightgray, 0.25), transparency = true)
    mplot = mesh!(ax, meshdata[].mesh, color = vcolor, colormap = @lift(cgrad($cmap_name, $num_color_bands, categorical = true)))

    cb = Colorbar(fig[1,1], 
                    mplot;
                halign = :right, 
                valign = :top,
                tellwidth = false,
                tellheight = false,
                label = "Fill Time",
                height = 280,
                width = 18,
                tickformat = "{:.3f}",
                ticklabelsize = 12,
                labelsize = 13,
                labelpadding = 6,
                ticklabelpad = 6,
                )
                


    # scatter!(ax, interior_points, color = :black)

    interior_flow_dirs = lift(fill_time, stats) do ft,st
        st.flow_dirs[isfinite.(ft)]
    end


    lengthscale = Observable(0.5)
    # arrows3d!(ax, interior_points, interior_flow_dirs, color = (:black, 0.5), lengthscale = lengthscale)

    # # Plot weld lines (red) and air traps (yellow) over a semi-transparent part mesh
# mesh!(ax, meshdata.mesh, color = (:lightgray, 0.25), transparency = true)
# scatter!(ax, mask_points(res.masks.weld_lines, voxeldata.stepsize, voxeldata.corner), color = :red, markersize = 6)
# scatter!(ax, mask_points(res.masks.air_traps,  voxeldata.stepsize, voxeldata.corner), color = :gold, markersize = 10)

    cb_weld = Checkbox(fig, checked = false)
    cb_air = Checkbox(fig, checked = false)

    weld_lines = lift(voxeldata, stats) do vd, st
        mask_points(st.masks.weld_lines, vd.voxel_spacing, vd.corner)
    end

    scatter!(ax, weld_lines, color = :red, markersize = 6, alpha = @lift($(cb_weld.checked) ? 1.0 : 0.0))
    
    air_traps = lift(voxeldata, stats) do vd, st
        mask_points(st.masks.air_traps, vd.voxel_spacing, vd.corner)
    end
    
    scatter!(ax, air_traps, color = :gold, markersize = 10, alpha = @lift($(cb_air.checked) ? 1.0 : 0.0))

    tb_res = Textbox(fig, placeholder = "100", validator = Int)
    tb_lenscale = Textbox(fig, placeholder=  "0.5", validator = Float64)
    tb_t2 = Textbox(fig, placeholder = "0.0", validator = Float64)

    BOTTOMBAR[1,1] = vgrid!(
        hgrid!(Label(fig, "Resolution"), tb_res, Label(fig, "Lengthscale"), tb_lenscale, Label(fig, "Gate 2 time"), tb_t2),
        hgrid!(Label(fig, "Draw Weld Lines"), cb_weld, Label(fig, "Draw Air Traps"), cb_air),
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