
function set_overdraw_recursive!(scene::Scene)
    for p in scene.plots
        p.overdraw[] = true
    end
    for child in scene.children
        set_overdraw_recursive!(child)
    end
end

function main()
    fig = Figure(size = (1400, 850))

    rowsize!(fig.layout, 1, Relative(1.0))
    rowgap!(fig.layout, 0)

    # 1. Background 3D CAD Scene
    lscene = LScene(
        fig[1, 1];
        show_axis = false,
        scenekw = (
            backgroundcolor = RGBf(0.96, 0.96, 0.97),
            clear = true,
        )
    )

    BOTTOMBAR = GridLayout(fig[2, 1], tellwidth = false, tellheight = true)

    fn_step = Observable(joinpath(pkgdir(@__MODULE__), "assets", "plate_with_ribs_2.step"))
    meshdata = lift(fn_step) do fn
        load_step_mesh_as_surface(fn; target_element_size = 1.0)
    end

    res = Observable(100)
    voxeldata = lift(meshdata, res) do md, r
        VoxelData(md; res = r)
    end

    interior_points = lift(voxeldata) do vd
        mask_points(vd.grid, vd.voxel_spacing, vd.corner)
    end

    p1 = Point3f(0, 25, 0)
    p2 = Point3f(100, 25, 0)
    t2 = Observable(0.0)

    gates = lift(t2) do t
        [
            ValveGate(position = p1, open_time = 0.0, radius = 2.0),
            ValveGate(position = p2, open_time = t, radius = 2.0)
        ]
    end

    stats = lift(meshdata, voxeldata, gates) do md, vd, gts
        filltime = solve_eikonal_fim_ka(vd, gts)
        project_voxel_to_mesh_triangles(md, vd, filltime)
    end

    vertex_itp_f32 = lift(meshdata) do md
        AngleWeightInterpolator(md, Float32)
    end

    vertex_f32_buffer = lift(meshdata) do md
        zeros(Float32, length(md.mesh.position))
    end

    fill_time = lift(vertex_itp_f32, stats, vertex_f32_buffer) do itp, st, buff
        itp(buff, st.fill_time)
    end

    # Track total fill time range [0, t_max]
    t_max = lift(fill_time) do ft
        m = maximum(ft)
        isfinite(m) && m > 0.0f0 ? m : 1.0f0
    end
    color_limits = @lift((0.0f0, $t_max))

    # 2. Dynamic Fill-Front Colormap (30 stops: values above current time are grey)
    n_stops = 30
    fill_step = Observable(n_stops) # start at 30 (fully filled); drag to 1 for all-grey
    cmap_name = Observable(:turbo)
    unfilled_grey = RGBAf(0.78, 0.78, 0.80, 1.0)

    # Full unmasked categorical colormap for the Colorbar legend
    base_cmap = lift(cmap_name) do cname
        cgrad(cname, n_stops, categorical = true)
    end

    # Masked categorical colormap for the mesh: bands beyond current fill fraction become grey
    mesh_cmap = lift(cmap_name, fill_step) do cname, step
        full_colors = cgrad(cname, n_stops, categorical = true).colors
        n_filled = clamp(round(Int, (step - 1) / (n_stops - 1) * n_stops), 0, n_stops)
        masked_colors = [i <= n_filled ? full_colors[i] : unfilled_grey for i in 1:n_stops]
        cgrad(masked_colors, n_stops, categorical = true)
    end

    mplot = mesh!(
        lscene,
        @lift($meshdata.mesh);
        color = fill_time,
        colorrange = color_limits,
        colormap = mesh_cmap,
        shading = true,
    )

    # Orthographic CAD camera setup with zero near-plane clipping
    center!(lscene.scene)
    cam = cameracontrols(lscene.scene)
    cam.settings.projectiontype[] = Makie.Orthographic
    cam.settings.cad[] = true
    cam.settings.zoom_shift_lookat[] = true

    # Prevent Makie's update_cam! from overriding near/far with aggressive positive clipping
    if hasproperty(cam.settings, :near_clip)
        cam.settings.near_clip[] = 1e-5
    end
    cam.near[] = -500.0f0
    cam.far[]  = 5000.0f0

    # Keep near < 0 whenever Camera3D recomputes camera matrices on scroll/orbit
    on(cam.eyeposition) do _
        if cam.near[] > -500.0f0
            cam.near[] = -500.0f0
            cam.far[]  = 5000.0f0
        end
    end
    update_cam!(lscene.scene, cam)

    # 3. Dedicated Pixel-Space Overlay Scene (never occluded by 3D mesh depth)
    overlay_scene = Scene(fig.scene; clear = false, camera = campixel!)

    # --- A. Top-Right Floating Colorbar ---
    bar_w, bar_h = 16, 260
    card_w, card_h = 114, bar_h + 24

    cb_card_rect = lift(lscene.scene.viewport) do vp
        x0 = vp.origin[1] + vp.widths[1] - card_w - 14
        y0 = vp.origin[2] + vp.widths[2] - card_h - 14
        Rect2f(x0, y0, card_w, card_h)
    end

    cb_bbox = @lift(BBox(
        $cb_card_rect.origin[1] + 12,
        $cb_card_rect.origin[1] + $cb_card_rect.widths[1] - 10,
        $cb_card_rect.origin[2] + 12,
        $cb_card_rect.origin[2] + $cb_card_rect.widths[2] - 12
    ))

    poly!(
        overlay_scene,
        cb_card_rect,
        color = RGBAf(1.0, 1.0, 1.0, 0.85),
        strokecolor = RGBAf(0.5, 0.5, 0.5, 0.4),
        strokewidth = 1.0,
        overdraw = true,
    )

    cb = Colorbar(
        overlay_scene;
        bbox = cb_bbox,
        limits = color_limits,
        colormap = base_cmap,
        width = bar_w,
        height = bar_h,
        halign = :left,
        valign = :center,
        label = "Fill Time",
        tickformat = "{:.3f}",
        ticklabelsize = 12,
        labelsize = 13,
        labelpadding = 6,
        ticklabelpad = 6,
        tickalign = 0,
        tellwidth = false,
        tellheight = false,
    )
    set_overdraw_recursive!(cb.blockscene)
    on(color_limits) do _
        set_overdraw_recursive!(cb.blockscene)
    end

    # --- B. Bottom Floating Fill-Time Animation Slider (30 stops) ---
    slider_card_h = 54
    slider_card_rect = lift(lscene.scene.viewport) do vp
        w = clamp(vp.widths[1] * 0.52, 340.0, 680.0)
        x0 = vp.origin[1] + (vp.widths[1] - w) * 0.5
        y0 = vp.origin[2] + 16
        Rect2f(x0, y0, w, slider_card_h)
    end

    poly!(
        overlay_scene,
        slider_card_rect,
        color = RGBAf(1.0, 1.0, 1.0, 0.88),
        strokecolor = RGBAf(0.5, 0.5, 0.5, 0.4),
        strokewidth = 1.0,
        overdraw = true,
    )

    btn_bbox = @lift(BBox(
        $slider_card_rect.origin[1] + 12,
        $slider_card_rect.origin[1] + 46,
        $slider_card_rect.origin[2] + 11,
        $slider_card_rect.origin[2] + $slider_card_rect.widths[2] - 11
    ))

    slider_bbox = @lift(BBox(
        $slider_card_rect.origin[1] + 56,
        $slider_card_rect.origin[1] + $slider_card_rect.widths[1] - 165,
        $slider_card_rect.origin[2] + 11,
        $slider_card_rect.origin[2] + $slider_card_rect.widths[2] - 11
    ))

    lbl_pos = @lift(Point2f(
        $slider_card_rect.origin[1] + $slider_card_rect.widths[1] - 155,
        $slider_card_rect.origin[2] + $slider_card_rect.widths[2] * 0.5
    ))

    is_playing = Observable(false)
    play_btn = Button(
        overlay_scene;
        bbox = btn_bbox,
        label = @lift($is_playing ? "||" : "▶"),
        fontsize = 13,
        tellwidth = false,
        tellheight = false,
    )

    fill_slider = Slider(
        overlay_scene;
        bbox = slider_bbox,
        range = 1:n_stops,
        startvalue = n_stops,
        tellwidth = false,
        tellheight = false,
    )

    # Connect slider value -> fill_step Observable
    on(fill_slider.value) do val
        fill_step[] = val
    end

    time_label_str = lift(fill_step, t_max) do step, tmax
        frac = (step - 1) / (n_stops - 1)
        t_cur = frac * tmax
        pct = round(Int, frac * 100)
        "$(round(t_cur; digits=3)) / $(round(tmax; digits=3)) ($(pct)%)"
    end

    text!(
        overlay_scene,
        lbl_pos;
        text = time_label_str,
        fontsize = 12,
        color = :gray20,
        align = (:left, :center),
        overdraw = true,
    )

    set_overdraw_recursive!(play_btn.blockscene)
    set_overdraw_recursive!(fill_slider.blockscene)

    # Play / Pause animation loop across the 30 stops
    on(play_btn.clicks) do _
        is_playing[] = !is_playing[]
        if is_playing[]
            @async begin
                while is_playing[] && isopen(fig.scene)
                    sleep(0.06)
                    is_playing[] || break
                    cur = fill_slider.value[]
                    if cur < n_stops
                        set_close_to!(fill_slider, cur + 1)
                    else
                        # Pause briefly at 100% fill before restarting from empty
                        sleep(0.35)
                        is_playing[] || break
                        set_close_to!(fill_slider, 1)
                    end
                end
            end
        end
    end

    return fig
end

