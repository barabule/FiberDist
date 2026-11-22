
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

    

    # Replace the `gates = lift(t2)` block with a directly mutable Observable:
    p1 = Point3f(0, 25, 0)
    p2 = Point3f(100, 25, 0)
    t2 = Observable(0.0)

    gates = Observable([
        ValveGate(position = p1, open_time = 0.0, radius = 2.0),
        ValveGate(position = p2, open_time = t2[], radius = 2.0)
    ])

    on(t2) do t
        gts = copy(gates[])
        if length(gts) >= 2
            g = gts[2]
            gts[2] = ValveGate(position = g.position, open_time = t, radius = g.radius)
            gates[] = gts
        end
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

####################        Camera ############################
    # Orthographic CAD camera setup with zero near-plane clipping

    center!(lscene.scene)
    cam = cameracontrols(lscene.scene)
    cam.settings.projectiontype[] = Makie.Orthographic
    cam.settings.cad[] = true
    cam.settings.zoom_shift_lookat[] = true
    cam.controls.left_key[] = false
    cam.controls.right_key[] = false

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

#######################    Gates     ######################################

    # 3. Build transparent yellow gate cylinders oriented along the nearest triangle normal
    # Track hovered / dragged gate index (0 = none) for visual highlight & interaction
    # Disable WASD keyboard camera panning so 'a' and 'd' can add/delete gates
    if hasproperty(cam.settings, :keyboard_translations)
        cam.settings.keyboard_translations[] = false
    elseif hasproperty(cam.settings, :key_controls)
        cam.settings.key_controls[] = false
    end
    update_cam!(lscene.scene, cam)

    # Track hovered / dragged gate index (0 = none)
    hovered_gate = Observable(0)
    dragged_gate = Ref(0)

    # 3. Build gate cylinders with per-vertex colors so ONLY the active gate highlights
    gate_meshes = lift(gates, stats, voxeldata, hovered_gate) do gts, st, vd, h_idx
        dx = Float32(vd.voxel_spacing[1])
        centroids = st.centroids
        normals   = st.normals

        col_normal = RGBAf(1.0, 0.85, 0.0, 0.45)
        col_active = RGBAf(1.0, 0.40, 0.0, 0.80)

        all_pts   = Point3f[]
        all_faces = GeometryBasics.TriangleFace{Int}[]
        all_cols  = RGBAf[]

        for (idx, g) in enumerate(gts)
            pos = Point3f(g.position)
            r   = max(Float32(g.radius), dx)
            h   = max(3.0f0 * r, 3.0f0 * dx)

            nearest_idx = argmin([sum(abs2, c - pos) for c in centroids])
            n_dir = Vec3f(normals[nearest_idx])
            n_len = sqrt(sum(abs2, n_dir))
            n_hat = n_len > 1f-6 ? (n_dir / n_len) : Vec3f(0.0f0, 0.0f0, 1.0f0)

            p_base = pos - 0.25f0 * dx * n_hat
            p_tip  = pos + h * n_hat

            cyl = GeometryBasics.Cylinder(p_base, p_tip, r)
            m   = GeometryBasics.triangle_mesh(GeometryBasics.Tesselation(cyl, 32))

            v_offset = length(all_pts)
            m_pts    = GeometryBasics.coordinates(m)
            m_faces  = GeometryBasics.faces(m)
            c_gate   = (idx == h_idx) ? col_active : col_normal

            append!(all_pts, m_pts)
            append!(all_cols, fill(c_gate, length(m_pts)))
            for f in m_faces
                push!(all_faces, GeometryBasics.TriangleFace{Int}(
                    f[1] + v_offset,
                    f[2] + v_offset,
                    f[3] + v_offset
                ))
            end
        end

        if isempty(all_pts)
            m_empty = GeometryBasics.triangle_mesh(GeometryBasics.Tesselation(
                GeometryBasics.Cylinder(Point3f(0), Point3f(0, 0, 1f-4), 1f-4), 8
            ))
            return GeometryBasics.normal_mesh(GeometryBasics.mesh(
                GeometryBasics.coordinates(m_empty),
                GeometryBasics.faces(m_empty);
                color = fill(RGBAf(0, 0, 0, 0), length(GeometryBasics.coordinates(m_empty)))
            ))
        end

        return GeometryBasics.normal_mesh(
            GeometryBasics.mesh(all_pts, all_faces; color = all_cols)
        )
    end

    gate_plot = mesh!(
        lscene,
        gate_meshes;
        transparency = true,
        shading = true,
    )

    # Helper: find nearest gate index within 1 gate radius (or min 18 px) of screen cursor `mp`
    function find_near_gate(mp)
        gts = gates[]
        isempty(gts) && return 0

        dx = Float32(voxeldata[].voxel_spacing[1])
        vp_origin = Point2f(lscene.scene.viewport[].origin)
        mouse_px = Point2f(mp[1], mp[2])
        mouse_local = mouse_px - vp_origin

        plt, _ = Makie.pick(lscene.scene, mouse_px)
        hit_gate_mesh = (plt == gate_plot)

        best_idx = 0
        best_dist = Inf32

        cam_view = cameracontrols(lscene.scene)
        right_vec = Vec3f(normalize(cross(cam_view.lookat[] - cam_view.eyeposition[], cam_view.upvector[])))

        for (idx, g) in enumerate(gts)
            pos = Point3f(g.position)
            r   = max(Float32(g.radius), dx)

            p_scr = Point2f(Makie.project(lscene.scene, pos))
            p_rad_scr = Point2f(Makie.project(lscene.scene, pos + r * right_vec))

            r_px = max(norm(p_rad_scr - p_scr), 18.0f0)
            d_px = norm(mouse_local - p_scr)

            thresh = hit_gate_mesh ? max(r_px * 2.5f0, 45.0f0) : r_px
            if d_px <= thresh && d_px < best_dist
                best_dist = d_px
                best_idx = idx
            end
        end
        return best_idx
    end

    # Helper: raycast through gate cylinder onto the part surface mesh via Makie.pick
    # Exact orthographic/perspective raycast from screen pixel -> Raycore BVH hit point
    function pick_surface_point(mp)
        vp = lscene.scene.viewport[]
        # Convert window pixel coordinates to normalized device coordinates [-1, 1]
        u = 2.0f0 * (Float32(mp[1]) - Float32(vp.origin[1])) / Float32(vp.widths[1]) - 1.0f0
        v = 2.0f0 * (Float32(mp[2]) - Float32(vp.origin[2])) / Float32(vp.widths[2]) - 1.0f0
        (-1.0f0 <= u <= 1.0f0 && -1.0f0 <= v <= 1.0f0) || return nothing

        # Unproject near (z = -1) and far (z = +1) points through inverse projectionview
        inv_pv = inv(lscene.scene.camera.projectionview[])
        p_near_h = inv_pv * Vec4f(u, v, -1.0f0, 1.0f0)
        p_far_h  = inv_pv * Vec4f(u, v,  1.0f0, 1.0f0)

        p_near = Point3f(p_near_h[1], p_near_h[2], p_near_h[3]) / p_near_h[4]
        p_far  = Point3f(p_far_h[1],  p_far_h[2],  p_far_h[3])  / p_far_h[4]

        ray_vec = p_far - p_near
        ray_len = norm(ray_vec)
        ray_len > 1f-6 || return nothing

        ray = Raycore.Ray(o = p_near, d = SVector{3, Float32}(ray_vec / ray_len))
        hit_found, _, distance, _, _ = Raycore.closest_hit(meshdata[].tlas, ray)
        hit_found || return nothing

        return p_near + Float32(distance) * (ray_vec / ray_len)
    end

    # 4. Mouse Press / Release (priority = 10 intercepts left-drag before Camera3D orbits)
    on(events(fig).mousebutton; priority = 10) do event
        if event.button == Mouse.left
            if event.action == Mouse.press
                mp = events(fig).mouseposition[]
                g_idx = find_near_gate(mp)
                if g_idx > 0
                    dragged_gate[] = g_idx
                    hovered_gate[] = g_idx
                    return Consume(true)
                end
            elseif event.action == Mouse.release
                if dragged_gate[] > 0
                    dragged_gate[] = 0
                    hovered_gate[] = find_near_gate(events(fig).mouseposition[])
                    return Consume(true)
                end
            end
        end
        return Consume(false)
    end

    # 5. Mouse Move (priority = 10 prevents Camera3D rotation while dragging a gate)
    on(events(fig).mouseposition; priority = 10) do mp
        g_drag = dragged_gate[]
        if g_drag > 0 && ispressed(fig, Mouse.left)
            new_pos = pick_surface_point(mp)
            if new_pos !== nothing && 1 <= g_drag <= length(gates[])
                gts = copy(gates[])
                old_g = gts[g_drag]
                if sum(abs2, Point3f(old_g.position) - new_pos) > 1f-6
                    gts[g_drag] = ValveGate(
                        position = new_pos,
                        open_time = old_g.open_time,
                        radius = old_g.radius
                    )
                    gates[] = gts
                end
            end
            return Consume(true)
        else
            new_hover = find_near_gate(mp)
            if hovered_gate[] != new_hover
                hovered_gate[] = new_hover
            end
            return Consume(false)
        end
    end


    
    # update_cam!(lscene.scene, cam)

    # 6. Keyboard Shortcuts on lscene.scene (priority = 20 intercepts before Camera3D)
    on(events(lscene.scene).keyboardbutton; priority = 20) do event
        event.action == Keyboard.press || return Consume(false)

        if event.key == Keyboard.a
            mp = events(fig).mouseposition[]
            new_pos = pick_surface_point(mp)
            if new_pos !== nothing
                gts = copy(gates[])
                default_r = isempty(gts) ? Float32(voxeldata[].voxel_spacing[1]) : gts[end].radius
                push!(gts, ValveGate(position = new_pos, open_time = 0.0f0, radius = default_r))
                gates[] = gts
                hovered_gate[] = length(gts)
            end
            return Consume(true)
        elseif event.key == Keyboard.d
            gts = gates[]
            if length(gts) > 1
                mp = events(fig).mouseposition[]
                target_idx = hovered_gate[] > 0 ? hovered_gate[] : find_near_gate(mp)
                if 1 <= target_idx <= length(gts)
                    new_gts = copy(gts)
                    deleteat!(new_gts, target_idx)
                    dragged_gate[] = 0
                    gates[] = new_gts
                    hovered_gate[] = find_near_gate(mp)
                end
            end
            return Consume(true)
        end
        return Consume(false)
    end
    # on(events(fig).keyboardbutton; priority = 10) do event
    #     event.action == Keyboard.press || return Consume(false)

    #     if event.key == Keyboard.a
    #         mp = events(fig).mouseposition[]
    #         new_pos = pick_surface_point(mp)
    #         if new_pos !== nothing
    #             gts = copy(gates[])
    #             default_r = isempty(gts) ? Float32(voxeldata[].voxel_spacing[1]) : gts[end].radius
    #             push!(gts, ValveGate(position = new_pos, open_time = 0.0f0, radius = default_r))
    #             gates[] = gts
    #             hovered_gate[] = length(gts)
    #         end
    #         return Consume(true)
    #     elseif event.key == Keyboard.d
    #         gts = gates[]
    #         if length(gts) > 1
    #             mp = events(fig).mouseposition[]
    #             target_idx = hovered_gate[] > 0 ? hovered_gate[] : find_near_gate(mp)
    #             if 1 <= target_idx <= length(gts)
    #                 new_gts = copy(gts)
    #                 deleteat!(new_gts, target_idx)
    #                 dragged_gate[] = 0
    #                 gates[] = new_gts
    #                 hovered_gate[] = find_near_gate(mp)
    #             end
    #         end
    #         return Consume(true)
    #     end
    #     return Consume(false)
    # end

    

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

    ########################   OVERLAYS     ##############################################
# --- C. Floating Overlays Panel (below Colorbar on the right) ---
    show_flow_dirs  = Observable(false)
    show_weld_lines = Observable(true)
    show_air_traps  = Observable(true)

    panel_w = card_w + 26
    panel_h = 150

    panel_card_rect = lift(lscene.scene.viewport, cb_card_rect) do vp, cb_rect
        x0 = vp.origin[1] + vp.widths[1] - panel_w - 14
        y0 = cb_rect.origin[2] - panel_h - 12
        Rect2f(x0, y0, panel_w, panel_h)
    end

    poly!(
        overlay_scene,
        panel_card_rect,
        color = RGBAf(1.0, 1.0, 1.0, 0.88),
        strokecolor = RGBAf(0.5, 0.5, 0.5, 0.4),
        strokewidth = 1.0,
        overdraw = true,
    )

    panel_title_pos = @lift(Point2f(
        $panel_card_rect.origin[1] + 12,
        $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - 18
    ))

    text!(
        overlay_scene,
        panel_title_pos;
        text = "Overlays",
        fontsize = 13,
        font = :bold,
        color = :gray15,
        align = (:left, :center),
        overdraw = true,
    )

    # 1) Flow Direction Checkbox
    chk_flow_bbox = @lift(BBox(
        $panel_card_rect.origin[1] + 12,
        $panel_card_rect.origin[1] + 30,
        $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - 52,
        $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - 34
    ))

    chk_flow_lbl_pos = @lift(Point2f(
        $panel_card_rect.origin[1] + 36,
        $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - 43
    ))

    chk_flow = Checkbox(
        overlay_scene;
        bbox = chk_flow_bbox,
        checked = show_flow_dirs[],
        tellwidth = false,
        tellheight = false,
    )
    on(chk_flow.checked) do val
        show_flow_dirs[] = val
    end
    set_overdraw_recursive!(chk_flow.blockscene)

    text!(
        overlay_scene,
        chk_flow_lbl_pos;
        text = "Flow Direction",
        fontsize = 12,
        color = :gray20,
        align = (:left, :center),
        overdraw = true,
    )

    # 2) Weld Lines Checkbox
    chk_weld_bbox = @lift(BBox(
        $panel_card_rect.origin[1] + 12,
        $panel_card_rect.origin[1] + 30,
        $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - 80,
        $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - 62
    ))

    chk_weld_lbl_pos = @lift(Point2f(
        $panel_card_rect.origin[1] + 36,
        $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - 71
    ))

    chk_weld = Checkbox(
        overlay_scene;
        bbox = chk_weld_bbox,
        checked = show_weld_lines[],
        tellwidth = false,
        tellheight = false,
    )
    on(chk_weld.checked) do val
        show_weld_lines[] = val
    end
    set_overdraw_recursive!(chk_weld.blockscene)

    text!(
        overlay_scene,
        chk_weld_lbl_pos;
        text = "Weld Lines",
        fontsize = 12,
        color = :gray20,
        align = (:left, :center),
        overdraw = true,
    )

    # 3) Air Traps Checkbox
    chk_air_bbox = @lift(BBox(
        $panel_card_rect.origin[1] + 12,
        $panel_card_rect.origin[1] + 30,
        $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - 108,
        $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - 90
    ))

    chk_air_lbl_pos = @lift(Point2f(
        $panel_card_rect.origin[1] + 36,
        $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - 99
    ))

    chk_air = Checkbox(
        overlay_scene;
        bbox = chk_air_bbox,
        checked = show_air_traps[],
        tellwidth = false,
        tellheight = false,
    )
    on(chk_air.checked) do val
        show_air_traps[] = val
    end
    set_overdraw_recursive!(chk_air.blockscene)

    text!(
        overlay_scene,
        chk_air_lbl_pos;
        text = "Air Traps",
        fontsize = 12,
        color = :gray20,
        align = (:left, :center),
        overdraw = true,
    )

    # --- D. 3D Flow Direction Line Segments ---
    flow_segments = lift(stats, voxeldata, show_flow_dirs) do st, vd, visible
        pts = Point3f[]
        visible || return pts

        dx = Float32(vd.voxel_spacing[1])
        normal_offset = 0.08f0 * dx
        half_len      = 0.35f0 * dx

        centroids = st.centroids
        normals   = st.normals
        flow_dirs = st.flow_dir

        sizehint!(pts, 2 * length(centroids))
        @inbounds for i in eachindex(centroids)
            u = flow_dirs[i]
            u_norm2 = sum(abs2, u)
            u_norm2 > 1f-6 || continue

            n = normals[i]
            p_mid = Point3f(centroids[i] + normal_offset * n)
            step_vec = Vec3f((half_len / sqrt(u_norm2)) * u)

            push!(pts, p_mid - step_vec)
            push!(pts, p_mid + step_vec)
        end
        return pts
    end

    linesegments!(
        lscene,
        flow_segments;
        color = RGBAf(0.08, 0.08, 0.08, 0.85),
        linewidth = 1.2,
        visible = show_flow_dirs,
    )




    # --- E. 3D Weld Line Segments (oriented along seam tangent n × u) ---
    weld_segments = lift(stats, voxeldata, show_weld_lines) do st, vd, visible
        pts = Point3f[]
        visible || return pts

        dx = Float32(vd.voxel_spacing[1])
        normal_offset = 0.12f0 * dx  # slightly above flow lines to avoid z-fighting
        half_len      = 0.55f0 * dx

        centroids = st.centroids
        normals   = st.normals
        flow_dirs = st.flow_dir
        weld_mask = st.masks.weld_lines

        @inbounds for i in eachindex(centroids)
            weld_mask[i] || continue

            n = normals[i]
            u = flow_dirs[i]

            # Weld seam runs tangent to the surface and perpendicular to the colliding flow
            t_seam = cross(n, u)
            t_norm2 = sum(abs2, t_seam)

            # Fallback if u is parallel to n: pick any tangent perpendicular to n
            if t_norm2 <= 1f-6
                ref = abs(n[1]) < 0.9f0 ? SVector{3, Float32}(1f0, 0f0, 0f0) : SVector{3, Float32}(0f0, 1f0, 0f0)
                t_seam = cross(n, ref)
                t_norm2 = sum(abs2, t_seam)
            end
            t_norm2 > 1f-6 || continue

            p_mid = Point3f(centroids[i] + normal_offset * n)
            step_vec = Vec3f((half_len / sqrt(t_norm2)) * t_seam)

            push!(pts, p_mid - step_vec)
            push!(pts, p_mid + step_vec)
        end
        return pts
    end

    linesegments!(
        lscene,
        weld_segments;
        color = RGBAf(1.0, 0.88, 0.0, 0.95),
        linewidth = 3.5,
        visible = show_weld_lines,
    )


    # --- F. 3D Air Trap Spheres (semi-transparent red markers on air-trap triangles) ---
    air_trap_pts = lift(stats, voxeldata, show_air_traps) do st, vd, visible
        pts = Point3f[]
        visible || return pts

        dx = Float32(vd.voxel_spacing[1])
        normal_offset = 0.15f0 * dx

        centroids = st.centroids
        normals   = st.normals
        air_mask  = st.masks.air_traps

        @inbounds for i in eachindex(centroids)
            air_mask[i] || continue
            push!(pts, Point3f(centroids[i] + normal_offset * normals[i]))
        end
        return pts
    end

    air_trap_radius = @lift(0.65f0 * Float32($voxeldata.voxel_spacing[1]))

    meshscatter!(
        lscene,
        air_trap_pts;
        marker = GeometryBasics.Sphere(Point3f(0), 1.0f0),
        markersize = air_trap_radius,
        color = RGBAf(0.95, 0.12, 0.12, 0.55),
        transparency = true,
        shading = true,
        visible = show_air_traps,
    )




    return fig
end

