function set_overdraw_recursive!(plot::AbstractPlot)
    if haskey(plot.attributes, :overdraw)
        try
            plot.overdraw[] = true
        catch
        end
    end
    for child_plot in plot.plots
        set_overdraw_recursive!(child_plot)
    end
end

function set_overdraw_recursive!(scene::Scene)
    for p in scene.plots
        set_overdraw_recursive!(p)
    end
    for child in scene.children
        set_overdraw_recursive!(child)
    end
end


"""
    align_flow_segments!(pts::Vector{Point3f}, st, dx::Float32; n_iters::Int = 2)

In-place post-processing filter for `flow_segments` (`[p1_start, p1_end, p2_start, p2_end, ...]`).
Smooths tangent directions and laterally aligns upstream/downstream dashes onto coherent curves
while keeping points offset along surface normals without BVH raycasts.
"""
function align_flow_segments!(pts::Vector{Point3f}, st, dx::Float32; n_iters::Int = 2)
    n_segs = length(pts) ÷ 2
    n_segs == 0 && return pts

    centroids = st.centroids
    normals   = st.normals
    flow_dirs = st.flow_dir

    active_tris = Vector{Int}(undef, n_segs)
    k = 0
    @inbounds for i in eachindex(centroids)
        if sum(abs2, flow_dirs[i]) > 1f-6
            k += 1
            k <= n_segs && (active_tris[k] = i)
        end
    end
    k == n_segs || return pts

    cell_size = 1.6f0 * dx
    inv_cell  = 1.0f0 / cell_size
    search_r2 = (1.6f0 * dx)^2

    grid = Dict{NTuple{3, Int}, Vector{Int}}()
    sizehint!(grid, n_segs)

    mids = Vector{Point3f}(undef, n_segs)
    dirs = Vector{Vec3f}(undef, n_segs)
    half_lens = Vector{Float32}(undef, n_segs)

    @inbounds for s in 1:n_segs
        p0 = pts[2s - 1]
        p1 = pts[2s]
        mid = 0.5f0 * (p0 + p1)
        seg = p1 - p0
        len = norm(seg)

        mids[s] = mid
        half_lens[s] = 0.5f0 * len
        dirs[s] = len > 1f-6 ? (seg / len) : Vec3f(1, 0, 0)

        ck = (floor(Int, mid[1] * inv_cell), floor(Int, mid[2] * inv_cell), floor(Int, mid[3] * inv_cell))
        bucket = get!(Vector{Int}, grid, ck)
        push!(bucket, s)
    end

    anchors = copy(mids)
    max_nudge = 0.48f0 * dx

    for _ in 1:n_iters
        new_mids = copy(mids)
        new_dirs = copy(dirs)

        @inbounds for s in 1:n_segs
            tri_i = active_tris[s]
            n = Vec3f(normals[tri_i])
            u = dirs[s]
            p = mids[s]

            t_perp_raw = cross(n, u)
            tp_len = norm(t_perp_raw)
            tp_len > 1f-5 || continue
            t_perp = t_perp_raw / tp_len

            ck = (floor(Int, p[1] * inv_cell), floor(Int, p[2] * inv_cell), floor(Int, p[3] * inv_cell))

            u_accum = 2.0f0 * u
            lat_shift_sum = 0.0f0
            w_lat_sum = 0.0f0

            for dz in -1:1, dy in -1:1, dx_i in -1:1
                bucket = get(grid, (ck[1] + dx_i, ck[2] + dy, ck[3] + dz), nothing)
                bucket === nothing && continue

                for nb in bucket
                    nb == s && continue
                    tri_nb = active_tris[nb]

                    dot(n, Vec3f(normals[tri_nb])) > 0.75f0 || continue

                    delta = mids[nb] - p
                    d2 = sum(abs2, delta)
                    (1f-8 < d2 <= search_r2) || continue

                    u_nb = dirs[nb]
                    cos_u = dot(u, u_nb)
                    cos_u > 0.5f0 || continue

                    d = sqrt(d2)
                    w_dist = 1.0f0 - d / (1.6f0 * dx)

                    u_accum += w_dist * u_nb

                    along = abs(dot(delta, u)) / d
                    w_align = w_dist * (along^2)
                    if w_align > 1f-4
                        lat_offset = dot(delta, t_perp)
                        lat_shift_sum += w_align * lat_offset
                        w_lat_sum += w_align
                    end
                end
            end

            u_tan = u_accum - dot(u_accum, n) * n
            ut_len = norm(u_tan)
            u_smooth = ut_len > 1f-5 ? (u_tan / ut_len) : u
            new_dirs[s] = u_smooth

            if w_lat_sum > 1f-4
                p_candidate = p + (lat_shift_sum / w_lat_sum) * t_perp
                total_offset = p_candidate - anchors[s]
                tan_offset = total_offset - dot(total_offset, n) * n
                off_len = norm(tan_offset)
                if off_len > max_nudge
                    tan_offset *= (max_nudge / off_len)
                end
                new_mids[s] = anchors[s] + tan_offset
            end
        end

        mids .= new_mids
        dirs .= new_dirs
    end

    @inbounds for s in 1:n_segs
        step_vec = half_lens[s] * dirs[s]
        pts[2s - 1] = mids[s] - step_vec
        pts[2s]     = mids[s] + step_vec
    end

    return pts
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

    # --- Geometry & Import State ---
    CAD_EXTS  = Set([".step", ".stp", ".iges", ".igs", ".brep"])
    MESH_EXTS = Set([".stl", ".obj", ".ply", ".off", ".2dm", ".msh"])

    default_step = joinpath(pkgdir(@__MODULE__), "assets", "plate_with_ribs_2.step")
    current_filepath = Observable(default_step)
    is_cad_file      = Observable(true)

    cad_elem_size  = Observable(1.0)
    cad_min_curv   = Observable(1.0)
    cad_circle_pts = Observable(15)

    meshdata = Observable(
        load_step_mesh_as_surface(
            default_step;
            target_element_size = cad_elem_size[],
            min_curvature_size = cad_min_curv[],
            min_circle_points = cad_circle_pts[],
        )
    )

    function first_triangle_centroid(md)
        verts = GeometryBasics.coordinates(md.mesh)
        faces = GeometryBasics.faces(md.mesh)
        isempty(faces) && return Point3f(0, 0, 0)
        f1 = faces[1]
        return Point3f((verts[f1[1]] + verts[f1[2]] + verts[f1[3]]) / 3.0f0)
    end

    # --- Voxelization & Solver Gate State ---
    res = Observable(100)
    voxel_confirmed = Observable(true)

    voxeldata = lift(meshdata, res) do md, r
        VoxelData(md; res = r)
    end

    init_pos = Point3f(first(meshdata[].mesh.position))
    init_r   = Float32(voxeldata[].voxel_spacing[1])
    gates = Observable([ValveGate(position = init_pos, open_time = 0.0f0, radius = init_r)])
    solver_gates = Observable(copy(gates[]))

    flow_line_half_len = Observable(1.0)

    on(meshdata) do md
        c1 = first_triangle_centroid(md)
        r_def = max(Float32(voxeldata[].voxel_spacing[1]), 1.5f0)
        new_gates = [ValveGate(position = c1, open_time = 0.0f0, radius = r_def)]
        dragged_gate[] = 0
        gates[] = new_gates
        solver_gates[] = copy(new_gates)
        center!(lscene.scene)
        flow_line_half_len[] = compute_flow_line_len(md)
    end

    last_stats = Ref(
        project_voxel_to_mesh_triangles(meshdata[], voxeldata[], solve_eikonal_fim_ka(voxeldata[], gates[]))
    )

    stats = lift(meshdata, voxeldata, solver_gates, voxel_confirmed) do md, vd, gts, confirmed
        if !confirmed
            return last_stats[]
        end
        sol = solve_eikonal_fim_ka(vd, gts)
        res_stats = project_voxel_to_mesh_triangles(md, vd, sol)
        last_stats[] = res_stats
        return res_stats
    end

    vertex_itp_f32 = lift(meshdata) do md
        AngleWeightInterpolator(md, Float32)
    end

    fill_time = lift(vertex_itp_f32, stats, meshdata, voxel_confirmed) do itp, st, md, confirmed
        n_verts = length(md.mesh.position)
        n_faces = length(GeometryBasics.faces(md.mesh))
        if !confirmed || length(st.fill_time) != n_faces
            return zeros(Float32, n_verts)
        end
        return itp(zeros(Float32, n_verts), st.fill_time)
    end


    t_max = lift(fill_time) do ft
        m = maximum(ft)
        isfinite(m) && m > 0.0f0 ? m : 1.0f0
    end
    color_limits = @lift((0.0f0, $t_max))

    # --- Colormap & Voxel-Preview Transparent Mesh Mode ---
    n_stops = 30
    fill_step = Observable(n_stops)
    cmap_name = Observable(:turbo)
    unfilled_grey = RGBAf(0.78, 0.78, 0.80, 1.0)
    preview_grey  = RGBAf(0.72, 0.74, 0.78, 0.28)

    base_cmap = lift(cmap_name) do cname
        cgrad(cname, n_stops, categorical = true)
    end

    # Updating a 30-color colormap texture is instantaneous and keeps GPU per-pixel interpolation smooth!
    mesh_cmap = lift(cmap_name, fill_step, voxel_confirmed) do cname, step, confirmed
        if !confirmed
            return cgrad(fill(preview_grey, n_stops), n_stops, categorical = true)
        end
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
        transparency = @lift(!$voxel_confirmed),
        shading = true,
    )



    # --- 3D Voxel Slice Scatter Overlay ---
    slice_axis = Observable(3)
    slice_idx  = Observable(1)

    slice_max = lift(voxeldata, slice_axis) do vd, ax
        max(size(vd.grid, ax), 1)
    end

    on(slice_max) do mx
        slice_idx[] = clamp(round(Int, mx / 2), 1, mx)
    end

    voxel_slice_pts = lift(voxeldata, slice_axis, slice_idx, voxel_confirmed) do vd, ax, idx, confirmed
        pts = Point3f[]
        confirmed && return pts

        grid = vd.grid
        nx, ny, nz = size(grid)
        dx, dy, dz = Float32.(vd.voxel_spacing)
        c0 = Point3f(vd.corner)

        if ax == 1 && 1 <= idx <= nx
            @inbounds for k in 1:nz, j in 1:ny
                if grid[idx, j, k]
                    push!(pts, c0 + Vec3f((idx - 0.5f0) * dx, (j - 0.5f0) * dy, (k - 0.5f0) * dz))
                end
            end
        elseif ax == 2 && 1 <= idx <= ny
            @inbounds for k in 1:nz, i in 1:nx
                if grid[i, idx, k]
                    push!(pts, c0 + Vec3f((i - 0.5f0) * dx, (idx - 0.5f0) * dy, (k - 0.5f0) * dz))
                end
            end
        elseif ax == 3 && 1 <= idx <= nz
            @inbounds for j in 1:ny, i in 1:nx
                if grid[i, j, idx]
                    push!(pts, c0 + Vec3f((i - 0.5f0) * dx, (j - 0.5f0) * dy, (idx - 0.5f0) * dz))
                end
            end
        end
        return pts
    end

    scatter!(
        lscene,
        voxel_slice_pts;
        color = RGBf(0.05, 0.05, 0.05),
        markersize = 6.5,
        strokewidth = 0,
        overdraw = true,
        visible = @lift(!$voxel_confirmed),
    )

    ####################        Camera & Gates ############################
    center!(lscene.scene)
    cam = cameracontrols(lscene.scene)
    cam.settings.center[] = false
    cam.settings.projectiontype[] = Makie.Orthographic
    cam.settings.cad[] = true
    cam.settings.zoom_shift_lookat[] = true
    cam.controls.left_key[] = false
    cam.controls.right_key[] = false
    if hasproperty(cam.settings, :near_clip)
        cam.settings.near_clip[] = 1e-5
    end
    cam.near[] = -500.0f0
    cam.far[]  = 5000.0f0

    on(cam.eyeposition) do _
        if cam.near[] > -500.0f0
            cam.near[] = -500.0f0
            cam.far[]  = 5000.0f0
        end
    end

    for prop in (:up_key, :down_key, :left_key, :right_key, :forward_key, :backward_key)
        if hasproperty(cam.settings, prop)
            getproperty(cam.settings, prop)[] = Keyboard.unknown
        end
    end
    update_cam!(lscene.scene, cam)

    hovered_gate = Observable(0)
    dragged_gate = Ref(0)

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
                    f[1] + v_offset, f[2] + v_offset, f[3] + v_offset
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
        visible = voxel_confirmed,
    )

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

    function pick_surface_point(mp)
        vp = lscene.scene.viewport[]
        u = 2.0f0 * (Float32(mp[1]) - Float32(vp.origin[1])) / Float32(vp.widths[1]) - 1.0f0
        v = 2.0f0 * (Float32(mp[2]) - Float32(vp.origin[2])) / Float32(vp.widths[2]) - 1.0f0
        (-1.0f0 <= u <= 1.0f0 && -1.0f0 <= v <= 1.0f0) || return nothing

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

    # =========================================================================
    # Gate Drag, Hover & Deferred Solver Trigger (Mouse Events)
    # =========================================================================
    gate_moved = Ref(false)

    # --- Mouse Button: Start Drag on Press, Commit to Solver on Release ---
    on(events(fig).mousebutton; priority = 10) do event
        voxel_confirmed[] || return Consume(false)
        if event.button == Mouse.left
            if event.action == Mouse.press
                mp = events(fig).mouseposition[]
                g_idx = find_near_gate(mp)
                if g_idx > 0
                    dragged_gate[] = g_idx
                    hovered_gate[] = g_idx
                    gate_moved[] = false
                    return Consume(true)
                end
            elseif event.action == Mouse.release
                if dragged_gate[] > 0
                    dragged_gate[] = 0
                    hovered_gate[] = find_near_gate(events(fig).mouseposition[])
                    if gate_moved[]
                        gate_moved[] = false
                        solver_gates[] = copy(gates[]) # Run Eikonal solver once on release
                    end
                    return Consume(true)
                end
            end
        end
        return Consume(false)
    end

    # --- Mouse Position: Live Visual Drag & Hover Highlight (No Solver) ---
    on(events(fig).mouseposition; priority = 10) do mp
        voxel_confirmed[] || return Consume(false)
        g_drag = dragged_gate[]
        if g_drag > 0 && ispressed(fig, Mouse.left)
            new_pos = pick_surface_point(mp)
            if new_pos !== nothing && 1 <= g_drag <= length(gates[])
                gts = copy(gates[])
                old_g = gts[g_drag]
                if sum(abs2, Point3f(old_g.position) - new_pos) > 1f-6
                    gts[g_drag] = ValveGate(position = new_pos, open_time = old_g.open_time, radius = old_g.radius)
                    gates[] = gts # Updates visual gate mesh only
                    gate_moved[] = true
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
    # =========================================================================

    textbox_focused = Ref(false)

    on(events(lscene.scene).keyboardbutton; priority = 20) do event
        (voxel_confirmed[] && !textbox_focused[]) || return Consume(false)
        event.action == Keyboard.press || return Consume(false)

        if event.key == Keyboard.a
            mp = events(fig).mouseposition[]
            new_pos = pick_surface_point(mp)
            if new_pos !== nothing
                gts = copy(gates[])
                default_r = isempty(gts) ? Float32(voxeldata[].voxel_spacing[1]) : gts[end].radius
                push!(gts, ValveGate(position = new_pos, open_time = 0.0f0, radius = default_r))
                dragged_gate[] = 0
                gates[] = gts
                solver_gates[] = copy(gts)
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
                    solver_gates[] = copy(new_gts)
                    hovered_gate[] = find_near_gate(mp)
                end
            end
            return Consume(true)
        end
        return Consume(false)
    end

    # =========================================================================
    # 3. Pixel-Space Overlay Scene
    # =========================================================================
    overlay_scene = Scene(fig.scene; clear = false, camera = campixel!)

    # -------------------------------------------------------------------------
    # PANEL 1 (Upper-Left): Geometry Import & CAD Remeshing
    # -------------------------------------------------------------------------
    left_panel_w = 230
    geom_panel_h = @lift($is_cad_file ? 185 : 82)

    geom_card_rect = lift(lscene.scene.viewport, geom_panel_h) do vp, h
        x0 = vp.origin[1] + 14
        y0 = vp.origin[2] + vp.widths[2] - h - 14
        Rect2f(x0, y0, left_panel_w, h)
    end

    poly!(
        overlay_scene,
        geom_card_rect,
        color = RGBAf(1.0, 1.0, 1.0, 0.90),
        strokecolor = RGBAf(0.5, 0.5, 0.5, 0.4),
        strokewidth = 1.0,
        overdraw = true,
    )

    text!(
        overlay_scene,
        @lift(Point2f($geom_card_rect.origin[1] + 12, $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 16));
        text = "Geometry",
        fontsize = 13,
        font = :bold,
        color = :gray15,
        align = (:left, :center),
        overdraw = true,
    )

    btn_load = Button(
        overlay_scene;
        bbox = @lift(BBox(
            $geom_card_rect.origin[1] + 12,
            $geom_card_rect.origin[1] + left_panel_w - 12,
            $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 54,
            $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 30
        )),
        label = "Load Geometry...",
        fontsize = 12,
        tellwidth = false,
        tellheight = false,
    )
    set_overdraw_recursive!(btn_load.blockscene)

    text!(
        overlay_scene,
        @lift(Point2f($geom_card_rect.origin[1] + 12, $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 67));
        text = @lift(basename($current_filepath)),
        fontsize = 11,
        color = :gray35,
        align = (:left, :center),
        overdraw = true,
    )

    text!(
        overlay_scene,
        @lift(Point2f($geom_card_rect.origin[1] + 12, $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 92));
        text = "Element Size:",
        fontsize = 11,
        color = :gray20,
        align = (:left, :center),
        visible = is_cad_file,
        overdraw = true,
    )

    tb_elem_size = Textbox(
        overlay_scene;
        bbox = @lift(BBox(
            $geom_card_rect.origin[1] + 135,
            $geom_card_rect.origin[1] + left_panel_w - 12,
            $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 103,
            $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 81
        )),
        stored_string = string(cad_elem_size[]),
        placeholder = "2.0",
        fontsize = 11,
        tellwidth = false,
        tellheight = false,
    )
    set_overdraw_recursive!(tb_elem_size.blockscene)

    text!(
        overlay_scene,
        @lift(Point2f($geom_card_rect.origin[1] + 12, $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 118));
        text = "Min Curvature:",
        fontsize = 11,
        color = :gray20,
        align = (:left, :center),
        visible = is_cad_file,
        overdraw = true,
    )

    tb_min_curv = Textbox(
        overlay_scene;
        bbox = @lift(BBox(
            $geom_card_rect.origin[1] + 135,
            $geom_card_rect.origin[1] + left_panel_w - 12,
            $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 129,
            $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 107
        )),
        stored_string = string(cad_min_curv[]),
        placeholder = "1.0",
        fontsize = 11,
        tellwidth = false,
        tellheight = false,
    )
    set_overdraw_recursive!(tb_min_curv.blockscene)

    btn_remesh = Button(
        overlay_scene;
        bbox = @lift(BBox(
            $geom_card_rect.origin[1] + 12,
            $geom_card_rect.origin[1] + left_panel_w - 12,
            $geom_card_rect.origin[2] + 12,
            $geom_card_rect.origin[2] + 38
        )),
        label = "Remesh CAD",
        fontsize = 12,
        tellwidth = false,
        tellheight = false,
    )
    set_overdraw_recursive!(btn_remesh.blockscene)

    function sync_cad_widgets!(show_cad::Bool)
        for tb in (tb_elem_size, tb_min_curv)
            for p in tb.blockscene.plots
                p.visible[] = show_cad
            end
        end
        for p in btn_remesh.blockscene.plots
            p.visible[] = show_cad
        end
    end
    on(sync_cad_widgets!, is_cad_file)

    function run_cad_remesh!()
        is_cad_file[] || return
        es = tryparse(Float64, tb_elem_size.stored_string[])
        mc = tryparse(Float64, tb_min_curv.stored_string[])
        es !== nothing && es > 0 && (cad_elem_size[] = es)
        mc !== nothing && mc > 0 && (cad_min_curv[] = mc)

        voxel_confirmed[] = false
        meshdata[] = load_step_mesh_as_surface(
            current_filepath[];
            target_element_size = cad_elem_size[],
            min_curvature_size = cad_min_curv[],
            min_circle_points = cad_circle_pts[],
        )
    end

    on(btn_remesh.clicks) do _
        run_cad_remesh!()
    end

    on(btn_load.clicks) do _
        filepath = pick_file()
        isempty(filepath) && return
        isfile(filepath) || return

        ext = lowercase(splitext(filepath)[2])
        if ext in CAD_EXTS
            current_filepath[] = filepath
            is_cad_file[] = true
            voxel_confirmed[] = false
            meshdata[] = load_step_mesh_as_surface(
                filepath;
                target_element_size = cad_elem_size[],
                min_curvature_size = cad_min_curv[],
                min_circle_points = cad_circle_pts[],
            )
        elseif ext in MESH_EXTS
            current_filepath[] = filepath
            is_cad_file[] = false
            voxel_confirmed[] = false
            meshdata[] = MeshData(filepath)
        else
            return
        end
    end

    # -------------------------------------------------------------------------
    # PANEL 2 (Upper-Left, below Geometry): Voxelization & Slice Preview
    # -------------------------------------------------------------------------
    vox_panel_h = 166
    vox_card_rect = lift(lscene.scene.viewport, geom_card_rect) do vp, g_rect
        x0 = g_rect.origin[1]
        y0 = g_rect.origin[2] - vox_panel_h - 12
        Rect2f(x0, y0, left_panel_w, vox_panel_h)
    end

    poly!(
        overlay_scene,
        vox_card_rect,
        color = RGBAf(1.0, 1.0, 1.0, 0.90),
        strokecolor = RGBAf(0.5, 0.5, 0.5, 0.4),
        strokewidth = 1.0,
        overdraw = true,
    )

    text!(
        overlay_scene,
        @lift(Point2f($vox_card_rect.origin[1] + 12, $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 16));
        text = "Voxelization",
        fontsize = 13,
        font = :bold,
        color = :gray15,
        align = (:left, :center),
        overdraw = true,
    )

    text!(
        overlay_scene,
        @lift(Point2f($vox_card_rect.origin[1] + 12, $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 42));
        text = "Max Voxels:",
        fontsize = 11,
        color = :gray20,
        align = (:left, :center),
        overdraw = true,
    )

    tb_res = Textbox(
        overlay_scene;
        bbox = @lift(BBox(
            $vox_card_rect.origin[1] + 95,
            $vox_card_rect.origin[1] + left_panel_w - 12,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 54,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 30
        )),
        stored_string = string(res[]),
        placeholder = "100",
        fontsize = 11,
        tellwidth = false,
        tellheight = false,
    )
    set_overdraw_recursive!(tb_res.blockscene)

    for tb in (tb_elem_size, tb_min_curv, tb_res)
        on(tb.focused) do f
            textbox_focused[] = f
        end
    end

    on(tb_res.stored_string) do s
        val = tryparse(Int, strip(s))
        if val !== nothing && 10 <= val <= 1000
            voxel_confirmed[] = false
            if res[] != val
                res[] = val
            end
        end
    end

    btn_w3 = (left_panel_w - 24 - 12) / 3
    btn_x = Button(
        overlay_scene;
        bbox = @lift(BBox(
            $vox_card_rect.origin[1] + 12,
            $vox_card_rect.origin[1] + 12 + btn_w3,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 88,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 64
        )),
        label = @lift($slice_axis == 1 ? "[X]" : "X"),
        fontsize = 11,
        tellwidth = false,
        tellheight = false,
    )
    btn_y = Button(
        overlay_scene;
        bbox = @lift(BBox(
            $vox_card_rect.origin[1] + 18 + btn_w3,
            $vox_card_rect.origin[1] + 18 + 2 * btn_w3,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 88,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 64
        )),
        label = @lift($slice_axis == 2 ? "[Y]" : "Y"),
        fontsize = 11,
        tellwidth = false,
        tellheight = false,
    )
    btn_z = Button(
        overlay_scene;
        bbox = @lift(BBox(
            $vox_card_rect.origin[1] + 24 + 2 * btn_w3,
            $vox_card_rect.origin[1] + left_panel_w - 12,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 88,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 64
        )),
        label = @lift($slice_axis == 3 ? "[Z]" : "Z"),
        fontsize = 11,
        tellwidth = false,
        tellheight = false,
    )
    for b in (btn_x, btn_y, btn_z)
        set_overdraw_recursive!(b.blockscene)
    end

    on(btn_x.clicks) do _
        voxel_confirmed[] = false
        slice_axis[] = 1
    end
    on(btn_y.clicks) do _
        voxel_confirmed[] = false
        slice_axis[] = 2
    end
    on(btn_z.clicks) do _
        voxel_confirmed[] = false
        slice_axis[] = 3
    end

    slice_slider = Slider(
        overlay_scene;
        bbox = @lift(BBox(
            $vox_card_rect.origin[1] + 12,
            $vox_card_rect.origin[1] + left_panel_w - 64,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 118,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 96
        )),
        range = @lift(1:$slice_max),
        startvalue = slice_idx[],
        tellwidth = false,
        tellheight = false,
    )
    set_overdraw_recursive!(slice_slider.blockscene)

    on(slice_slider.value) do val
        slice_idx[] = val
    end
    on(slice_idx) do val
        if slice_slider.value[] != val
            set_close_to!(slice_slider, val)
        end
    end

    text!(
        overlay_scene,
        @lift(Point2f($vox_card_rect.origin[1] + left_panel_w - 58, $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 107));
        text = @lift("$( $slice_idx )/$( $slice_max )"),
        fontsize = 10,
        color = :gray25,
        align = (:left, :center),
        overdraw = true,
    )

    btn_vox_ok = Button(
        overlay_scene;
        bbox = @lift(BBox(
            $vox_card_rect.origin[1] + 12,
            $vox_card_rect.origin[1] + left_panel_w - 12,
            $vox_card_rect.origin[2] + 12,
            $vox_card_rect.origin[2] + 38
        )),
        label = @lift($voxel_confirmed ? "Confirmed ✓" : "OK (Run Solver)"),
        fontsize = 12,
        tellwidth = false,
        tellheight = false,
    )
    set_overdraw_recursive!(btn_vox_ok.blockscene)

    on(btn_vox_ok.clicks) do _
        val = tryparse(Int, strip(tb_res.stored_string[]))
        if val !== nothing && 10 <= val <= 1000 && res[] != val
            res[] = val
        end
        voxel_confirmed[] = true
    end

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
                        sleep(0.35)
                        is_playing[] || break
                        set_close_to!(fill_slider, 1)
                    end
                end
            end
        end
    end

    ########################   OVERLAYS     ##############################################
    show_flow_dirs  = Observable(false)
    show_weld_lines = Observable(false)
    show_air_traps  = Observable(false)

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

    flow_segments = lift(stats, voxeldata, show_flow_dirs, flow_line_half_len) do st, vd, visible, half_len
        pts = Point3f[]
        visible || return pts

        dx = Float32(vd.voxel_spacing[1])
        normal_offset = 0.08f0 * dx
        # half_len      = 0.35f0 * dx

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

        align_flow_segments!(pts, st, dx; n_iters = 2)
        return pts
    end

    linesegments!(
        lscene,
        flow_segments;
        color = RGBAf(0.08, 0.08, 0.08, 0.85),
        linewidth = 1.2,
        visible = show_flow_dirs,
    )

    weld_segments = lift(stats, voxeldata, show_weld_lines) do st, vd, visible
        pts = Point3f[]
        visible || return pts

        dx = Float32(vd.voxel_spacing[1])
        normal_offset = 0.12f0 * dx
        half_len      = 0.55f0 * dx

        centroids = st.centroids
        normals   = st.normals
        flow_dirs = st.flow_dir
        weld_mask = st.masks.weld_lines

        @inbounds for i in eachindex(centroids)
            weld_mask[i] || continue

            n = normals[i]
            u = flow_dirs[i]

            t_seam = cross(n, u)
            t_norm2 = sum(abs2, t_seam)

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


function compute_flow_line_len(md::MeshData)

    pts, tris = md.mesh.position, md.mesh.faces

    len = 0.0
    for tri in tris
        i1, i2, i3 = tri
        e12 = pts[i1] - pts[i2]
        e13 = pts[i1] - pts[i3]
        e23 = pts[i2] - pts[i3]
        l12 = dot(e12, e12)
        l13 = dot(e13, e13)
        l23 = dot(e23, e23)
        lmax = max(l12, max(l13, l23))
        len += sqrt(lmax)
    end
    return len / length(tris)
end