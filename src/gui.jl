# =============================================================================
# 1. Central Application State
# =============================================================================

Base.@kwdef mutable struct AppState
    # Scenes & Main Plots
    fig::Figure                                             # Main application window
    lscene::LScene                                          # 3D viewport scene
    overlay_scene::Scene                                    # 2D HUD scene
    triad_scene::Scene                                      # Orientation triad scene
    mesh_plot::Any = nothing                                # Surface mesh plot
    gate_plot::Any = nothing                                # Valve gate plot

    # Geometry & CAD Import State
    current_filepath::Observable{String}                    # Loaded file path
    is_cad_file::Observable{Bool}                           # CAD format flag
    cad_elem_size::Observable{Float64}                      # Target element size
    cad_min_curv::Observable{Float64}                       # Min curvature size
    cad_circle_pts::Observable{Int}                         # Circle point count
    meshdata::Observable{MeshData}                          # Active surface mesh

    # Voxelization & Solver State
    res::Observable{Int}                                    # Max axis voxels
    voxel_confirmed::Observable{Bool}                       # Solver confirmation flag
    voxeldata::Observable{VoxelData}                        # Voxelized part grid
    gates::Observable{Vector{ValveGate}}                    # Interactive valve gates
    solver_gates::Observable{Vector{ValveGate}}             # Solver-ready gates
    last_stats::Base.RefValue{Any}                          # Cached projection stats
    stats::Observable{Any}                                  # Triangle projection stats
    vertex_itp_f32::Observable{Any}                         # Face-to-vertex interpolator
    fill_time::Observable{Vector{Float32}}                  # Vertex fill time
    t_max::Observable{Float32}                              # Maximum fill time
    color_limits::Observable{Tuple{Float32, Float32}}       # Colormap time range

    # Visualization & Animation State
    n_stops::Int                                            # Timeline animation stops
    fill_step::Observable{Int}                              # Current animation step
    cmap_name::Observable{Symbol}                           # Active colormap name
    base_cmap::Observable{Any}                              # Full color gradient
    mesh_cmap::Observable{Any}                              # Masked fill colormap
    is_playing::Observable{Bool} = Observable(false)        # Animation playback flag

    # Slice Preview State
    slice_axis::Observable{Int}                             # Preview slice axis
    slice_idx::Observable{Int}                              # Active slice index
    slice_max::Observable{Int}                              # Max slice index

    # Diagnostic Overlays State
    show_flow_dirs::Observable{Bool} = Observable(false)    # Flow direction toggle
    show_weld_lines::Observable{Bool} = Observable(false)   # Weld lines toggle
    show_air_traps::Observable{Bool} = Observable(false)    # Air traps toggle
    show_wireframe::Observable{Bool} = Observable(false)    # Mesh wireframe toggle
    flow_line_half_len::Observable{Float64}                 # Flow segment half-length

    # Interactive Input State
    hovered_gate::Observable{Int} = Observable(0)           # Hovered gate index
    dragged_gate::Base.RefValue{Int} = Ref(0)               # Dragged gate index
    resizing_gate::Base.RefValue{Bool} = Ref(false)         # Gate resize active
    resize_start_y::Base.RefValue{Float32} = Ref(0.0f0)     # Drag start Y
    resize_start_r::Base.RefValue{Float32} = Ref(1.0f0)     # Initial gate radius
    gate_moved::Base.RefValue{Bool} = Ref(false)            # Gate changed flag
    textbox_focused::Base.RefValue{Bool} = Ref(false)       # Textbox focus flag
end

const CAD_EXTS  = Set([".step", ".stp", ".iges", ".igs", ".brep"])
const MESH_EXTS = Set([".stl", ".obj", ".ply", ".off", ".2dm", ".msh"])


# =============================================================================
# 2. Pure Helpers & Geometry Utilities
# =============================================================================

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

function set_visible_recursive!(plot::AbstractPlot, vis::Bool)
    if haskey(plot.attributes, :visible)
        try
            plot.visible[] = vis
        catch
        end
    end
    for child_plot in plot.plots
        set_visible_recursive!(child_plot, vis)
    end
end

function set_visible_recursive!(scene::Scene, vis::Bool)
    for p in scene.plots
        set_visible_recursive!(p, vis)
    end
    for child in scene.children
        set_visible_recursive!(child, vis)
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

function compute_flow_line_len(md::MeshData)
    pts, tris = md.mesh.position, md.mesh.faces
    isempty(tris) && return 1.0

    len = 0.0
    for tri in tris
        i1, i2, i3 = tri
        e12 = pts[i1] - pts[i2]
        e13 = pts[i1] - pts[i3]
        e23 = pts[i2] - pts[i3]
        lmax = max(dot(e12, e12), max(dot(e13, e13), dot(e23, e23)))
        len += sqrt(lmax)
    end
    return len / length(tris)
end

function first_triangle_centroid(md::MeshData)
    verts = md.mesh.position
    faces = md.mesh.faces
    isempty(faces) && return Point3f(0, 0, 0)
    f1 = first(faces)
    return Point3f((verts[f1[1]] + verts[f1[2]] + verts[f1[3]]) / 3.0f0)
end

function pick_surface_point(state::AppState, mp)
    vp = state.lscene.scene.viewport[]
    u = 2.0f0 * (Float32(mp[1]) - Float32(vp.origin[1])) / Float32(vp.widths[1]) - 1.0f0
    v = 2.0f0 * (Float32(mp[2]) - Float32(vp.origin[2])) / Float32(vp.widths[2]) - 1.0f0
    (-1.0f0 <= u <= 1.0f0 && -1.0f0 <= v <= 1.0f0) || return nothing

    inv_pv = inv(state.lscene.scene.camera.projectionview[])
    p_near_h = inv_pv * Vec4f(u, v, -1.0f0, 1.0f0)
    p_far_h  = inv_pv * Vec4f(u, v,  1.0f0, 1.0f0)

    p_near = Point3f(p_near_h[1], p_near_h[2], p_near_h[3]) / p_near_h[4]
    p_far  = Point3f(p_far_h[1],  p_far_h[2],  p_far_h[3])  / p_far_h[4]

    ray_vec = p_far - p_near
    ray_len = norm(ray_vec)
    ray_len > 1f-6 || return nothing

    ray = Raycore.Ray(o = p_near, d = SVector{3, Float32}(ray_vec / ray_len))
    hit_found, _, distance, _, _ = Raycore.closest_hit(state.meshdata[].tlas, ray)
    hit_found || return nothing

    return p_near + Float32(distance) * (ray_vec / ray_len)
end

function find_near_gate(state::AppState, mp)
    gts = state.gates[]
    isempty(gts) && return 0

    dx = Float32(state.voxeldata[].voxel_spacing[1])
    vp_origin = Point2f(state.lscene.scene.viewport[].origin)
    mouse_px = Point2f(mp[1], mp[2])
    mouse_local = mouse_px - vp_origin

    plt, _ = Makie.pick(state.lscene.scene, mouse_px)
    hit_gate_mesh = (plt == state.gate_plot)

    best_idx = 0
    best_dist = Inf32
    cam_view = cameracontrols(state.lscene.scene)
    right_vec = Vec3f(normalize(cross(cam_view.lookat[] - cam_view.eyeposition[], cam_view.upvector[])))

    for (idx, g) in enumerate(gts)
        pos = Point3f(g.position)
        r   = max(Float32(g.radius), dx)
        p_scr = Point2f(Makie.project(state.lscene.scene, pos))
        p_rad_scr = Point2f(Makie.project(state.lscene.scene, pos + r * right_vec))
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


"""
    with_computed_gate_delays(gates::AbstractVector{ValveGate}, vd::VoxelData;
                              r_ref::Float32 = 2.0f0,
                              power::Float32 = 2.2f0,
                              alpha::Float32 = 0.035f0) -> Vector{ValveGate}

Returns a copy of `gates` with `open_time` populated from each gate's `radius`
and normalized so the earliest gate opens at `t = 0.0f0`.
"""
function with_computed_gate_delays(
    gates::AbstractVector{ValveGate},
    vd::VoxelData;
    r_ref::Float32 = 2.0f0,
    power::Float32 = 2.2f0,
    alpha::Float32 = 0.035f0
    )::Vector{ValveGate}

    isempty(gates) && return ValveGate[]

    nx, ny, nz = size(vd.grid)
    dx, dy, dz = Float32.(vd.voxel_spacing)
    L_char = max(nx * dx, ny * dy, nz * dz)

    dx_mean = (dx + dy + dz) / 3.0f0
    base_h2 = 4.0f0 * dx_mean * dx_mean

    sum_F = 0.0
    count_F = 0
    @inbounds for idx in eachindex(vd.grid)
        if vd.grid[idx]
            rsq = Float32(vd.thickness[idx])
            if rsq > 0.0f0
                sum_F += Float64(base_h2 * rsq)
                count_F += 1
            end
        end
    end

    F_char = count_F > 0 ? Float32(sum_F / count_F) : base_h2
    t_ref = L_char / max(F_char, 1f-6)

    raw_times = Vector{Float32}(undef, length(gates))
    @inbounds for (i, g) in enumerate(gates)
        r = clamp(Float32(g.radius), GATE_RADIUS_MIN, GATE_RADIUS_MAX)
        ratio = max(r_ref / r, 1.0f0)
        raw_times[i] = alpha * t_ref * (ratio^power - 1.0f0)
    end

    t0 = minimum(raw_times)
    return [
        ValveGate(
            position  = g.position,
            radius    = g.radius,
            open_time = raw_times[i] - t0
        )
        for (i, g) in enumerate(gates)
    ]
end

# =============================================================================
# 3. State Initialization & Reactive Data Pipeline
# =============================================================================

function init_app_state()::AppState
    fig = Figure(size = (1400, 850))
    rowsize!(fig.layout, 1, Relative(1.0))
    rowgap!(fig.layout, 0)

    lscene = LScene(
        fig[1, 1];
        show_axis = false,
        scenekw = (
            backgroundcolor = RGBf(0.96, 0.96, 0.97),
            clear = true,
        )
    )
    overlay_scene = Scene(fig.scene; clear = false, camera = campixel!)
    triad_vp = lift(lscene.scene.viewport) do vp
        s = clamp(round(Int, 0.13 * vp.widths[2]), 70, 220)
        Rect2i(round(Int, vp.origin[1]) + 14, round(Int, vp.origin[2]) + 14, s, s)
    end
    triad_scene = Scene(fig.scene; clear = false, viewport = triad_vp)

    default_step = joinpath(pkgdir(@__MODULE__), "assets", "plate_with_ribs_2.step")
    current_filepath = Observable(default_step)
    is_cad_file      = Observable(true)

    cad_elem_size  = Observable(1.0)
    cad_min_curv   = Observable(1.0)
    cad_circle_pts = Observable(15)

    meshdata = Observable{MeshData}(
        load_step_mesh_as_surface(
            default_step;
            target_element_size = cad_elem_size[],
            min_curvature_size = cad_min_curv[],
            min_circle_points = cad_circle_pts[],
        )
    )

    res = Observable(100)
    voxel_confirmed = Observable(true)

    voxeldata = map!(Observable{VoxelData}(), meshdata, res) do md, r
        VoxelData(md; res = r)
    end

    init_pos = Point3f(first(meshdata[].mesh.position))
    init_r   = Float32(voxeldata[].voxel_spacing[1])
    gates = Observable([ValveGate(position = init_pos, open_time = 0.0f0, radius = init_r)])
    solver_gates = Observable(with_computed_gate_delays(gates[], voxeldata[]))

    last_stats = Ref{Any}(
        project_voxel_to_mesh_triangles(meshdata[], voxeldata[], solve_eikonal_fim_ka(voxeldata[], solver_gates[]))
    )

    stats = map!(Observable{Any}(), meshdata, voxeldata, solver_gates, voxel_confirmed) do md, vd, gts, confirmed
        confirmed || return last_stats[]
        sol = solve_eikonal_fim_ka(vd, gts)
        res_stats = project_voxel_to_mesh_triangles(md, vd, sol)
        last_stats[] = res_stats
        return res_stats
    end

    vertex_itp_f32 = map!(md -> AngleWeightInterpolator(md, Float32), Observable{Any}(), meshdata)

    fill_time = lift(vertex_itp_f32, stats, meshdata, voxel_confirmed) do itp, st, md, confirmed
        n_verts = length(md.mesh.position)
        n_faces = length(GeometryBasics.faces(md.mesh))
        if !confirmed || length(st.fill_time) != n_faces
            return zeros(Float32, n_verts)
        end
        ft = itp(zeros(Float32, n_verts), st.fill_time)
        @inbounds for i in eachindex(ft)
            isfinite(ft[i]) || (ft[i] = 0.0f0)
        end
        return ft
    end

    # Pre-resize `fill_time.val` BEFORE `mesh_plot` (priority = 0) reacts to `meshdata`
    on(meshdata; priority = 100) do md
        n_verts = length(md.mesh.position)
        if length(fill_time[]) != n_verts
            fill_time.val = zeros(Float32, n_verts)
        end
    end

    t_max = lift(fill_time) do ft
        m = maximum(ft)
        isfinite(m) && m > 0.0f0 ? m : 1.0f0
    end
    color_limits = @lift((0.0f0, $t_max))

    n_stops = 30
    fill_step = Observable(n_stops)
    cmap_name = Observable(:turbo)
    unfilled_grey = RGBAf(0.78, 0.78, 0.80, 1.0)
    preview_grey  = RGBAf(0.72, 0.74, 0.78, 0.28)

    base_cmap = map!(cname -> cgrad(cname, n_stops, categorical = true), Observable{Any}(), cmap_name)

    mesh_cmap = map!(Observable{Any}(), cmap_name, fill_step, voxel_confirmed) do cname, step, confirmed
        if !confirmed
            return cgrad(fill(preview_grey, n_stops), n_stops, categorical = true)
        end
        full_colors = cgrad(cname, n_stops, categorical = true).colors
        n_filled = clamp(round(Int, (step - 1) / (n_stops - 1) * n_stops), 0, n_stops)
        masked_colors = [i <= n_filled ? full_colors[i] : unfilled_grey for i in 1:n_stops]
        cgrad(masked_colors, n_stops, categorical = true)
    end

    slice_axis = Observable(3)
    slice_idx  = Observable(1)
    slice_max  = lift((vd, ax) -> max(size(vd.grid, ax), 1), voxeldata, slice_axis)

    on(slice_max) do mx
        slice_idx[] = clamp(round(Int, mx / 2), 1, mx)
    end

    state = AppState(
        fig                = fig,
        lscene             = lscene,
        overlay_scene      = overlay_scene,
        triad_scene        = triad_scene,
        current_filepath   = current_filepath,
        is_cad_file        = is_cad_file,
        cad_elem_size      = cad_elem_size,
        cad_min_curv       = cad_min_curv,
        cad_circle_pts     = cad_circle_pts,
        meshdata           = meshdata,
        res                = res,
        voxel_confirmed    = voxel_confirmed,
        voxeldata          = voxeldata,
        gates              = gates,
        solver_gates       = solver_gates,
        last_stats         = last_stats,
        stats              = stats,
        vertex_itp_f32     = vertex_itp_f32,
        fill_time          = fill_time,
        t_max              = t_max,
        color_limits       = color_limits,
        n_stops            = n_stops,
        fill_step          = fill_step,
        cmap_name          = cmap_name,
        base_cmap          = base_cmap,
        mesh_cmap          = mesh_cmap,
        slice_axis         = slice_axis,
        slice_idx          = slice_idx,
        slice_max          = slice_max,
        flow_line_half_len = Observable(compute_flow_line_len(meshdata[])),
    )

    on(state.meshdata) do md
        c1 = first_triangle_centroid(md)
        r_def = max(Float32(state.voxeldata[].voxel_spacing[1]), 1.5f0)
        new_gates = [ValveGate(position = c1, open_time = 0.0f0, radius = r_def)]
        state.dragged_gate[] = 0
        state.gates[] = new_gates
        state.solver_gates[] = with_computed_gate_delays(new_gates, state.voxeldata[])
        fit_camera_to_mesh!(state.lscene, md)
        state.flow_line_half_len[] = compute_flow_line_len(md)
    end

    on(state.res) do _
        updated = with_computed_gate_delays(state.gates[], state.voxeldata[])
        state.gates[] = updated
        state.solver_gates[] = updated
    end

    return state
end


# =============================================================================
# 4. 3D Scene, Camera & Overlays
# =============================================================================

"""
    fit_camera_to_mesh!(lscene::LScene, md::MeshData; view_dir = Vec3f(1.0f0, -1.2f0, 0.85f0))

Computes the exact AABB of `md.mesh`, centers `cam.lookat[]` on the geometry's bounding-box
center (even if imported far from `(0, 0, 0)`), and frames the part in an isometric view.
"""
function fit_camera_to_mesh!(
    lscene::LScene,
    md::MeshData;
    view_dir::Vec3f = Vec3f(1.0f0, -1.2f0, 0.85f0)
)
    verts = md.mesh.position
    isempty(verts) && return

    pmin = Point3f(Inf32, Inf32, Inf32)
    pmax = Point3f(-Inf32, -Inf32, -Inf32)
    @inbounds for v in verts
        p = Point3f(v)
        pmin = min.(pmin, p)
        pmax = max.(pmax, p)
    end

    bbox_center = 0.5f0 * (pmin + pmax)
    diag = norm(pmax - pmin)
    radius = max(0.5f0 * diag, 1.0f0)

    cam = cameracontrols(lscene.scene)
    dir = normalize(view_dir)
    dist = 2.2f0 * radius
    eye = bbox_center + dir * dist

    # Temporarily allow centering, update lookat/eye/up, and adjust orthographic zoom scale
    was_centered = cam.settings.center[]
    cam.settings.center[] = true
    update_cam!(lscene.scene, cam, eye, bbox_center, Vec3f(0.0f0, 0.0f0, 1.0f0))
    cam.settings.center[] = was_centered

    # Ensure orthographic projection scale fits the entire bounding sphere
    if hasproperty(cam, :ortho_zoom)
        cam.ortho_zoom[] = 1.15f0 * radius
    end

    # Scale near/far clip planes relative to part size so off-origin parts never clip
    cam.near[] = -10.0f0 * radius
    cam.far[]  =  10.0f0 * radius
    update_cam!(lscene.scene, cam)
end

function configure_cad_camera!(state::AppState)
    lscene = state.lscene
    cam = cameracontrols(lscene.scene)

    cam.settings.projectiontype[] = Makie.Orthographic
    cam.settings.cad[] = true
    cam.settings.zoom_shift_lookat[] = true
    cam.controls.left_key[] = false
    cam.controls.right_key[] = false
    cam.controls.up_key[] = false
    cam.controls.down_key[] = false


    # Keep rotation centered on `cam.lookat[]` (the geometry center or clicked pivot)
    if hasproperty(cam.settings, :rotation_center)
        cam.settings.rotation_center[] = :lookat
    end
    # Prevent unwanted camera roll (keeps Z-up turntable behavior like SolidWorks/Fusion)
    if hasproperty(cam.settings, :fixed_axis)
        cam.settings.fixed_axis[] = true
    end
    if hasproperty(cam.settings, :near_clip)
        cam.settings.near_clip[] = 1e-5
    end

    for prop in (:up_key, :down_key, :left_key, :right_key, :forward_key, :backward_key)
        if hasproperty(cam.settings, prop)
            getproperty(cam.settings, prop)[] = Keyboard.unknown
        end
    end

    # Frame initial geometry around its true bounding-box center
    fit_camera_to_mesh!(lscene, state.meshdata[])

    # Keep near/far planes wide enough during pan/zoom
    on(cam.eyeposition) do _
        verts = state.meshdata[].mesh.position
        isempty(verts) && return
        bb = GeometryBasics.Rect3(verts)
        r = max(0.5f0 * norm(bb.widths), 1.0f0)
        if cam.near[] > -5.0f0 * r
            cam.near[] = -10.0f0 * r
            cam.far[]  =  10.0f0 * r
        end
    end
end

function build_gate_meshes(state::AppState)
    return lift(state.gates, state.stats, state.voxeldata, state.hovered_gate) do gts, st, vd, h_idx
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
            r = clamp(Float32(g.radius), GATE_RADIUS_MIN, GATE_RADIUS_MAX)
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
end

function build_voxel_slice_points(state::AppState)
    return lift(state.voxeldata, state.slice_axis, state.slice_idx, state.voxel_confirmed) do vd, ax, idx, confirmed
        pts = Point3f[]
        confirmed && return pts

        grid = vd.grid
        nx, ny, nz = size(grid)
        dx, dy, dz = Float32.(vd.voxel_spacing)
        c0 = Point3f(vd.corner)

        if ax == 1 && 1 <= idx <= nx
            @inbounds for k in 1:nz, j in 1:ny
                grid[idx, j, k] && push!(pts, c0 + Vec3f((idx - 0.5f0) * dx, (j - 0.5f0) * dy, (k - 0.5f0) * dz))
            end
        elseif ax == 2 && 1 <= idx <= ny
            @inbounds for k in 1:nz, i in 1:nx
                grid[i, idx, k] && push!(pts, c0 + Vec3f((i - 0.5f0) * dx, (idx - 0.5f0) * dy, (k - 0.5f0) * dz))
            end
        elseif ax == 3 && 1 <= idx <= nz
            @inbounds for j in 1:ny, i in 1:nx
                grid[i, j, idx] && push!(pts, c0 + Vec3f((i - 0.5f0) * dx, (j - 0.5f0) * dy, (idx - 0.5f0) * dz))
            end
        end
        return pts
    end
end

function setup_3d_scene!(state::AppState)
    lscene = state.lscene

    plain_mesh = lift(state.meshdata) do md
        m = md.mesh
        coords = Point3f[Point3f(p) for p in GeometryBasics.coordinates(m)]
        faces  = [
            GeometryBasics.TriangleFace(f[1], f[2], f[3])
            for f in GeometryBasics.faces(m)
        ]
        return GeometryBasics.mesh(coords, faces)
    end

    state.mesh_plot = mesh!(
        lscene,
        plain_mesh;
        color = state.fill_time,
        colorrange = state.color_limits,
        colormap = state.mesh_cmap,
        transparency = @lift(!$(state.voxel_confirmed)),
        shading = true,
    )

    wireframe!(
        lscene,
        plain_mesh;
        color = RGBAf(0.12, 0.12, 0.15, 0.55),
        linewidth = 0.8,
        transparency = true,
        depth_shift = -1f-4,
        inspectable = false,
        visible = state.show_wireframe,
    )

    scatter!(
        lscene,
        build_voxel_slice_points(state);
        color = RGBf(0.05, 0.05, 0.05),
        markersize = 6.5,
        strokewidth = 0,
        overdraw = true,
        visible = @lift(!$(state.voxel_confirmed)),
    )

    configure_cad_camera!(state)

    state.gate_plot = mesh!(
        lscene,
        build_gate_meshes(state);
        transparency = true,
        shading = true,
        visible = state.voxel_confirmed,
    )

    setup_diagnostic_overlays_3d!(state)
    setup_orientation_triad!(state)
end

"""
    build_triad_mesh()

Constructs a colored orientation triad mesh of 3 cylinders and 3 cones
(X = red, Y = green, Z = blue) centered at the origin.
"""
function build_triad_mesh()
    cyl_r = 0.055f0
    cyl_len = 0.72f0
    cone_r = 0.14f0
    cone_len = 0.32f0

    axes_spec = (
        (Vec3f(1.0f0, 0.0f0, 0.0f0), RGBAf(0.88, 0.18, 0.18, 1.0)),
        (Vec3f(0.0f0, 1.0f0, 0.0f0), RGBAf(0.18, 0.72, 0.22, 1.0)),
        (Vec3f(0.0f0, 0.0f0, 1.0f0), RGBAf(0.18, 0.42, 0.90, 1.0)),
    )

    all_pts   = Point3f[]
    all_faces = GeometryBasics.TriangleFace{Int}[]
    all_cols  = RGBAf[]

    for (dir, col) in axes_spec
        p0 = Point3f(-cyl_r * dir)
        p1 = Point3f(cyl_len * dir)
        p2 = Point3f((cyl_len + cone_len) * dir)

        for prim in (
            GeometryBasics.Cylinder(p0, p1, cyl_r),
            GeometryBasics.Cone(p1, p2, cone_r),
        )
            m = GeometryBasics.triangle_mesh(GeometryBasics.Tesselation(prim, 24))
            v_offset = length(all_pts)
            m_pts = GeometryBasics.coordinates(m)
            append!(all_pts, m_pts)
            append!(all_cols, fill(col, length(m_pts)))
            for f in GeometryBasics.faces(m)
                push!(all_faces, GeometryBasics.TriangleFace{Int}(
                    f[1] + v_offset, f[2] + v_offset, f[3] + v_offset
                ))
            end
        end
    end

    return GeometryBasics.normal_mesh(
        GeometryBasics.mesh(all_pts, all_faces; color = all_cols)
    )
end

function setup_orientation_triad!(state::AppState)
    mesh!(
        state.triad_scene,
        build_triad_mesh();
        shading = true,
        depth_shift = -0.97f0,
        inspectable = false,
    )

    cam = cameracontrols(state.lscene.scene)
    triad_cam = state.triad_scene.camera
    proj = Makie.orthographicprojection(Float32, -1.35f0, 1.35f0, -1.35f0, 1.35f0, 0.0f0, 100.0f0)

    function sync_triad_camera!()
        view_vec = Vec3f(cam.eyeposition[] - cam.lookat[])
        v_len = norm(view_vec)
        dir = v_len > 1f-6 ? (view_vec / v_len) : Vec3f(0.0f0, 0.0f0, 1.0f0)
        up_raw = Vec3f(cam.upvector[])
        u_len = norm(up_raw)
        up = u_len > 1f-6 ? (up_raw / u_len) : Vec3f(0.0f0, 1.0f0, 0.0f0)

        eye = 50.0f0 * dir
        view_mat = Makie.lookat(eye, Vec3f(0.0f0, 0.0f0, 0.0f0), up)
        triad_cam.view[] = view_mat
        triad_cam.projection[] = proj
        triad_cam.projectionview[] = proj * view_mat
        triad_cam.eyeposition[] = eye
        triad_cam.upvector[] = up
        triad_cam.view_direction[] = -dir
    end

    onany(cam.eyeposition, cam.lookat, cam.upvector; update = true) do _, _, _
        sync_triad_camera!()
    end
end

function setup_diagnostic_overlays_3d!(state::AppState)
    lscene = state.lscene

    flow_segments = lift(state.stats, state.voxeldata, state.show_flow_dirs, state.flow_line_half_len) do st, vd, visible, half_len
        pts = Point3f[]
        visible || return pts

        dx = Float32(vd.voxel_spacing[1])
        normal_offset = 0.08f0 * dx
        centroids = st.centroids
        normals   = st.normals
        flow_dirs = st.flow_dir

        sizehint!(pts, 2 * length(centroids))
        @inbounds for i in eachindex(centroids)
            u = flow_dirs[i]
            u_norm2 = sum(abs2, u)
            u_norm2 > 1f-6 || continue

            p_mid = Point3f(centroids[i] + normal_offset * normals[i])
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
        visible = state.show_flow_dirs,
    )

    weld_segments = lift(state.stats, state.voxeldata, state.show_weld_lines) do st, vd, visible
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
        visible = state.show_weld_lines,
    )

    air_trap_pts = lift(state.stats, state.voxeldata, state.show_air_traps) do st, vd, visible
        pts = Point3f[]
        visible || return pts

        dx = Float32(vd.voxel_spacing[1])
        normal_offset = 0.15f0 * dx
        centroids = st.centroids
        normals   = st.normals
        air_mask  = st.masks.air_traps

        @inbounds for i in eachindex(centroids)
            air_mask[i] && push!(pts, Point3f(centroids[i] + normal_offset * normals[i]))
        end
        return pts
    end

    air_trap_radius = @lift(0.65f0 * Float32($(state.voxeldata).voxel_spacing[1]))

    meshscatter!(
        lscene,
        air_trap_pts;
        marker = GeometryBasics.Sphere(Point3f(0), 1.0f0),
        markersize = air_trap_radius,
        color = RGBAf(0.95, 0.12, 0.12, 0.55),
        transparency = true,
        shading = true,
        visible = state.show_air_traps,
    )
end


# =============================================================================
# 5. Mouse & Keyboard Interaction Handlers
# =============================================================================
const GATE_RADIUS_MIN = 0.4f0 # 0.8 mm diameter (SFRP pinpoint minimum)
const GATE_RADIUS_MAX = 6.0f0 # 12.0 mm diameter (large structural valve gate)

function is_shift_pressed(fig::Figure)::Bool
    return ispressed(fig, Keyboard.left_shift) || ispressed(fig, Keyboard.right_shift)
end

function register_interaction_handlers!(state::AppState)
    fig = state.fig

    # Gate selection, drag/resize start, and solver trigger on release
    on(events(fig).mousebutton; priority = 10) do event
        state.voxel_confirmed[] || return Consume(false)
        if event.button == Mouse.left
            if event.action == Mouse.press
                mp = events(fig).mouseposition[]
                shift_down = is_shift_pressed(fig)

                # If Shift is held, allow grabbing either the hovered gate or nearest gate under cursor
                g_idx = find_near_gate(state, mp)
                if g_idx == 0 && shift_down && state.hovered_gate[] > 0
                    g_idx = state.hovered_gate[]
                end

                if 1 <= g_idx <= length(state.gates[])
                    state.dragged_gate[] = g_idx
                    state.hovered_gate[] = g_idx
                    state.gate_moved[] = false

                    if shift_down
                        state.resizing_gate[] = true
                        state.resize_start_y[] = Float32(mp[2])
                        state.resize_start_r[] = Float32(state.gates[][g_idx].radius)
                    else
                        state.resizing_gate[] = false
                    end
                    return Consume(true)
                end
            elseif event.action == Mouse.release
                if state.dragged_gate[] > 0
                    state.dragged_gate[] = 0
                    state.resizing_gate[] = false
                    state.hovered_gate[] = find_near_gate(state, events(fig).mouseposition[])
                    if state.gate_moved[]
                        state.gate_moved[] = false
                        updated = with_computed_gate_delays(state.gates[], state.voxeldata[])
                        state.gates[] = updated
                        state.solver_gates[] = updated
                    end
                    return Consume(true)
                end
            end
        end
        return Consume(false)
    end

    # Live visual gate move OR Shift-drag radius scaling + hover highlight
    on(events(fig).mouseposition; priority = 10) do mp
        state.voxel_confirmed[] || return Consume(false)
        g_drag = state.dragged_gate[]

        if g_drag > 0 && ispressed(fig, Mouse.left) && 1 <= g_drag <= length(state.gates[])
            gts = copy(state.gates[])
            old_g = gts[g_drag]

            if state.resizing_gate[]
                # Dragging up increases radius, dragging down decreases (~0.025 mm per pixel)
                dy_px = Float32(mp[2]) - state.resize_start_y[]
                raw_r = state.resize_start_r[] + 0.025f0 * dy_px
                new_r = clamp(raw_r, GATE_RADIUS_MIN, GATE_RADIUS_MAX)

                if abs(Float32(old_g.radius) - new_r) > 1f-4
                    gts[g_drag] = ValveGate(position = old_g.position, open_time = old_g.open_time, radius = new_r)
                    state.gates[] = gts
                    state.gate_moved[] = true
                end
            else
                new_pos = pick_surface_point(state, mp)
                if new_pos !== nothing && sum(abs2, Point3f(old_g.position) - new_pos) > 1f-6
                    gts[g_drag] = ValveGate(position = new_pos, open_time = old_g.open_time, radius = old_g.radius)
                    state.gates[] = gts
                    state.gate_moved[] = true
                end
            end
            return Consume(true)
        else
            new_hover = find_near_gate(state, mp)
            if state.hovered_gate[] != new_hover
                state.hovered_gate[] = new_hover
            end
            return Consume(false)
        end
    end


# Set orbit pivot to clicked CAD surface point on Right/Middle press (CAD style)
    on(events(fig).mousebutton; priority = 15) do event
        if event.action == Mouse.press && event.button == Mouse.middle
            mp = events(fig).mouseposition[]
            hit_pt = pick_surface_point(state, mp)
            if hit_pt !== nothing
                cam = cameracontrols(state.lscene.scene)
                offset = cam.eyeposition[] - cam.lookat[]
                cam.lookat[] = hit_pt
                cam.eyeposition[] = hit_pt + offset
                update_cam!(state.lscene.scene, cam)
            end
        end
        return Consume(false) # Do NOT consume so Makie's orbit drag still starts!
    end


    # Keyboard shortcuts: 'a' to add gate under cursor, 'd' to delete hovered gate
    on(events(state.lscene.scene).keyboardbutton; priority = 20) do event
        (state.voxel_confirmed[] && !state.textbox_focused[]) || return Consume(false)
        event.action == Keyboard.press || return Consume(false)

        if event.key == Keyboard.a
            mp = events(fig).mouseposition[]
            new_pos = pick_surface_point(state, mp)
            if new_pos !== nothing
                gts = copy(state.gates[])
                default_r = isempty(gts) ? Float32(state.voxeldata[].voxel_spacing[1]) : gts[end].radius
                push!(gts, ValveGate(position = new_pos, open_time = 0.0f0, radius = default_r))
                state.dragged_gate[] = 0
                updated = with_computed_gate_delays(gts, state.voxeldata[])
                state.gates[] = updated
                state.solver_gates[] = updated
                state.hovered_gate[] = length(gts)
            end
            return Consume(true)
        elseif event.key == Keyboard.d
            gts = state.gates[]
            if length(gts) > 1
                mp = events(fig).mouseposition[]
                target_idx = state.hovered_gate[] > 0 ? state.hovered_gate[] : find_near_gate(state, mp)
                if 1 <= target_idx <= length(gts)
                    new_gts = copy(gts)
                    deleteat!(new_gts, target_idx)
                    state.dragged_gate[] = 0
                    updated = with_computed_gate_delays(new_gts, state.voxeldata[])
                    state.gates[] = updated
                    state.solver_gates[] = updated
                    state.hovered_gate[] = find_near_gate(state, mp)
                end
            end
            return Consume(true)
        elseif event.key == Keyboard.f
            fit_camera_to_mesh!(state.lscene, state.meshdata[])
            return Consume(true)
        end
        return Consume(false)
    end
end
# =============================================================================
# 6. Modular HUD / Overlay Panels
# =============================================================================

function draw_panel_card!(overlay_scene::Scene, rect::Observable{Rect2f}; alpha::Float64 = 0.90)
    poly!(
        overlay_scene,
        rect,
        color = RGBAf(1.0, 1.0, 1.0, alpha),
        strokecolor = RGBAf(0.5, 0.5, 0.5, 0.4),
        strokewidth = 1.0,
        overdraw = true,
    )
end

function track_textbox_focus!(state::AppState, tbs...)
    for tb in tbs
        on(tb.focused) do f
            state.textbox_focused[] = f
        end
    end
end

function set_block_visible!(block, vis::Bool)
    if hasproperty(block, :elements)
        for el in values(block.elements)
            set_visible_recursive!(el, vis)
        end
    end
    if hasproperty(block, :blockscene) && block.blockscene !== nothing
        set_visible_recursive!(block.blockscene, vis)
    end
end


function build_geometry_panel!(state::AppState, left_panel_w::Int)
    sc = state.overlay_scene
    geom_panel_h = @lift($(state.is_cad_file) ? 202 : 82)

    geom_card_rect = lift(state.lscene.scene.viewport, geom_panel_h) do vp, h
        x0 = vp.origin[1] + 14
        y0 = vp.origin[2] + vp.widths[2] - h - 14
        Rect2f(x0, y0, left_panel_w, h)
    end
    draw_panel_card!(sc, geom_card_rect; alpha = 0.90)

    text!(
        sc,
        @lift(Point2f($geom_card_rect.origin[1] + 12, $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 16));
        text = "Geometry", fontsize = 13, font = :bold, color = :gray15,
        align = (:left, :center), overdraw = true,
    )

    btn_load = Button(
        sc;
        bbox = @lift(BBox(
            $geom_card_rect.origin[1] + 12, $geom_card_rect.origin[1] + left_panel_w - 12,
            $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 54,
            $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 30
        )),
        label = "Load Geometry...", fontsize = 12, tellwidth = false, tellheight = false,
    )
    set_overdraw_recursive!(btn_load.blockscene)

    geom_label = lift(state.current_filepath, state.meshdata) do fp, md
        n_v = length(GeometryBasics.coordinates(md.mesh))
        n_f = length(GeometryBasics.faces(md.mesh))
        "$(basename(fp)) (V:$(n_v), F:$(n_f))"
    end

    text!(
        sc,
        @lift(Point2f($geom_card_rect.origin[1] + 12, $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 67));
        text = geom_label, fontsize = 10, color = :gray35,
        align = (:left, :center), overdraw = true,
    )

    offscreen_bbox = BBox(-2000.0, -1999.0, -2000.0, -1999.0)

    # Row 1: Element Size
    text!(
        sc,
        @lift(Point2f($geom_card_rect.origin[1] + 12, $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 95));
        text = "Element Size:", fontsize = 11, color = :gray20,
        align = (:left, :center), visible = state.is_cad_file, overdraw = true,
    )

    tb_elem_size = Textbox(
        sc;
        bbox = lift(geom_card_rect, state.is_cad_file) do r, show_cad
            show_cad || return offscreen_bbox
            BBox(
                r.origin[1] + 105, r.origin[1] + left_panel_w - 12,
                r.origin[2] + r.widths[2] - 108,
                r.origin[2] + r.widths[2] - 82
            )
        end,
        stored_string = string(state.cad_elem_size[]),
        displayed_string = string(state.cad_elem_size[]),
        placeholder = "",
        textpadding = (6, 6, 4, 4),
        fontsize = 11, tellwidth = false, tellheight = false,
    )
    set_overdraw_recursive!(tb_elem_size.blockscene)

    # Row 2: Min Curvature
    text!(
        sc,
        @lift(Point2f($geom_card_rect.origin[1] + 12, $geom_card_rect.origin[2] + $geom_card_rect.widths[2] - 129));
        text = "Min Curvature:", fontsize = 11, color = :gray20,
        align = (:left, :center), visible = state.is_cad_file, overdraw = true,
    )

    tb_min_curv = Textbox(
        sc;
        bbox = lift(geom_card_rect, state.is_cad_file) do r, show_cad
            show_cad || return offscreen_bbox
            BBox(
                r.origin[1] + 105, r.origin[1] + left_panel_w - 12,
                r.origin[2] + r.widths[2] - 142,
                r.origin[2] + r.widths[2] - 116
            )
        end,
        stored_string = string(state.cad_min_curv[]),
        displayed_string = string(state.cad_min_curv[]),
        placeholder = "",
        textpadding = (6, 6, 4, 4),
        fontsize = 11, tellwidth = false, tellheight = false,
    )
    set_overdraw_recursive!(tb_min_curv.blockscene)

    # Row 3: Remesh CAD button
    btn_remesh = Button(
        sc;
        bbox = lift(geom_card_rect, state.is_cad_file) do r, show_cad
            show_cad || return offscreen_bbox
            BBox(
                r.origin[1] + 12, r.origin[1] + left_panel_w - 12,
                r.origin[2] + 14, r.origin[2] + 42
            )
        end,
        label = "Remesh CAD", fontsize = 12, tellwidth = false, tellheight = false,
    )
    set_overdraw_recursive!(btn_remesh.blockscene)
    track_textbox_focus!(state, tb_elem_size, tb_min_curv)

    on(state.is_cad_file; update = true) do show_cad
        for widget in (tb_elem_size, tb_min_curv, btn_remesh)
            set_block_visible!(widget, show_cad)
        end
    end

    on(btn_remesh.clicks) do _
        state.is_cad_file[] || return
        es = tryparse(Float64, tb_elem_size.stored_string[])
        mc = tryparse(Float64, tb_min_curv.stored_string[])
        es !== nothing && es > 0 && (state.cad_elem_size[] = es)
        mc !== nothing && mc > 0 && (state.cad_min_curv[] = mc)

        state.voxel_confirmed[] = false
        state.meshdata[] = load_step_mesh_as_surface(
            state.current_filepath[];
            target_element_size = state.cad_elem_size[],
            min_curvature_size = state.cad_min_curv[],
            min_circle_points = state.cad_circle_pts[],
        )
    end

    on(btn_load.clicks) do _
        filepath = pick_file()
        (isempty(filepath) || !isfile(filepath)) && return

        ext = lowercase(splitext(filepath)[2])
        if ext in CAD_EXTS
            state.current_filepath[] = filepath
            state.is_cad_file[] = true
            state.voxel_confirmed[] = false
            state.meshdata[] = load_step_mesh_as_surface(
                filepath;
                target_element_size = state.cad_elem_size[],
                min_curvature_size = state.cad_min_curv[],
                min_circle_points = state.cad_circle_pts[],
            )
        elseif ext in MESH_EXTS
            state.current_filepath[] = filepath
            state.is_cad_file[] = false
            state.voxel_confirmed[] = false
            state.meshdata[] = MeshData(filepath)
        end
    end

    return geom_card_rect
end



function build_voxel_panel!(state::AppState, geom_card_rect::Observable{Rect2f}, left_panel_w::Int)
    sc = state.overlay_scene
    vox_panel_h = 166

    vox_card_rect = lift(geom_card_rect) do g_rect
        Rect2f(g_rect.origin[1], g_rect.origin[2] - vox_panel_h - 12, left_panel_w, vox_panel_h)
    end
    draw_panel_card!(sc, vox_card_rect; alpha = 0.90)

    text!(
        sc,
        @lift(Point2f($vox_card_rect.origin[1] + 12, $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 16));
        text = "Voxelization", fontsize = 13, font = :bold, color = :gray15,
        align = (:left, :center), overdraw = true,
    )

    text!(
        sc,
        @lift(Point2f($vox_card_rect.origin[1] + 12, $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 42));
        text = "Max Voxels:", fontsize = 11, color = :gray20,
        align = (:left, :center), overdraw = true,
    )

    tb_res = Textbox(
        sc;
        bbox = @lift(BBox(
            $vox_card_rect.origin[1] + 95, $vox_card_rect.origin[1] + left_panel_w - 12,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 54,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 30
        )),
        stored_string = string(state.res[]), placeholder = "100",
        fontsize = 11, tellwidth = false, tellheight = false,
    )
    set_overdraw_recursive!(tb_res.blockscene)
    track_textbox_focus!(state, tb_res)

    on(tb_res.stored_string) do s
        val = tryparse(Int, strip(s))
        if val !== nothing && 10 <= val <= 1000
            state.voxel_confirmed[] = false
            state.res[] != val && (state.res[] = val)
        end
    end

    btn_w3 = (left_panel_w - 36) / 3
    axis_labels = ("X", "Y", "Z")
    for ax in 1:3
        x_left  = 12 + (ax - 1) * (btn_w3 + 6)
        x_right = ax == 3 ? (left_panel_w - 12) : (x_left + btn_w3)
        lbl_str = axis_labels[ax]

        btn = Button(
            sc;
            bbox = @lift(BBox(
                $vox_card_rect.origin[1] + x_left, $vox_card_rect.origin[1] + x_right,
                $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 88,
                $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 64
            )),
            label = @lift($(state.slice_axis) == ax ? "[$lbl_str]" : lbl_str),
            fontsize = 11, tellwidth = false, tellheight = false,
        )
        set_overdraw_recursive!(btn.blockscene)
        on(btn.clicks) do _
            state.voxel_confirmed[] = false
            state.slice_axis[] = ax
        end
    end

    slice_slider = Slider(
        sc;
        bbox = @lift(BBox(
            $vox_card_rect.origin[1] + 12, $vox_card_rect.origin[1] + left_panel_w - 64,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 118,
            $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 96
        )),
        range = @lift(1:$(state.slice_max)),
        startvalue = state.slice_idx[],
        tellwidth = false, tellheight = false,
    )
    set_overdraw_recursive!(slice_slider.blockscene)

    on(slice_slider.value) do val
        state.slice_idx[] = val
    end
    on(state.slice_idx) do val
        slice_slider.value[] != val && set_close_to!(slice_slider, val)
    end

    text!(
        sc,
        @lift(Point2f($vox_card_rect.origin[1] + left_panel_w - 58, $vox_card_rect.origin[2] + $vox_card_rect.widths[2] - 107));
        text = @lift("$( $(state.slice_idx) )/$( $(state.slice_max) )"),
        fontsize = 10, color = :gray25, align = (:left, :center), overdraw = true,
    )

    btn_vox_ok = Button(
        sc;
        bbox = @lift(BBox(
            $vox_card_rect.origin[1] + 12, $vox_card_rect.origin[1] + left_panel_w - 12,
            $vox_card_rect.origin[2] + 12, $vox_card_rect.origin[2] + 38
        )),
        label = @lift($(state.voxel_confirmed) ? "Confirmed ✓" : "OK (Run Solver)"),
        fontsize = 12, tellwidth = false, tellheight = false,
    )
    set_overdraw_recursive!(btn_vox_ok.blockscene)

    on(btn_vox_ok.clicks) do _
        val = tryparse(Int, strip(tb_res.stored_string[]))
        if val !== nothing && 10 <= val <= 1000 && state.res[] != val
            state.res[] = val
        end
        state.voxel_confirmed[] = true
    end
end

function build_colorbar_panel!(state::AppState, card_w::Int)
    sc = state.overlay_scene
    bar_w, bar_h = 16, 260
    menu_h = 26
    card_h = bar_h + menu_h + 34

    cb_card_rect = lift(state.lscene.scene.viewport) do vp
        x0 = vp.origin[1] + vp.widths[1] - card_w - 14
        y0 = vp.origin[2] + vp.widths[2] - card_h - 14
        Rect2f(x0, y0, card_w, card_h)
    end
    draw_panel_card!(sc, cb_card_rect; alpha = 0.85)

    cmap_options = zip(
        ["turbo", "glasbey", "batlowWS"],
        [:turbo, :glasbey_bw_minc_20_minl_30_n256, :batlowWS],
    )
    cmap_menu = Menu(
        sc;
        bbox = @lift(BBox(
            $cb_card_rect.origin[1] + 10, $cb_card_rect.origin[1] + $cb_card_rect.widths[1] - 10,
            $cb_card_rect.origin[2] + $cb_card_rect.widths[2] - 12 - menu_h,
            $cb_card_rect.origin[2] + $cb_card_rect.widths[2] - 12
        )),
        options = cmap_options,
        default = "turbo",
        direction = :down,
        fontsize = 11,
        textpadding = (6, 6, 4, 4),
        tellwidth = false,
        tellheight = false,
    )
    set_overdraw_recursive!(cmap_menu.blockscene)
    on(cmap_menu.is_open) do _
        set_overdraw_recursive!(cmap_menu.blockscene)
    end
    on(cmap_menu.selection) do sel
        sel !== nothing && (state.cmap_name[] = sel)
    end

    cb = Colorbar(
        sc;
        bbox = @lift(BBox(
            $cb_card_rect.origin[1] + 12, $cb_card_rect.origin[1] + $cb_card_rect.widths[1] - 10,
            $cb_card_rect.origin[2] + 12, $cb_card_rect.origin[2] + $cb_card_rect.widths[2] - 22 - menu_h
        )),
        limits = state.color_limits,
        colormap = state.base_cmap,
        width = bar_w, height = bar_h,
        halign = :left, valign = :center,
        label = "Fill Time",
        tickformat = "{:.3f}",
        ticklabelsize = 12, labelsize = 13,
        labelpadding = 6, ticklabelpad = 6, tickalign = 0,
        tellwidth = false, tellheight = false,
    )
    set_overdraw_recursive!(cb.blockscene)
    on(state.color_limits) do _
        set_overdraw_recursive!(cb.blockscene)
    end

    return cb_card_rect
end

function build_timeline_panel!(state::AppState)
    sc = state.overlay_scene
    slider_card_h = 54

    slider_card_rect = lift(state.lscene.scene.viewport) do vp
        w = clamp(vp.widths[1] * 0.52, 340.0, 680.0)
        x0 = vp.origin[1] + (vp.widths[1] - w) * 0.5
        y0 = vp.origin[2] + 16
        Rect2f(x0, y0, w, slider_card_h)
    end
    draw_panel_card!(sc, slider_card_rect; alpha = 0.88)

    play_btn = Button(
        sc;
        bbox = @lift(BBox(
            $slider_card_rect.origin[1] + 12, $slider_card_rect.origin[1] + 46,
            $slider_card_rect.origin[2] + 11, $slider_card_rect.origin[2] + $slider_card_rect.widths[2] - 11
        )),
        label = @lift($(state.is_playing) ? "||" : "▶"),
        fontsize = 13, tellwidth = false, tellheight = false,
    )

    fill_slider = Slider(
        sc;
        bbox = @lift(BBox(
            $slider_card_rect.origin[1] + 56, $slider_card_rect.origin[1] + $slider_card_rect.widths[1] - 165,
            $slider_card_rect.origin[2] + 11, $slider_card_rect.origin[2] + $slider_card_rect.widths[2] - 11
        )),
        range = 1:state.n_stops,
        startvalue = state.n_stops,
        tellwidth = false, tellheight = false,
    )
    set_overdraw_recursive!(play_btn.blockscene)
    set_overdraw_recursive!(fill_slider.blockscene)

    on(fill_slider.value) do val
        state.fill_step[] = val
    end

    time_label_str = lift(state.fill_step, state.t_max) do step, tmax
        frac = (step - 1) / (state.n_stops - 1)
        t_cur = frac * tmax
        pct = round(Int, frac * 100)
        "$(round(t_cur; digits=3)) / $(round(tmax; digits=3)) ($(pct)%)"
    end

    text!(
        sc,
        @lift(Point2f(
            $slider_card_rect.origin[1] + $slider_card_rect.widths[1] - 155,
            $slider_card_rect.origin[2] + $slider_card_rect.widths[2] * 0.5
        ));
        text = time_label_str, fontsize = 12, color = :gray20,
        align = (:left, :center), overdraw = true,
    )

    on(play_btn.clicks) do _
        state.is_playing[] = !state.is_playing[]
        if state.is_playing[]
            @async begin
                while state.is_playing[] && isopen(state.fig.scene)
                    sleep(0.06)
                    state.is_playing[] || break
                    cur = fill_slider.value[]
                    if cur < state.n_stops
                        set_close_to!(fill_slider, cur + 1)
                    else
                        sleep(0.35)
                        state.is_playing[] || break
                        set_close_to!(fill_slider, 1)
                    end
                end
            end
        end
    end
end

function build_overlays_panel!(state::AppState, cb_card_rect::Observable{Rect2f}, card_w::Int)
    sc = state.overlay_scene
    panel_w = card_w
    panel_h = 150

    panel_card_rect = lift(state.lscene.scene.viewport, cb_card_rect) do vp, cb_rect
        x0 = vp.origin[1] + vp.widths[1] - panel_w - 14
        y0 = cb_rect.origin[2] - panel_h - 12
        Rect2f(x0, y0, panel_w, panel_h)
    end
    draw_panel_card!(sc, panel_card_rect; alpha = 0.88)

    text!(
        sc,
        @lift(Point2f($panel_card_rect.origin[1] + 12, $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - 18));
        text = "Overlays", fontsize = 13, font = :bold, color = :gray15,
        align = (:left, :center), overdraw = true,
    )

    items = (
        ("Flow Direction", state.show_flow_dirs),
        ("Weld Lines",     state.show_weld_lines),
        ("Air Traps",      state.show_air_traps),
        ("Wireframe",      state.show_wireframe),
    )

    for (idx, (label_txt, obs)) in enumerate(items)
        y_top = 34 + (idx - 1) * 28
        y_bot = y_top + 18
        y_mid = 0.5 * (y_top + y_bot)

        chk = Checkbox(
            sc;
            bbox = @lift(BBox(
                $panel_card_rect.origin[1] + 12, $panel_card_rect.origin[1] + 30,
                $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - y_bot,
                $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - y_top
            )),
            checked = obs[], tellwidth = false, tellheight = false,
        )
        on(chk.checked) do val
            obs[] = val
        end
        set_overdraw_recursive!(chk.blockscene)

        text!(
            sc,
            @lift(Point2f($panel_card_rect.origin[1] + 36, $panel_card_rect.origin[2] + $panel_card_rect.widths[2] - y_mid));
            text = label_txt, fontsize = 12, color = :gray20,
            align = (:left, :center), overdraw = true,
        )
    end
end

function build_hud!(state::AppState)
    left_panel_w = 260
    cb_card_w    = 140

    geom_rect = build_geometry_panel!(state, left_panel_w)
    build_voxel_panel!(state, geom_rect, left_panel_w)

    cb_rect = build_colorbar_panel!(state, cb_card_w)
    build_overlays_panel!(state, cb_rect, cb_card_w)
    build_timeline_panel!(state)
end


# =============================================================================
# 7. Entry Point
# =============================================================================

function main()
    state = init_app_state()
    setup_3d_scene!(state)
    register_interaction_handlers!(state)
    build_hud!(state)
    return state.fig
end