

function save_project(filepath::AbstractString, state::AppState)
    md = state.meshdata[]
    verts = GeometryBasics.coordinates(md.mesh)
    faces = GeometryBasics.faces(md.mesh)

    # Store mesh as plain (3 × N_v) and (3 × N_f) matrices
    vertices_mat = Matrix{Float32}(undef, 3, length(verts))
    @inbounds for (i, p) in enumerate(verts)
        vertices_mat[1, i] = p[1]
        vertices_mat[2, i] = p[2]
        vertices_mat[3, i] = p[3]
    end

    faces_mat = Matrix{Int32}(undef, 3, length(faces))
    @inbounds for (i, f) in enumerate(faces)
        faces_mat[1, i] = f[1]
        faces_mat[2, i] = f[2]
        faces_mat[3, i] = f[3]
    end

    gts = state.gates[]
    gate_pos = Matrix{Float32}(undef, 3, length(gts))
    gate_rad = Vector{Float32}(undef, length(gts))
    gate_t0  = Vector{Float32}(undef, length(gts))
    @inbounds for (i, g) in enumerate(gts)
        gate_pos[1, i] = g.position[1]
        gate_pos[2, i] = g.position[2]
        gate_pos[3, i] = g.position[3]
        gate_rad[i]    = g.radius
        gate_t0[i]     = g.open_time
    end

    payload = Dict{String, Any}(
        "schema_version"      => 1,
        "source_filepath"     => state.current_filepath[],
        "is_cad_file"         => state.is_cad_file[],
        "cad_elem_size"       => state.cad_elem_size[],
        "cad_min_curv"        => state.cad_min_curv[],
        "cad_circle_pts"      => state.cad_circle_pts[],
        "mesh_vertices"       => vertices_mat,
        "mesh_faces"          => faces_mat,
        "voxel_res"           => state.res[],
        "gate_positions"      => gate_pos,
        "gate_radii"          => gate_rad,
        "gate_open_times"     => gate_t0,
        "polymer"             => state.polymer[],
        "fiber"               => state.fiber[],
        "fiber_wt_fraction"   => state.fiber_wt_fraction[],
        "num_half_thk_layers" => state.num_half_thk_layers[],
    )

    jldsave(filepath; compress = true, project = payload)
end

function load_project!(state::AppState, filepath::AbstractString)
    data = load(filepath, "project")
    @assert get(data, "schema_version", 0) == 1 "Unsupported project file version"

    # 1. Reconstruct MeshData (and TLAS) directly from saved triangles—no CAD remesh needed
    v_mat = data["mesh_vertices"]::Matrix{Float32}
    f_mat = data["mesh_faces"]::Matrix{Int32}
    pts   = [Point3f(v_mat[1, i], v_mat[2, i], v_mat[3, i]) for i in axes(v_mat, 2)]
    tris  = [GeometryBasics.TriangleFace{Int}(f_mat[1, i], f_mat[2, i], f_mat[3, i]) for i in axes(f_mat, 2)]
    new_md = MeshData(GeometryBasics.mesh(pts, tris))

    # 2. Restore metadata & voxel resolution
    state.current_filepath[] = data["source_filepath"]
    state.is_cad_file[]      = data["is_cad_file"]
    state.cad_elem_size[]    = data["cad_elem_size"]
    state.cad_min_curv[]     = data["cad_min_curv"]
    state.cad_circle_pts[]   = data["cad_circle_pts"]
    state.voxel_confirmed[]  = false
    state.res[]              = data["voxel_res"]
    state.meshdata[]         = new_md

    # 3. Restore gates (overriding the default single gate created by on(state.meshdata))
    g_pos = data["gate_positions"]::Matrix{Float32}
    g_rad = data["gate_radii"]::Vector{Float32}
    g_t0  = data["gate_open_times"]::Vector{Float32}
    restored_gates = [
        ValveGate(
            position  = SVector{3, Float32}(g_pos[1, i], g_pos[2, i], g_pos[3, i]),
            radius    = g_rad[i],
            open_time = g_t0[i],
        )
        for i in eachindex(g_rad)
    ]
    state.gates[]        = restored_gates
    state.solver_gates[] = with_computed_gate_delays(restored_gates, state.voxeldata[])

    # 4. Restore material & layer settings, then trigger solver + FOT reconstruction
    state.polymer[]             = data["polymer"]
    state.fiber[]               = data["fiber"]
    state.fiber_wt_fraction[]   = data["fiber_wt_fraction"]
    state.num_half_thk_layers[] = data["num_half_thk_layers"]
    state.voxel_confirmed[]     = true
end