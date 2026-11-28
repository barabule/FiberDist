


@inline _read_field(x) = x
@inline _read_field(x::Observable) = x[]

"""
    project_to_dict(src) -> Dict{String, Any}

Extracts a schema-versioned dictionary of plain Julia primitives from any object `src`
(such as `AppState` or a `NamedTuple`) exposing the project fields. Fields may be
either `Observable`s or plain values.
"""
function project_to_dict(src)::Dict{String, Any}
    md    = _read_field(src.meshdata)
    verts = GeometryBasics.coordinates(md.mesh)
    faces = GeometryBasics.faces(md.mesh)

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

    gts      = _read_field(src.gates)
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

    return Dict{String, Any}(
        "schema_version"      => 1,
        "current_filepath"    => String(_read_field(src.current_filepath)),
        "is_cad_file"         => Bool(_read_field(src.is_cad_file)),
        "cad_elem_size"       => Float64(_read_field(src.cad_elem_size)),
        "cad_min_curv"        => Float64(_read_field(src.cad_min_curv)),
        "cad_circle_pts"      => Int(_read_field(src.cad_circle_pts)),
        "mesh_vertices"       => vertices_mat,
        "mesh_faces"          => faces_mat,
        "res"                 => Int(_read_field(src.res)),
        "gate_positions"      => gate_pos,
        "gate_radii"          => gate_rad,
        "gate_open_times"     => gate_t0,
        "polymer"             => Symbol(_read_field(src.polymer)),
        "fiber"               => Symbol(_read_field(src.fiber)),
        "fiber_wt_fraction"   => Float32(_read_field(src.fiber_wt_fraction)),
        "num_half_thk_layers" => Int(_read_field(src.num_half_thk_layers)),
    )
end

"""
    dict_to_project(data::AbstractDict) -> NamedTuple

Reconstructs pure domain data (`MeshData`, `Vector{ValveGate}`, and scalar settings)
from a serialized project dictionary without triggering any GUI or solver side effects.
"""
function dict_to_project(data::AbstractDict)
    @assert get(data, "schema_version", 0) == 1 "Unsupported project schema version"

    v_mat = data["mesh_vertices"]::Matrix{Float32}
    f_mat = data["mesh_faces"]::Matrix{Int32}
    pts   = [Point3f(v_mat[1, i], v_mat[2, i], v_mat[3, i]) for i in axes(v_mat, 2)]
    tris  = [GeometryBasics.TriangleFace{Int}(f_mat[1, i], f_mat[2, i], f_mat[3, i]) for i in axes(f_mat, 2)]
    md    = MeshData(GeometryBasics.mesh(pts, tris))

    g_pos = data["gate_positions"]::Matrix{Float32}
    g_rad = data["gate_radii"]::Vector{Float32}
    g_t0  = data["gate_open_times"]::Vector{Float32}
    gates = [
        ValveGate(
            position  = SVector{3, Float32}(g_pos[1, i], g_pos[2, i], g_pos[3, i]),
            radius    = g_rad[i],
            open_time = g_t0[i],
        )
        for i in eachindex(g_rad)
    ]

    return (
        current_filepath    = String(data["current_filepath"]),
        is_cad_file         = Bool(data["is_cad_file"]),
        cad_elem_size       = Float64(data["cad_elem_size"]),
        cad_min_curv        = Float64(data["cad_min_curv"]),
        cad_circle_pts      = Int(data["cad_circle_pts"]),
        meshdata            = md,
        res                 = Int(data["res"]),
        gates               = gates,
        polymer             = Symbol(data["polymer"]),
        fiber               = Symbol(data["fiber"]),
        fiber_wt_fraction   = Float32(data["fiber_wt_fraction"]),
        num_half_thk_layers = Int(data["num_half_thk_layers"]),
    )
end


##Generic storage and retrieval

function save_project_dict(filepath::AbstractString, data::AbstractDict)
    jldsave(filepath; compress = true, project = Dict{String, Any}(data))
    return filepath
end

function load_project_dict(filepath::AbstractString)::Dict{String, Any}
    return load(filepath, "project")::Dict{String, Any}
end