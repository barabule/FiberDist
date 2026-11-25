

# 1. Load STEP and generate surface mesh via Gmsh
function load_step_mesh_as_surface(filepath::AbstractString;
            target_element_size = 2.0,
            min_curvature_size = 1.0,
            min_circle_points = 15,
            )

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(filepath)
    gmsh.model.occ.synchronize()
    
    # --- CONTROL MESH SIZE HERE ---
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_element_size)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", min_curvature_size)
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", min_circle_points)

    # Generate 2D surface mesh
    gmsh.model.mesh.generate(2)
    
    node_tags, coord_coords, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    
    gmsh.finalize()
    
    pointmatrix = reshape(coord_coords, 3, :)
    # Extract triangles (element type 2 in Gmsh)
    tri_node_tags = elem_node_tags[findfirst(elem_types .== 2)]
    trimatrix = reshape(tri_node_tags, 3, :)

    nodes = [Point3f(pointmatrix[1, i], pointmatrix[2, i], pointmatrix[3, i]) for i in axes(pointmatrix, 2)]
    
    tag_to_idx = Dict{eltype(node_tags), UInt32}(
        tag => UInt32(i) for (i, tag) in enumerate(node_tags)
    )
    elements = map(1:size(trimatrix, 2)) do i
        GeometryBasics.TriangleFace{UInt32}(
            tag_to_idx[trimatrix[1, i]],
            tag_to_idx[trimatrix[2, i]],
            tag_to_idx[trimatrix[3, i]]
        )
    end
    
    return (;nodes,
            elements) |> build_scene_bvh
end

# 2. Build Raycore BVH Acceleration Structure from Mesh
function build_scene_bvh(data)
    nodes, elements = data
   
    mesh = GeometryBasics.mesh(nodes, elements) #we don't need vertex normals
    # @info typeof(mesh)
    # @info supertype(typeof(mesh))

    # tlas = TLAS([mesh], (mi, ti) -> UInt32(mi))
    return MeshData(mesh)
end



"""
    MeshData(fn::AbstractString)

TBW
"""
function MeshData(fn::AbstractString)
    isfile(fn) || throw(ArgumentError("File does not exist: $fn"))

    raw_mesh = load(fn)
    if !(raw_mesh isa GeometryBasics.AbstractMesh)
        error("Loaded file is not a mesh file (got $(typeof(raw_mesh)))")
    end

    tri_faces = GeometryBasics.decompose(TriangleFace{UInt32}, raw_mesh)
    vertices  = GeometryBasics.coordinates(raw_mesh)

    bb = GeometryBasics.Rect3(vertices)
    diag = norm(bb.widths)
    weld_tol = max(1e-5, 1e-6 * Float64(diag))

    welded = weld_vertices(vertices, tri_faces; tol = weld_tol)
    oriented_faces = flip_normals_outward(welded.nodes, welded.elements)

    if !is_valid(welded.nodes, oriented_faces)
        half_edges = Dict{Tuple{Int, Int}, Int}()
        for f in oriented_faces, (u, v) in ((Int(f[1]), Int(f[2])), (Int(f[2]), Int(f[3])), (Int(f[3]), Int(f[1])))
            half_edges[(u, v)] = get(half_edges, (u, v), 0) + 1
        end
        n_open = count(((u, v), c) -> c == 1 && get(half_edges, (v, u), 0) == 0, half_edges)
        n_nonmanifold = count(((_, _), c) -> c > 1, half_edges)
        error("Mesh is not watertight (open boundary edges: $n_open, non-manifold/flipped edges: $n_nonmanifold).")
    end

    mesh = GeometryBasics.Mesh(welded.nodes, oriented_faces)
    return MeshData(mesh)
end






"""
    weld_vertices(pts::AbstractVector{PT}, tris::AbstractVector{<:TriangleFace};
                  tol::Real = 1f-5) where PT <: StaticVector{3, T} where T <: Real

Merges duplicate vertex positions within `tol` using 27-neighbor spatial hashing
and removes zero-area degenerate triangles collapsed by welding.
"""
function weld_vertices(
    pts::AbstractVector{PT},
    tris::AbstractVector{<:TriangleFace};
    tol::Real = 1f-5
) where PT <: StaticVector{3, T} where T <: Real
    unique_pts = PT[]
    remap = Vector{Int}(undef, length(pts))
    tol_f = Float64(tol)
    tol2  = tol_f * tol_f
    inv_tol = inv(tol_f)

    grid = Dict{NTuple{3, Int}, Vector{Int}}()
    sizehint!(grid, length(pts))

    @inbounds for (i, p) in enumerate(pts)
        px, py, pz = Float64(p[1]), Float64(p[2]), Float64(p[3])
        cx = floor(Int, px * inv_tol)
        cy = floor(Int, py * inv_tol)
        cz = floor(Int, pz * inv_tol)

        found_idx = 0
        for dz in -1:1, dy in -1:1, dx in -1:1
            bucket = get(grid, (cx + dx, cy + dy, cz + dz), nothing)
            bucket === nothing && continue
            for cand in bucket
                q = unique_pts[cand]
                ddx = px - Float64(q[1])
                ddy = py - Float64(q[2])
                ddz = pz - Float64(q[3])
                if (ddx * ddx + ddy * ddy + ddz * ddz) <= tol2
                    found_idx = cand
                    break
                end
            end
            found_idx != 0 && break
        end

        if found_idx == 0
            push!(unique_pts, p)
            found_idx = length(unique_pts)
            bucket = get!(Vector{Int}, grid, (cx, cy, cz))
            push!(bucket, found_idx)
        end
        remap[i] = found_idx
    end

    welded_tris = TriangleFace{UInt32}[]
    sizehint!(welded_tris, length(tris))
    @inbounds for f in tris
        v1 = UInt32(remap[f[1]])
        v2 = UInt32(remap[f[2]])
        v3 = UInt32(remap[f[3]])
        if v1 != v2 && v2 != v3 && v3 != v1
            push!(welded_tris, TriangleFace{UInt32}(v1, v2, v3))
        end
    end

    return (; nodes = unique_pts, elements = welded_tris)
end


function signed_volume_6(
    pts::AbstractVector{<:StaticVector{3}},
    tris::AbstractVector{<:GeometryBasics.TriangleFace}
)::Float64
    isempty(tris) && return 0.0
    p0 = SVector{3, Float64}(pts[1])
    vol6 = 0.0
    @inbounds for f in tris
        p1 = SVector{3, Float64}(pts[f[1]]) - p0
        p2 = SVector{3, Float64}(pts[f[2]]) - p0
        p3 = SVector{3, Float64}(pts[f[3]]) - p0
        vol6 += dot(p1, cross(p2, p3))
    end
    return vol6
end

"""
    harmonize_winding(tris) -> Vector{TriangleFace{UInt32}}

Propagates consistent triangle orientation across shared manifold edges via BFS
so adjacent triangles always traverse shared edges in opposite directions.
"""
function harmonize_winding(tris::AbstractVector{<:GeometryBasics.TriangleFace})
    n = length(tris)
    out = [TriangleFace{UInt32}(UInt32(f[1]), UInt32(f[2]), UInt32(f[3])) for f in tris]
    n == 0 && return out

    # Map undirected edge (min, max) -> list of incident triangle indices
    edge_tris = Dict{Tuple{UInt32, UInt32}, Vector{Int}}()
    sizehint!(edge_tris, 3n ÷ 2)
    @inbounds for (ti, f) in enumerate(out)
        for (u, v) in ((f[1], f[2]), (f[2], f[3]), (f[3], f[1]))
            key = u < v ? (u, v) : (v, u)
            push!(get!(Vector{Int}, edge_tris, key), ti)
        end
    end

    visited = falses(n)
    queue = Int[]
    sizehint!(queue, n)

    @inbounds for seed in 1:n
        visited[seed] && continue
        visited[seed] = true
        push!(queue, seed)

        while !isempty(queue)
            t = popfirst!(queue)
            ft = out[t]
            for (u, v) in ((ft[1], ft[2]), (ft[2], ft[3]), (ft[3], ft[1]))
                key = u < v ? (u, v) : (v, u)
                inc = edge_tris[key]
                length(inc) == 2 || continue # only propagate across 2-manifold edges
                nb = inc[1] == t ? inc[2] : inc[1]
                visited[nb] && continue

                fnb = out[nb]
                # Neighbor must contain directed edge (v, u); if it has (u, v), flip it
                same_dir = (fnb[1] == u && fnb[2] == v) ||
                           (fnb[2] == u && fnb[3] == v) ||
                           (fnb[3] == u && fnb[1] == v)
                if same_dir
                    out[nb] = TriangleFace{UInt32}(fnb[1], fnb[3], fnb[2])
                end
                visited[nb] = true
                push!(queue, nb)
            end
        end
    end

    return out
end

function flip_normals_outward(
    pts::AbstractVector{<:StaticVector{3}},
    tris::AbstractVector{<:GeometryBasics.TriangleFace}
)
    consistent = harmonize_winding(tris)
    if signed_volume_6(pts, consistent) < 0.0
        return [TriangleFace{UInt32}(f[1], f[3], f[2]) for f in consistent]
    end
    return consistent
end

function is_valid(
    pts::AbstractVector{<:StaticVector{3}},
    tris::AbstractVector{<:GeometryBasics.TriangleFace}
)
    isempty(tris) && return false

    half_edges = Dict{Tuple{Int, Int}, Int}()
    sizehint!(half_edges, 3 * length(tris))

    @inbounds for f in tris
        v1, v2, v3 = convert(Int, f[1]), convert(Int, f[2]), convert(Int, f[3])
        (v1 == v2 || v2 == v3 || v3 == v1) && return false

        for (u, v) in ((v1, v2), (v2, v3), (v3, v1))
            half_edges[(u, v)] = get(half_edges, (u, v), 0) + 1
        end
    end

    for ((u, v), count) in half_edges
        count != 1 && return false
        get(half_edges, (v, u), 0) != 1 && return false
    end

    return signed_volume_6(pts, tris) > 0.0
end

function is_valid(mesh::GeometryBasics.AbstractMesh)
    pts  = GeometryBasics.coordinates(mesh)
    tris = GeometryBasics.decompose(TriangleFace{Int}, mesh)
    return is_valid(pts, tris)
end




function MeshAdjacency(mesh::GeometryBasics.AbstractMesh)
    tris = mesh.faces
    n_tris = length(tris)

    edges = Vector{Tuple{Int32, Int32, Int32}}(undef, 3 * n_tris)
    @inbounds for idx in 1:n_tris
        v1, v2, v3 = tris[idx]
        t = Int32(idx)
        edges[3idx - 2] = (min(v1, v2), max(v1, v2), t)
        edges[3idx - 1] = (min(v2, v3), max(v2, v3), t)
        edges[3idx]     = (min(v3, v1), max(v3, v1), t)
    end

    sort!(edges)

    counts = zeros(Int32, n_tris)
    @inbounds for e in 1:(length(edges) - 1)
        if edges[e][1] == edges[e + 1][1] && edges[e][2] == edges[e + 1][2]
            counts[edges[e][3]]     += Int32(1)
            counts[edges[e + 1][3]] += Int32(1)
        end
    end

    offsets = Vector{Int32}(undef, n_tris + 1)
    offsets[1] = Int32(1)
    @inbounds for i in 1:n_tris
        offsets[i + 1] = offsets[i] + counts[i]
    end

    neighbors = Vector{Int32}(undef, offsets[end] - 1)
    cursor = copy(offsets)

    @inbounds for e in 1:(length(edges) - 1)
        if edges[e][1] == edges[e + 1][1] && edges[e][2] == edges[e + 1][2]
            t1, t2 = edges[e][3], edges[e + 1][3]
            neighbors[cursor[t1]] = t2; cursor[t1] += Int32(1)
            neighbors[cursor[t2]] = t1; cursor[t2] += Int32(1)
        end
    end

    return MeshAdjacency(offsets, neighbors)
end

@inline tri_neighbors(adj::MeshAdjacency, idx::Int) =
    @view adj.neighbors[adj.offsets[idx]:(adj.offsets[idx + 1] - 1)]

