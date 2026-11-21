

# 1. Load STEP and generate surface mesh via Gmsh
function load_step_mesh_as_surface(filepath::String;
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
    
    elements = map(1:size(trimatrix, 2)) do i
        i1, i2, i3 = trimatrix[1, i], trimatrix[2, i], trimatrix[3, i]
        GeometryBasics.TriangleFace(i1, i2, i3)
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

    tlas = TLAS([mesh], (mi, ti) -> UInt32(mi))
    return MeshData(mesh, tlas)
end



function meshdata(fn::AbstractString; check_watertight = true)
    isfile(fn) || throw(ArgumentError("File does not exist: $fn"))

    raw_mesh = load(fn)
    if !(raw_mesh isa GeometryBasics.AbstractMesh)
        error("Loaded file is not a mesh file (got $(typeof(raw_mesh)))")
    end

    tri_faces = GeometryBasics.decompose(TriangleFace{UInt32}, raw_mesh)

    vertices = raw_mesh.position

    welded = weld_vertices(vertices, tri_faces)

    if check_watertight && !is_valid(welded.nodes, welded.elements)
        error("Mesh is not is_watertight!")
    end

    return build_scene_bvh(welded)
end


"""
    is_valid(mesh::GeometryBasics.AbstractMesh)

Returns `true` if `mesh` is:
1. Closed and 2-manifold (watertight)
2. Consistently oriented (adjacent triangles have opposite half-edge directions)
3. Outward-facing (positive signed volume)
"""
function is_valid(pts::AbstractVector{<:StaticVector{3}}, tris::AbstractVector{<:GeometryBasics.TriangleFace})
    # pts = mesh.position
    # tris = mesh.faces
    isempty(tris) && return false

    half_edges = Dict{Tuple{Int, Int}, Int}()
    sizehint!(half_edges, 3 * length(tris))

    # Accumulate 6 * signed volume using tetrahedra formed with the origin
    signed_vol_6 = 0.0

    for f in tris
        v1, v2, v3 = f[1], f[2], f[3]

        # 1. Check for degenerate index triangles
        (v1 == v2 || v2 == v3 || v3 == v1) && return false

        # 2. Record directed half-edges
        for (u, v) in ((v1, v2), (v2, v3), (v3, v1))
            half_edges[(u, v)] = get(half_edges, (u, v), 0) + 1
        end

        # 3. Accumulate signed volume: dot(p1, cross(p2, p3))
        p1, p2, p3 = pts[v1], pts[v2], pts[v3]
        signed_vol_6 += dot(p1, cross(p2, p3))
    end

    # 4. Verify watertightness AND consistent winding
    # If every edge (u, v) has an exact opposite twin (v, u), all triangles are consistently oriented
    for ((u, v), count) in half_edges
        count != 1 && return false
        get(half_edges, (v, u), 0) != 1 && return false
    end

    # 5. Verify normals point outward (positive volume)
    return signed_vol_6 > 0
end

function is_valid(mesh::GeometryBasics.AbstractMesh)
    return is_valid(mesh.position, mesh.faces)
end



"""
    weld_vertices(pts::AbstractVector{PT}, tris::AbstractVector{<:TriangleFace}; 
                        tol::Real=1e-8) where PT<:StaticVector{3, T} where T<:Real

Merges duplicate vertex positions within `tol` and remaps triangle indices.
"""
function weld_vertices(pts::AbstractVector{PT}, tris::AbstractVector{<:TriangleFace}; 
                        tol::Real=1e-8) where PT<:StaticVector{3, T} where T<:Real
    

    unique_pts = PT[]
    remap = Vector{Int}(undef, length(pts))
    inv_tol = inv(tol)

    # Spatial hashing via rounded grid coordinates
    grid = Dict{NTuple{3, Int}, Int}()

    for (i, p) in enumerate(pts)
        key = (round(Int, p[1] * inv_tol), round(Int, p[2] * inv_tol), round(Int, p[3] * inv_tol))
        idx = get(grid, key, 0)
        if idx == 0
            push!(unique_pts, p)
            idx = length(unique_pts)
            grid[key] = idx
        end
        remap[i] = idx
    end

    welded_tris = [TriangleFace(remap[f[1]], remap[f[2]], remap[f[3]]) for f in tris]
    return (;nodes = unique_pts, elements = welded_tris)
end



function build_mesh_adjacency(mesh)
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

