

# 1. Load STEP and generate surface mesh via Gmsh
function load_step_mesh_as_surface(filepath::String;
            target_element_size = 2.0)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(filepath)
    gmsh.model.occ.synchronize()
    
    # --- CONTROL MESH SIZE HERE ---
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_element_size)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", 15)

    # Generate 2D surface mesh
    gmsh.model.mesh.generate(2)
    
    node_tags, coord_coords, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    
    gmsh.finalize()
    
    nodes = reshape(coord_coords, 3, :)
    # Extract triangles (element type 2 in Gmsh)
    tri_node_tags = elem_node_tags[findfirst(elem_types .== 2)]
    elements = reshape(tri_node_tags, 3, :)
    
    return (;nodes, elements) |> build_scene_bvh
end

# 2. Build Raycore BVH Acceleration Structure from Mesh
function build_scene_bvh(data)
    nodes, elements = data
    # Convert nodes & elements into GeometryBasics triangles format required by Raycore
    points = [Point3f(nodes[1, i], nodes[2, i], nodes[3, i]) for i in axes(nodes, 2)]
    
    triangles = map(1:size(elements, 2)) do i
        i1, i2, i3 = elements[1, i], elements[2, i], elements[3, i]
        GeometryBasics.TriangleFace(i1, i2, i3)
    end
    # @info eltype(triangles)
    # @info eltype(triangles) <:GeometryBasics.AbstractFace
    # Build Raycore Top-Level Acceleration Structure (TLAS / BVH)
    mesh = triangle_mesh(points, triangles) #we don't need vertex normals
    # tlas = Raycore.TLAS(KernelAbstractions.CPU()) 
    # push!(tlas, mesh)
    # sync!(tlas)
    tlas = TLAS([mesh], (mi, ti) -> UInt32(mi))
    # return (;mesh, tlas)
    return MeshData{Float32}(mesh, tlas)
end



function meshdata(fn::AbstractString; check_watertight = false)
    isfile(fn) || throw(ArgumentError("File does not exist: $fn"))

    raw_mesh = load(fn)
    if !(data isa GeometryBasics.AbstractMesh)
        error("Loaded file is not a mesh file (got $(typeof(raw_mesh)))")
    end

    tri_faces = GeometryBasics.decompose(TriangleFace{UInt32}, raw_mesh)

    vertices = coordinates(raw_mesh)

    welded = weld_vertices(vertices, tri_faces)
    if !is_watertight(welded.elements)
        error("Mesh is not is_watertight!")
    end

    return build_scene_bvh((;nodes = vertices, elements = tri_faces))
end

"""
    is_watertight(tris::AbstractVector{<:TriangleFace})

Returns `true` if `mesh` is a closed, 2-manifold triangular mesh with consistent face winding.
Assumes all mesh nodes are welded (no duplicates). 
"""
function is_watertight(tris::AbstractVector{<:TriangleFace})
    # Extract triangle faces (decomposes quads/n-gons if present)
    
    isempty(tris) && return false

    # Count occurrences of each directed half-edge (u -> v)
    half_edges = Dict{Tuple{Int, Int}, Int}()
    sizehint!(half_edges, 3 * length(tris))

    for f in tris
        v1, v2, v3 = f[1], f[2], f[3]
        
        # Degenerate triangle check
        (v1 == v2 || v2 == v3 || v3 == v1) && return false

        for (u, v) in ((v1, v2), (v2, v3), (v3, v1))
            half_edges[(u, v)] = get(half_edges, (u, v), 0) + 1
        end
    end

    # Every directed edge (u, v) must appear once and have an exact opposite twin (v, u)
    for ((u, v), count) in half_edges
        count != 1 && return false
        get(half_edges, (v, u), 0) != 1 && return false
    end

    return true
end

function is_watertight(mesh::AbstractMesh)
    tris = decompose(mesh.faces)
    return is_watertight(tris)
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

