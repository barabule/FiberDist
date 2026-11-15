

# 1. Load STEP and generate surface mesh via Gmsh
function load_step_mesh_as_surface(filepath::String;
            target_element_size = 1.0)

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
    mesh = normal_mesh(points, triangles)
    # tlas = Raycore.TLAS(KernelAbstractions.CPU()) 
    # push!(tlas, mesh)
    # sync!(tlas)
    tlas = TLAS([mesh], (mi, ti) -> UInt32(mi))
    # return (;mesh, tlas)
    return MeshData{Float32}(mesh, tlas)
end
