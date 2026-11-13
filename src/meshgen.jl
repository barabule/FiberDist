

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
    
    return (;nodes, elements)
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
    @info eltype(triangles)
    @info eltype(triangles) <:GeometryBasics.AbstractFace
    # Build Raycore Top-Level Acceleration Structure (TLAS / BVH)
    mesh = normal_mesh(points, triangles)
    # tlas = Raycore.TLAS(KernelAbstractions.CPU()) 
    # push!(tlas, mesh)
    # sync!(tlas)
    tlas = TLAS([mesh], (mi, ti) -> UInt32(mi))
    return (;mesh, tlas)
end

# 3. Compute Local Thickness and Mid-Surface via Raycore Ray-Casting
function compute_thickness_field(tlasdata)
    
    mesh, tlas = tlasdata

    # thicknesses = zeros(Float32, size(nodes, 2))
    thicknesses = zeros(Float32, length(mesh.position))
    midpoints = zeros(eltype(mesh.position),length(mesh.position))
    

    for i in eachindex(mesh.position)
        pos = mesh.position[i]
        normal = mesh.normal[i]
        ray_dir = -normal
        ray = Raycore.Ray(o = pos - 1f-3 * normal,d =  ray_dir)
        # Query Raycore BVH for closest hit
        # result_closest = Raycore.closest_hit(tlas, ray)
        hit_found, triangle, distance, bary, instance_idx = Raycore.closest_hit(tlas, ray)
        # @info result_closest
        # return result_closest
        if hit_found && distance > 1f-4
            thicknesses[i] = distance
            midpoints[i] = pos .+ (ray_dir * (distance / 2.0f0))
        else
            thicknesses[i] = 0.0f0
            midpoints[i] = pos
        end

    end
    
    
    return (;thicknesses, midpoints)
end

function compute_thickness_field_by_face(tlasdata)
    
    mesh, tlas = tlasdata

    # thicknesses = zeros(Float32, size(nodes, 2))
    thicknesses = zeros(Float32, length(mesh.faces))
    midpoints = zeros(eltype(mesh.position),length(mesh.faces))
    

    for i in eachindex(mesh.faces)
        face = mesh.faces[i]
        i1, i2, i3 = face
        V1, V2, V3 = mesh.position[i1], mesh.position[i2], mesh.position[i3]
        pos = 1/3 * (V1 + V2 + V3)
        N1, N2, N3 = mesh.normal[i1], mesh.normal[i2], mesh.normal[i3]
        normal = 1/3 * (N1 + N2 + N3)
        ray_dir = -normal
        ray = Raycore.Ray(o = pos - 1f-3 * normal,d =  ray_dir)
        # Query Raycore BVH for closest hit
        # result_closest = Raycore.closest_hit(tlas, ray)
        hit_found, triangle, distance, bary, instance_idx = Raycore.closest_hit(tlas, ray)
        # @info result_closest
        # return result_closest
        if hit_found && distance > 1f-4
            thicknesses[i] = distance
            midpoints[i] = pos .+ (ray_dir * (distance / 2.0f0))
        else
            thicknesses[i] = 0.0f0
            midpoints[i] = pos
        end

    end
    
    
    return (;thicknesses, midpoints)
end


function draw_midpoints(meshdata, thickness_data)
    thicknesses, midpoints = thickness_data
    mesh = meshdata.mesh

    fig = Figure()
    ax = Axis3(fig[1,1])

    mesh!(ax, mesh, color = :grey, alpha = 0.5)
    wireframe!(ax, mesh, color = :black)
    scatter!(ax, midpoints, markersize = thicknesses)
    return fig
end