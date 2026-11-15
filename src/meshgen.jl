

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




"""
    build_cone_directions(n::SVector{3, Float32})

Generates a deterministic 17-ray cone pointing into the interior hemisphere (-n):
1 axial ray along -n, plus 2 rings of 8 rays at 30° and 60° off-axis.
"""
function build_cone_directions(n::SVector{3, Float32})
    inward = -n
    # Construct an orthonormal basis (t1, t2, inward)
    ref = abs(inward[1]) < 0.9f0 ? SVector(1.0f0, 0.0f0, 0.0f0) : SVector(0.0f0, 1.0f0, 0.0f0)
    t1 = normalize(cross(inward, ref))
    t2 = cross(inward, t1)

    dirs = MVector{17, SVector{3, Float32}}(undef)
    dirs[1] = inward

    idx = 2
    for tilt_deg in (30.0f0, 60.0f0)
        θ = deg2rad(tilt_deg)
        sin_θ, cos_θ = sincos(θ)
        for k in 0:7
            ϕ = Float32(k) * (2.0f0 * Float32(π) / 8.0f0)
            sin_ϕ, cos_ϕ = sincos(ϕ)
            dirs[idx] = normalize(cos_θ * inward + sin_θ * (cos_ϕ * t1 + sin_ϕ * t2))
            idx += 1
        end
    end
    return SVector(dirs)
end

"""
    compute_triangle_thicknesses(bvh, points, elements; eps_offset=1e-4f0)

Computes local wall thickness and mid-surface points at every triangle centroid
using the Shrinking Tangent Sphere method via Raycore.
"""
function compute_triangle_thicknesses(meshdata; eps_offset::Float32 = 1f-4)
    # bvh, points::Vector{Point3f}, elements::AbstractMatrix
    bvh, mesh = meshdata.tlas, meshdata.mesh

    n_tris = length(mesh.faces)
    thicknesses = zeros(Float32, n_tris)
    midpoints   = Vector{Point3f}(undef, n_tris)
    centroids   = Vector{Point3f}(undef, n_tris)
    normals     = Vector{SVector{3, Float32}}(undef, n_tris)

    for i in 1:n_tris
        i1, i2, i3 = mesh.faces[i]
        p1 = mesh.position[i1]
        p2 = mesh.position[i2]
        p3 = mesh.position[i3]

        c = (p1 + p2 + p3) / 3.0f0
        raw_n = cross(p2 - p1, p3 - p1)
        norm_len = norm(raw_n)

        # Guard against degenerate zero-area triangles
        if norm_len < 1f-12
            centroids[i]   = Point3f(c)
            normals[i]     = SVector(0.0f0, 0.0f0, 1.0f0)
            midpoints[i]   = Point3f(c)
            thicknesses[i] = 0.0f0
            continue
        end

        n = raw_n / norm_len
        centroids[i] = c
        normals[i]   = n

        # Ensure Gmsh outward normals point out of the solid;
        # offset origin slightly inward along -n to avoid self-hitting triangle i
        origin = c - n * eps_offset
        cone_dirs = build_cone_directions(n)

        min_diameter = Inf32

        for dir in cone_dirs
            ray = Raycore.Ray(o = origin, d = dir)
            hit_found, _, dist, _, _ = Raycore.closest_hit(bvh, ray)

            if hit_found && dist > eps_offset
                # Reconstruct hit point q in 3D space
                q = SVector{3, Float32}(origin) + dir * dist
                pq = c - q
                denom = dot(pq, n)

                # Only consider hits strictly behind the tangent plane (denom > 0)
                if denom > 1f-7
                    # Exact diameter of sphere tangent to (c, n) passing through q
                    d_sphere = dot(pq, pq) / denom
                    if d_sphere < min_diameter
                        min_diameter = d_sphere
                    end
                end
            end
        end

        if isfinite(min_diameter)
            thicknesses[i] = min_diameter
            midpoints[i]   = Point3f(c - n * (0.5f0 * min_diameter))
        else
            thicknesses[i] = 0.0f0
            midpoints[i]   = Point3f(c)
        end
    end

    return (;thicknesses, midpoints, centroids, normals)
end