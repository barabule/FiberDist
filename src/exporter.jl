

"""
    export_fea_csv(filepath::AbstractString, md::MeshData, st;
                   vertex_fill_time::Union{Nothing, AbstractVector{<:Real}} = nothing,
                   include_fiber_tensor::Bool = true)

Exports per-triangle centroid coordinates, normals, fill time, flow direction,
diagnostic masks (weld lines, air traps), and an optional 2nd-order Advani-Tucker
orientation tensor estimate (`a11..a23`) for FEA mapping (Ansys External Data,
Digimat-MAP, Converse, Abaqus/LS-DYNA scripts).
"""
function export_fea_csv(
    filepath::AbstractString,
    md::MeshData,
    st;
    vertex_fill_time::Union{Nothing, AbstractVector{<:Real}} = nothing,
    include_fiber_tensor::Bool = true
)
    centroids = st.centroids
    normals   = st.normals
    fill_time = st.fill_time
    flow_dir  = st.flow_dir
    weld_mask = st.masks.weld_lines
    air_mask  = st.masks.air_traps

    open(filepath, "w") do io
        if include_fiber_tensor
            println(io, "elem_id,x,y,z,nx,ny,nz,fill_time,flow_x,flow_y,flow_z,weld_line,air_trap,a11,a22,a33,a12,a13,a23")
        else
            println(io, "elem_id,x,y,z,nx,ny,nz,fill_time,flow_x,flow_y,flow_z,weld_line,air_trap")
        end

        @inbounds for i in eachindex(centroids)
            c  = centroids[i]
            n  = normals[i]
            u  = flow_dir[i]
            ft = Float32(fill_time[i])
            wl = weld_mask[i] ? 1 : 0
            at = air_mask[i]  ? 1 : 0

            # Normalize flow vector for clean orientation export
            unorm = sqrt(u[1]^2 + u[2]^2 + u[3]^2)
            ux, uy, uz = unorm > 1f-6 ? (u[1]/unorm, u[2]/unorm, u[3]/unorm) : (0.0f0, 0.0f0, 0.0f0)

            if include_fiber_tensor
                # Mildly aligned skin-core effective tensor (0.75 along flow + 0.25 isotropic planar)
                # At weld lines, collapse alignment toward transverse/planar isotropic
                w_align = wl == 1 ? 0.15f0 : 0.70f0
                w_iso   = (1.0f0 - w_align) / 3.0f0
                a11 = w_align * ux * ux + w_iso
                a22 = w_align * uy * uy + w_iso
                a33 = w_align * uz * uz + w_iso
                a12 = w_align * ux * uy
                a13 = w_align * ux * uz
                a23 = w_align * uy * uz

                @printf(io, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    i, c[1], c[2], c[3], n[1], n[2], n[3], ft, ux, uy, uz, wl, at,
                    a11, a22, a33, a12, a13, a23)
            else
                @printf(io, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d\n",
                    i, c[1], c[2], c[3], n[1], n[2], n[3], ft, ux, uy, uz, wl, at)
            end
        end
    end
    return filepath
end

"""
    export_vtk_legacy(filepath::AbstractString, md::MeshData, st,
                      vertex_fill_time::AbstractVector{<:Real})

Writes a self-contained VTK Legacy ASCII (`POLYDATA` / `UNSTRUCTURED_GRID`) file
containing surface triangle connectivity, nodal `fill_time`, and elemental
`fill_time`, `flow_direction`, `weld_lines`, and `air_traps`.
"""
function export_vtk_legacy(
    filepath::AbstractString,
    md::MeshData,
    st,
    vertex_fill_time::AbstractVector{<:Real}
)
    pts   = GeometryBasics.coordinates(md.mesh)
    faces = GeometryBasics.faces(md.mesh)
    n_pts   = length(pts)
    n_faces = length(faces)

    open(filepath, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, "Injection Molding Fill & Flow Analysis")
        println(io, "ASCII")
        println(io, "DATASET UNSTRUCTURED_GRID")

        println(io, "POINTS $n_pts float")
        @inbounds for p in pts
            @printf(io, "%.6f %.6f %.6f\n", p[1], p[2], p[3])
        end

        println(io, "CELLS $n_faces $(4 * n_faces)")
        @inbounds for f in faces
            # VTK uses 0-based vertex indexing
            println(io, "3 $(Int(f[1]) - 1) $(Int(f[2]) - 1) $(Int(f[3]) - 1)")
        end

        println(io, "CELL_TYPES $n_faces")
        for _ in 1:n_faces
            println(io, "5") # VTK_TRIANGLE
        end

        # Elemental (Cell) Fields
        println(io, "CELL_DATA $n_faces")
        println(io, "SCALARS fill_time_elem float 1")
        println(io, "LOOKUP_TABLE default")
        @inbounds for v in st.fill_time
            @printf(io, "%.6f\n", Float32(v))
        end

        println(io, "SCALARS weld_line int 1")
        println(io, "LOOKUP_TABLE default")
        @inbounds for w in st.masks.weld_lines
            println(io, w ? 1 : 0)
        end

        println(io, "SCALARS air_trap int 1")
        println(io, "LOOKUP_TABLE default")
        @inbounds for a in st.masks.air_traps
            println(io, a ? 1 : 0)
        end

        println(io, "VECTORS flow_direction float")
        @inbounds for u in st.flow_dir
            @printf(io, "%.6f %.6f %.6f\n", u[1], u[2], u[3])
        end

        # Nodal (Point) Fields
        if length(vertex_fill_time) == n_pts
            println(io, "POINT_DATA $n_pts")
            println(io, "SCALARS fill_time_node float 1")
            println(io, "LOOKUP_TABLE default")
            @inbounds for v in vertex_fill_time
                @printf(io, "%.6f\n", Float32(v))
            end
        end
    end
    return filepath
end

"""
    export_moldflow_ele(basepath::AbstractString, md::MeshData, st)

Writes Moldflow-compatible ASCII `.pat` (Patran Neutral mesh) and `.ele`
(elemental orientation tensor & weld line quality) files for Digimat-MAP / Converse.
"""
function export_moldflow_ele(basepath::AbstractString, md::MeshData, st)
    pts   = GeometryBasics.coordinates(md.mesh)
    faces = GeometryBasics.faces(md.mesh)

    pat_path = basepath * ".pat"
    ele_path = basepath * ".ele"

    open(pat_path, "w") do io
        @printf(io, "25       0       0       1       0       0       0       0       0\n")
        @printf(io, "Mold Simulation Surface Mesh Export\n")
        @printf(io, "26       0       0       1%8d%8d       0       0       0\n", length(pts), length(faces))
        @printf(io, "01-JAN-26   00:00:00      0.000000\n")
        @inbounds for (i, p) in enumerate(pts)
            @printf(io, " 1%8d       0       2       0       0       0       0       0\n", i)
            @printf(io, "%16.9E%16.9E%16.9E\n", p[1], p[2], p[3])
            @printf(io, "1G       6       0       0  000000\n")
        end
        @inbounds for (i, f) in enumerate(faces)
            @printf(io, " 2%8d       3       2       0       0       0       0       0\n", i)
            @printf(io, "       3       0       0       0 0.000000000E+00 0.000000000E+00 0.000000000E+00\n")
            @printf(io, "%8d%8d%8d\n", Int(f[1]), Int(f[2]), Int(f[3]))
        end
        @printf(io, "99       0       0       1       0       0       0       0       0\n")
    end

    open(ele_path, "w") do io
        println(io, "Fiber orientation tensor on elements")
        println(io, "1")
        println(io, "0.0")
        println(io, length(faces))
        @inbounds for i in eachindex(faces)
            u = st.flow_dir[i]
            unorm = sqrt(u[1]^2 + u[2]^2 + u[3]^2)
            ux, uy, uz = unorm > 1f-6 ? (u[1]/unorm, u[2]/unorm, u[3]/unorm) : (1.0f0, 0.0f0, 0.0f0)
            w_align = st.masks.weld_lines[i] ? 0.15f0 : 0.70f0
            w_iso   = (1.0f0 - w_align) / 3.0f0
            a11 = w_align * ux * ux + w_iso
            a22 = w_align * uy * uy + w_iso
            a33 = w_align * uz * uz + w_iso
            a12 = w_align * ux * uy
            a23 = w_align * uy * uz
            a13 = w_align * ux * uz
            @printf(io, "%8d %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n",
                i, a11, a22, a33, a12, a23, a13)
        end
    end

    return (pat = pat_path, ele = ele_path)
end