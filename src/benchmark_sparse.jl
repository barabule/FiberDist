import FiberDist as FD
using GeometryBasics: Point3f
using Printf

# Analytical geometry helper for a 1000mm x 300mm x 50mm part:
# - 1000 x 300 x 2.5 mm base plate (z in [0, 2.5])
# - Longitudinal & transverse 1.0 mm thick ribs rising to z = 50 mm
function make_large_ribbed_plate_mesh()
    verts = Point3f[]
    faces = NTuple{3, Int}[]

    function add_box!(x0, y0, z0, x1, y1, z1)
        base = length(verts)
        push!(verts,
            Point3f(x0, y0, z0), Point3f(x1, y0, z0), Point3f(x1, y1, z0), Point3f(x0, y1, z0),
            Point3f(x0, y0, z1), Point3f(x1, y0, z1), Point3f(x1, y1, z1), Point3f(x0, y1, z1)
        )
        append!(faces, [
            (base+1, base+3, base+2), (base+1, base+4, base+3),
            (base+5, base+6, base+7), (base+5, base+7, base+8),
            (base+1, base+2, base+6), (base+1, base+6, base+5),
            (base+3, base+4, base+8), (base+3, base+8, base+7),
            (base+1, base+5, base+8), (base+1, base+8, base+4),
            (base+2, base+3, base+7), (base+2, base+7, base+6)
        ])
    end

    # Base plate: 1000 x 300 x 2.5 mm
    add_box!(0f0, 0f0, 0f0, 1000f0, 300f0, 2.5f0)
    # 3 longitudinal 1.0 mm ribs (y = 75, 150, 225 mm, height up to 50 mm)
    for yr in (75f0, 150f0, 225f0)
        add_box!(10f0, yr - 0.5f0, 2.5f0, 990f0, yr + 0.5f0, 50f0)
    end
    # 5 transverse 1.0 mm ribs
    for xr in (200f0, 400f0, 600f0, 800f0)
        add_box!(xr - 0.5f0, 10f0, 2.5f0, xr + 0.5f0, 290f0, 50f0)
    end

    is_inside(p::Point3f) = begin
        x, y, z = p[1], p[2], p[3]
        in_plate = (0f0 <= x <= 1000f0) && (0f0 <= y <= 300f0) && (0f0 <= z <= 2.5f0)
        in_long_rib = (10f0 <= x <= 990f0) && (2.5f0 <= z <= 50f0) &&
                      (abs(y - 75f0) <= 0.5f0 || abs(y - 150f0) <= 0.5f0 || abs(y - 225f0) <= 0.5f0)
        in_trans_rib = (10f0 <= y <= 290f0) && (2.5f0 <= z <= 50f0) &&
                       (abs(x - 200f0) <= 0.5f0 || abs(x - 400f0) <= 0.5f0 || abs(x - 600f0) <= 0.5f0 || abs(x - 800f0) <= 0.5f0)
        return in_plate || in_long_rib || in_trans_rib
    end

    return verts, faces, is_inside
end

verts, faces, is_inside = make_large_ribbed_plate_mesh()
gates = [Point3f(250f0, 150f0, 1.25f0), Point3f(750f0, 150f0, 1.25f0)]

# Warmup JIT at dx = 2.0 mm
FD.voxelize_and_solve_sparse(verts, faces, gates; dx = 2.0f0)

for dx in (1.0f0, 0.5f0, 0.3333f0)
    t_el = @elapsed grid = voxelize_and_solve_sparse(verts, faces, gates; dx = dx, max_wall_mm = 6.0f0)
    dense_vox = prod(grid.voxel_dims)
    active_bricks = length(grid.brick_coords)
    occ_vox = sum(count_ones, grid.occupancy)
    mem_mb = Base.summarysize(grid) / 1024^2
    t_proj = @elapsed tri_t, tri_w = project_to_surface_sparse(grid, verts, faces)

    @printf(
        "dx=%.4f mm | dense_box=%6.1fM vox | active_bricks=%6d | occ_vox=%7d (%.2f%%) | RAM=%6.1f MiB | solve=%.1f ms | proj=%.2f ms | tmax=%.2f\n",
        dx, dense_vox / 1e6, active_bricks, occ_vox, 100 * occ_vox / dense_vox,
        mem_mb, t_el * 1000, t_proj * 1000, maximum(filter(isfinite, grid.fill_time))
    )
end