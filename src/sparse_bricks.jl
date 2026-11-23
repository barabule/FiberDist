module SparseBrick

using GeometryBasics:Point3f, Vec3f
using Raycore
using StaticArrays: SVector
import ..FiberDist: MeshData, build_scene_bvh
using LinearAlgebra:normalize


# Slightly tilted unit vector to minimize shared edge/vertex grazing hits
const _RAY_DIR_3F = normalize(SVector{3, Float32}(0.57735026f0, 0.61237246f0, 0.5393599f0))

@inline function is_inside_raycore(
    bvh,
    pt::Point3f;
    eps_step::Float32 = 1f-4,
    dedup_tol::Float32 = 1f-3
)::Bool
    hits = 0
    last_t = -Inf32
    curr_o = pt
    traveled = 0.0f0

    while true
        ray = Raycore.Ray(o = curr_o, d = _RAY_DIR_3F)
        hit_found, _, distance, _, _ = Raycore.closest_hit(bvh, ray)
        hit_found || break

        t_hit = traveled + Float32(distance)
        if (t_hit - last_t) > dedup_tol
            hits += 1
            last_t = t_hit
        end

        step = Float32(distance) + eps_step
        traveled += step
        curr_o = pt + _RAY_DIR_3F * traveled
    end

    return isodd(hits)
end



const BRICK_DIM = 8
const BRICK_VOL = 512 # 8 * 8 * 8

"""
SparseBrickGrid 
8x8x8 VDB style layout

"""
struct SparseBrickGrid{T<:AbstractFloat}
    origin::Point3f
    dx::T
    voxel_dims::NTuple{3, Int}      # Full fine grid dims (nx, ny, nz)
    brick_dims::NTuple{3, Int}      # Macro grid dims (nbx, nby, nbz)
    brick_map::Array{Int32, 3}      # (nbx, nby, nbz) -> 1..n_bricks (0 = empty)
    brick_coords::Vector{NTuple{3, Int16}} # b -> (bx, by, bz)
    neighbors::Matrix{Int32}        # (6, n_bricks): (-x, +x, -y, +y, -z, +z) brick IDs
    occupancy::Matrix{UInt64}       # (8, n_bricks): 512-bit bitmask per brick (64 bytes/brick!)
    thick_r_sq::Matrix{UInt16}      # (512, n_bricks)
    fill_time::Matrix{Float32}      # (512, n_bricks)
end


function build_sparse_bricks(
    meshdata::MeshData,
    origin::Point3f,
    dx::Float32,
    voxel_dims::NTuple{3, Int}
)

    mesh_verts = meshdata.mesh.position
    mesh_faces = meshdata.mesh.faces
    accel = meshdata.tlas

    nx, ny, nz = voxel_dims
    nbx = cld(nx, BRICK_DIM)
    nby = cld(ny, BRICK_DIM)
    nbz = cld(nz, BRICK_DIM)
    brick_size = Float32(BRICK_DIM) * dx
    inv_brick_size = 1.0f0 / brick_size

    # 0x00 = unknown, 0x01 = intersects CAD surface, 0x02 = exterior air
    tag = zeros(UInt8, nbx, nby, nbz)

    # Pass A: Bin triangles into macro-bricks (guarantees 1mm ribs are captured)
    @inbounds for (i1, i2, i3) in mesh_faces
        p1, p2, p3 = mesh_verts[i1], mesh_verts[i2], mesh_verts[i3]
        min_x = min(p1[1], p2[1], p3[1]) - origin[1]
        min_y = min(p1[2], p2[2], p3[2]) - origin[2]
        min_z = min(p1[3], p2[3], p3[3]) - origin[3]
        max_x = max(p1[1], p2[1], p3[1]) - origin[1]
        max_y = max(p1[2], p2[2], p3[2]) - origin[2]
        max_z = max(p1[3], p2[3], p3[3]) - origin[3]

        bx0 = clamp(floor(Int, min_x * inv_brick_size) + 1, 1, nbx)
        bx1 = clamp(floor(Int, max_x * inv_brick_size) + 1, 1, nbx)
        by0 = clamp(floor(Int, min_y * inv_brick_size) + 1, 1, nby)
        by1 = clamp(floor(Int, max_y * inv_brick_size) + 1, 1, nby)
        bz0 = clamp(floor(Int, min_z * inv_brick_size) + 1, 1, nbz)
        bz1 = clamp(floor(Int, max_z * inv_brick_size) + 1, 1, nbz)

        for bz in bz0:bz1, by in by0:by1, bx in bx0:bx1
            tag[bx, by, bz] = 0x01
        end
    end

    # Pass B: Flood-fill exterior air on macro-grid (pad by 1 layer conceptually)
    queue = Tuple{Int16, Int16, Int16}[]
    sizehint!(queue, nbx * nby * 2)
    @inbounds for bz in 1:nbz, by in 1:nby, bx in 1:nbx
        if (bx == 1 || bx == nbx || by == 1 || by == nby || bz == 1 || bz == nbz) && tag[bx, by, bz] == 0x00
            tag[bx, by, bz] = 0x02
            push!(queue, (Int16(bx), Int16(by), Int16(bz)))
        end
    end

    head = 1
    @inbounds while head <= length(queue)
        bx, by, bz = queue[head]
        head += 1
        for (dx_b, dy_b, dz_b) in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1))
            nx_b, ny_b, nz_b = bx + dx_b, by + dy_b, bz + dz_b
            if 1 <= nx_b <= nbx && 1 <= ny_b <= nby && 1 <= nz_b <= nbz && tag[nx_b, ny_b, nz_b] == 0x00
                tag[nx_b, ny_b, nz_b] = 0x02
                push!(queue, (Int16(nx_b), Int16(ny_b), Int16(nz_b)))
            end
        end
    end

    # Allocate dense 8x8x8 bricks ONLY for surface (0x01) and interior (0x00) macro-cells
    brick_map = zeros(Int32, nbx, nby, nbz)
    brick_coords = NTuple{3, Int16}[]
    brick_is_interior = Bool[]

    @inbounds for bz in 1:nbz, by in 1:nby, bx in 1:nbx
        if tag[bx, by, bz] != 0x02
            push!(brick_coords, (Int16(bx), Int16(by), Int16(bz)))
            brick_map[bx, by, bz] = Int32(length(brick_coords))
            push!(brick_is_interior, tag[bx, by, bz] == 0x00)
        end
    end

    n_bricks = length(brick_coords)
    neighbors = zeros(Int32, 6, n_bricks)
    occupancy = zeros(UInt64, BRICK_DIM, n_bricks)

    # Precompute 6-neighbor brick IDs & fill interior bricks with 0xFFFF...
    Threads.@threads :static for b in 1:n_bricks
        @inbounds begin
            bx, by, bz = brick_coords[b]
            bx > 1   && (neighbors[1, b] = brick_map[bx - 1, by, bz])
            bx < nbx && (neighbors[2, b] = brick_map[bx + 1, by, bz])
            by > 1   && (neighbors[3, b] = brick_map[bx, by - 1, bz])
            by < nby && (neighbors[4, b] = brick_map[bx, by + 1, bz])
            bz > 1   && (neighbors[5, b] = brick_map[bx, by, bz - 1])
            bz < nbz && (neighbors[6, b] = brick_map[bx, by, bz + 1])

            if brick_is_interior[b]
                for lk in 1:BRICK_DIM
                    occupancy[lk, b] = typemax(UInt64)
                end
            end
        end
    end

    # Pass C: Column-wise Z-ray voxelization ONLY across (bx, by) columns that contain surface bricks
    z_start   = origin[3] - dx
    ray_dir   = SVector{3, Float32}(0.0f0, 0.0f0, 1.0f0)
    eps_step  = dx * 1f-3
    dedup_tol = dx * 1f-2

    # Collect (bx, by) brick columns that have at least one non-interior surface brick
    surface_cols = Tuple{Int16, Int16}[]
    for by in 1:nby, bx in 1:nbx
        has_surface = false
        for bz in 1:nbz
            b = brick_map[bx, by, bz]
            if b != 0 && !brick_is_interior[b]
                has_surface = true
                break
            end
        end
        has_surface && push!(surface_cols, (Int16(bx), Int16(by)))
    end

    Threads.@threads :static for col_idx in eachindex(surface_cols)
        hits = Float32[]
        clean_hits = Float32[]
        bx, by = surface_cols[col_idx]
        i0 = (Int(bx) - 1) * BRICK_DIM
        j0 = (Int(by) - 1) * BRICK_DIM

        for lj in 0:(BRICK_DIM - 1)
            gj = j0 + lj + 1
            gj > ny && break
            y = origin[2] + (Float32(gj) - 0.5f0) * dx

            for li in 0:(BRICK_DIM - 1)
                gi = i0 + li + 1
                gi > nx && break
                x = origin[1] + (Float32(gi) - 0.5f0) * dx

                empty!(hits)
                curr_z = z_start
                while true
                    ray = Raycore.Ray(o = Point3f(x, y, curr_z), d = ray_dir)
                    hit_found, _, distance, _, _ = Raycore.closest_hit(accel, ray)
                    hit_found || break

                    z_hit = curr_z + Float32(distance)
                    push!(hits, z_hit)
                    curr_z = z_hit + eps_step
                end

                length(hits) < 2 && continue
                sort!(hits)

                empty!(clean_hits)
                push!(clean_hits, hits[1])
                @inbounds for h_idx in 2:length(hits)
                    if (hits[h_idx] - clean_hits[end]) > dedup_tol
                        push!(clean_hits, hits[h_idx])
                    end
                end

                bit_xy = UInt64(1) << (li + (lj << 3))
                n_clean = length(clean_hits)
                @inbounds for m in 1:2:(n_clean - 1)
                    z_enter = clean_hits[m]
                    z_exit  = clean_hits[m + 1]

                    k_min = clamp(ceil(Int,  (z_enter - origin[3]) / dx + 0.5f0), 2, nz - 1)
                    k_max = clamp(floor(Int, (z_exit  - origin[3]) / dx + 0.5f0), 2, nz - 1)

                    for gk in k_min:k_max
                        bz = cld(gk, BRICK_DIM)
                        b = brick_map[bx, by, bz]
                        if b != 0 && !brick_is_interior[b]
                            lk = (gk - 1) & 7
                            occupancy[lk + 1, b] |= bit_xy
                        end
                    end
                end
            end
        end
    end

    return SparseBrickGrid{Float32}(
        origin, dx, voxel_dims, (nbx, nby, nbz),
        brick_map, brick_coords, neighbors, occupancy,
        zeros(UInt16, BRICK_VOL, n_bricks),
        fill(Inf32, BRICK_VOL, n_bricks)
    )
end


@inline function _brick_is_occupied(occupancy::Matrix{UInt64}, b::Int, li::Int, lj::Int, lk::Int)::Bool
    @inbounds return ((occupancy[lk + 1, b] >> (li + (lj << 3))) & 0x1) != 0x0
end

@inline function _brick_lin_idx(li::Int, lj::Int, lk::Int)::Int
    return li + (lj << 3) + (lk << 6) + 1 # 1..512
end

"""
    compute_brick_edt!(grid::SparseBrickGrid; max_wall_radius_vox::Int = 24)

Computes exact squared Euclidean distance to the nearest mold wall (`thick_r_sq`)
for every occupied voxel in `grid` using a 3-pass separable parabolic EDT across
active 8×8×8 bricks (`O(n_bricks)` memory, zero dense bounding-box allocation).
"""
function compute_brick_edt!(grid::SparseBrickGrid; max_wall_radius_vox::Int = 24)
    n_bricks = length(grid.brick_coords)
    n_bricks == 0 && return grid

    max_chain_bricks = cld(max_wall_radius_vox, BRICK_DIM) + 1
    inf_d2 = Float32(max_wall_radius_vox * max_wall_radius_vox * 4)

    # Scratch buffer: squared distance per voxel (512 × n_bricks)
    d2 = fill(inf_d2, BRICK_VOL, n_bricks)
    neighbors = grid.neighbors
    occupancy = grid.occupancy

    # Initialize: empty air voxels = 0.0f0, occupied fluid voxels = inf_d2
    Threads.@threads :static for b in 1:n_bricks
        @inbounds for lk in 0:7
            row_mask = occupancy[lk + 1, b]
            for lj in 0:7, li in 0:7
                bit = (row_mask >> (li + (lj << 3))) & 0x1
                d2[_brick_lin_idx(li, lj, lk), b] = bit == 0x0 ? 0.0f0 : inf_d2
            end
        end
    end

    # --- Pass 1: Exact 1D distance along X across brick chains ---
    Threads.@threads :static for b in 1:n_bricks
        @inbounds for lk in 0:7, lj in 0:7
            # Seed from -X neighbor chain (up to max_chain_bricks back)
            prev_d = 0.0f0
            cur_b = b
            steps_back = 0
            for _ in 1:max_chain_bricks
                nb = neighbors[1, cur_b]
                if nb == 0
                    prev_d = 0.0f0
                    break
                end
                cur_b = nb
                steps_back += 1
            end
            # Walk forward from cur_b -> b to propagate exact X-distance into brick b
            for step in 0:steps_back
                for li in 0:7
                    idx = _brick_lin_idx(li, lj, lk)
                    if d2[idx, cur_b] == 0.0f0
                        prev_d = 0.0f0
                    else
                        prev_d += 1.0f0
                        if cur_b == b
                            d2[idx, b] = prev_d * prev_d
                        end
                    end
                end
                cur_b = neighbors[2, cur_b]
                cur_b == 0 && break
            end

            # Backward pass from +X neighbor chain into brick b
            next_d = 0.0f0
            cur_b = b
            steps_fwd = 0
            for _ in 1:max_chain_bricks
                nb = neighbors[2, cur_b]
                if nb == 0
                    next_d = 0.0f0
                    break
                end
                cur_b = nb
                steps_fwd += 1
            end
            for step in 0:steps_fwd
                for li in 7:-1:0
                    idx = _brick_lin_idx(li, lj, lk)
                    if d2[idx, cur_b] == 0.0f0
                        next_d = 0.0f0
                    else
                        next_d += 1.0f0
                        if cur_b == b
                            val2 = next_d * next_d
                            val2 < d2[idx, b] && (d2[idx, b] = val2)
                        end
                    end
                end
                cur_b = neighbors[1, cur_b]
                cur_b == 0 && break
            end
        end
    end

    # --- Pass 2 & 3: Separable 1D Parabolic Envelope along Y then Z ---
    # Per-thread scratch for Felzenszwalb lower envelope over up to (2*max_chain_bricks + 1)*8 samples
    max_len = (2 * max_chain_bricks + 1) * BRICK_DIM
    n_tid = Threads.maxthreadid()
    f_bufs = [Vector{Float32}(undef, max_len) for _ in 1:n_tid]
    v_bufs = [Vector{Int32}(undef, max_len)   for _ in 1:n_tid]
    z_bufs = [Vector{Float32}(undef, max_len + 1) for _ in 1:n_tid]

    @inline function _parabolic_1d!(f::Vector{Float32}, v::Vector{Int32}, z::Vector{Float32}, n::Int)
        @inbounds begin
            k = 1
            v[1] = 1
            z[1] = -Inf32
            z[2] = Inf32
            for q in 2:n
                fq = f[q]
                s = ((fq + Float32(q * q)) - (f[v[k]] + Float32(v[k] * v[k]))) / Float32(2 * (q - v[k]))
                while s <= z[k]
                    k -= 1
                    s = ((fq + Float32(q * q)) - (f[v[k]] + Float32(v[k] * v[k]))) / Float32(2 * (q - v[k]))
                end
                k += 1
                v[k] = Int32(q)
                z[k] = s
                z[k + 1] = Inf32
            end
        end
    end

    # Pass 2: Y-axis
    d2_y = similar(d2)
    Threads.@threads :static for b in 1:n_bricks
        tid = Threads.threadid()
        f, v, z = f_bufs[tid], v_bufs[tid], z_bufs[tid]
        @inbounds for lk in 0:7, li in 0:7
            # Collect -Y bricks
            left_bricks = max_chain_bricks
            cur_b = b
            actual_left = 0
            for _ in 1:max_chain_bricks
                nb = neighbors[3, cur_b]
                nb == 0 && break
                cur_b = nb
                actual_left += 1
            end
            # Fill f[1:len] walking +Y from cur_b
            pos = 1
            f[pos] = 0.0f0 # exterior boundary at -Y end
            pos += 1
            walk_b = cur_b
            b_start_pos = 0
            for _ in 0:(actual_left + max_chain_bricks)
                walk_b == b && (b_start_pos = pos)
                for lj in 0:7
                    f[pos] = d2[_brick_lin_idx(li, lj, lk), walk_b]
                    pos += 1
                end
                walk_b = neighbors[4, walk_b]
                walk_b == 0 && break
            end
            f[pos] = 0.0f0 # exterior boundary at +Y end
            len = pos

            _parabolic_1d!(f, v, z, len)
            k = 1
            for lj in 0:7
                q = b_start_pos + lj
                while z[k + 1] < Float32(q)
                    k += 1
                end
                dq = Float32(q - v[k])
                d2_y[_brick_lin_idx(li, lj, lk), b] = dq * dq + f[v[k]]
            end
        end
    end

    # Pass 3: Z-axis -> write UInt16 thick_r_sq
    thick_r_sq = grid.thick_r_sq
    Threads.@threads :static for b in 1:n_bricks
        tid = Threads.threadid()
        f, v, z = f_bufs[tid], v_bufs[tid], z_bufs[tid]
        @inbounds for lj in 0:7, li in 0:7
            cur_b = b
            actual_left = 0
            for _ in 1:max_chain_bricks
                nb = neighbors[5, cur_b]
                nb == 0 && break
                cur_b = nb
                actual_left += 1
            end
            pos = 1
            f[pos] = 0.0f0
            pos += 1
            walk_b = cur_b
            b_start_pos = 0
            for _ in 0:(actual_left + max_chain_bricks)
                walk_b == b && (b_start_pos = pos)
                for lk in 0:7
                    f[pos] = d2_y[_brick_lin_idx(li, lj, lk), walk_b]
                    pos += 1
                end
                walk_b = neighbors[6, walk_b]
                walk_b == 0 && break
            end
            f[pos] = 0.0f0
            len = pos

            _parabolic_1d!(f, v, z, len)
            k = 1
            for lk in 0:7
                q = b_start_pos + lk
                while z[k + 1] < Float32(q)
                    k += 1
                end
                dq = Float32(q - v[k])
                val = dq * dq + f[v[k]]
                idx = _brick_lin_idx(li, lj, lk)
                thick_r_sq[idx, b] = d2[idx, b] == 0.0f0 ? UInt16(0) : UInt16(clamp(round(Int, val), 1, 65534))
            end
        end
    end

    return grid
end


@inline function _godunov_3d_isotropic(
    tx::Float32, ty::Float32, tz::Float32,
    inv_dx2::Float32, inv_F2::Float32
)::Float32
    # Sort tx <= ty <= tz
    a, b, c = tx, ty, tz
    if a > b; a, b = b, a; end
    if b > c; b, c = c, b; end
    if a > b; a, b = b, a; end

    isfinite(a) || return Inf32
    h = sqrt(inv_F2 / inv_dx2) # local step cost dx / F

    # 1D update
    t1 = a + h
    t1 <= b && return t1

    # 2D quadratic update: (t - a)^2 + (t - b)^2 = h^2
    disc2 = 2.0f0 * h * h - (b - a) * (b - a)
    t2 = disc2 >= 0.0f0 ? 0.5f0 * (a + b + sqrt(disc2)) : t1
    t2 <= c && return t2

    # 3D quadratic update: (t - a)^2 + (t - b)^2 + (t - c)^2 = h^2
    sum_abc = a + b + c
    disc3 = sum_abc * sum_abc - 3.0f0 * (a * a + b * b + c * c - h * h)
    return disc3 >= 0.0f0 ? (sum_abc + sqrt(disc3)) / 3.0f0 : t2
end

"""
    solve_eikonal_sparse!(grid::SparseBrickGrid, gate_points::AbstractVector{Point3f};
                          tol::Float32 = 1f-5, max_iters::Int = 600)

Solves the anisotropic/isotropic variable-speed Eikonal equation directly on
`grid::SparseBrickGrid` using Red-Black 8×8×8 brick coloring with in-brick
forward+backward sweeps and block-skipping wake-ups.
"""
function solve_eikonal_sparse!(
    grid::SparseBrickGrid,
    gate_points::AbstractVector{Point3f};
    tol::Float32 = 1f-5,
    max_iters::Int = 600
)
    n_bricks = length(grid.brick_coords)
    n_bricks == 0 && return grid

    dx = Float32(grid.dx)
    inv_dx2 = 1.0f0 / (dx * dx)
    base_h2 = 4.0f0 * dx * dx

    T = grid.fill_time
    fill!(T, Inf32)
    thick_r_sq = grid.thick_r_sq
    occupancy = grid.occupancy
    neighbors = grid.neighbors

    # Precompute inv_F2 per voxel (512 × n_bricks) and partition bricks into Red / Black lists
    inv_F2 = fill(Inf32, BRICK_VOL, n_bricks)
    is_gate = zeros(UInt8, BRICK_VOL, n_bricks)

    red_bricks = Int32[]
    black_bricks = Int32[]
    sizehint!(red_bricks, n_bricks ÷ 2 + 1)
    sizehint!(black_bricks, n_bricks ÷ 2 + 1)

    @inbounds for b in 1:n_bricks
        bx, by, bz = grid.brick_coords[b]
        if isodd(Int(bx) + Int(by) + Int(bz))
            push!(red_bricks, Int32(b))
        else
            push!(black_bricks, Int32(b))
        end
    end

    Threads.@threads :static for b in 1:n_bricks
        @inbounds for idx in 1:BRICK_VOL
            rsq = Float32(thick_r_sq[idx, b])
            if rsq > 0.0f0
                inv_F2[idx, b] = 1.0f0 / ((base_h2 * rsq)^2)
            end
        end
    end

    brick_active = zeros(UInt8, n_bricks)

    # Seed gate voxels (snap to nearest occupied voxel in brick if slightly off-center)
    inv_dx = 1.0f0 / dx
    nx, ny, nz = grid.voxel_dims
    @inbounds for gp in gate_points
        gi = clamp(floor(Int, (gp[1] - grid.origin[1]) * inv_dx) + 1, 1, nx)
        gj = clamp(floor(Int, (gp[2] - grid.origin[2]) * inv_dx) + 1, 1, ny)
        gk = clamp(floor(Int, (gp[3] - grid.origin[3]) * inv_dx) + 1, 1, nz)

        bx = cld(gi, BRICK_DIM)
        by = cld(gj, BRICK_DIM)
        bz = cld(gk, BRICK_DIM)
        b = grid.brick_map[bx, by, bz]
        b == 0 && continue

        li = (gi - 1) & 7
        lj = (gj - 1) & 7
        lk = (gk - 1) & 7
        idx = _brick_lin_idx(li, lj, lk)
        if thick_r_sq[idx, b] > 0
            T[idx, b] = 0.0f0
            is_gate[idx, b] = 0x01
            brick_active[b] = 0x01
            for dir in 1:6
                nb = neighbors[dir, b]
                nb > 0 && (brick_active[nb] = 0x01)
            end
        end
    end

    @inline function _sweep_brick!(b::Int)::Float32
        local_max = 0.0f0
        nb_xm = neighbors[1, b]
        nb_xp = neighbors[2, b]
        nb_ym = neighbors[3, b]
        nb_yp = neighbors[4, b]
        nb_zm = neighbors[5, b]
        nb_zp = neighbors[6, b]

        # Forward + Backward local Gauss-Seidel pass inside the 8×8×8 brick
        @inbounds for pass in 1:2
            k_range = pass == 1 ? (0:7) : (7:-1:0)
            j_range = pass == 1 ? (0:7) : (7:-1:0)
            i_range = pass == 1 ? (0:7) : (7:-1:0)

            for lk in k_range
                row_mask = occupancy[lk + 1, b]
                row_mask == 0x0 && continue
                for lj in j_range
                    for li in i_range
                        ((row_mask >> (li + (lj << 3))) & 0x1) == 0x0 && continue
                        idx = _brick_lin_idx(li, lj, lk)
                        is_gate[idx, b] != 0x00 && continue

                        # Fetch 6 neighbors (L1-resident inside brick; face-wrapped across neighbors)
                        tx_m = li > 0 ? T[idx - 1, b]  : (nb_xm > 0 ? T[idx + 7,  nb_xm] : Inf32)
                        tx_p = li < 7 ? T[idx + 1, b]  : (nb_xp > 0 ? T[idx - 7,  nb_xp] : Inf32)
                        ty_m = lj > 0 ? T[idx - 8, b]  : (nb_ym > 0 ? T[idx + 56, nb_ym] : Inf32)
                        ty_p = lj < 7 ? T[idx + 8, b]  : (nb_yp > 0 ? T[idx - 56, nb_yp] : Inf32)
                        tz_m = lk > 0 ? T[idx - 64, b] : (nb_zm > 0 ? T[idx + 448, nb_zm] : Inf32)
                        tz_p = lk < 7 ? T[idx + 64, b] : (nb_zp > 0 ? T[idx - 448, nb_zp] : Inf32)

                        t_old = T[idx, b]
                        t_new = _godunov_3d_isotropic(
                            min(tx_m, tx_p), min(ty_m, ty_p), min(tz_m, tz_p),
                            inv_dx2, inv_F2[idx, b]
                        )
                        if t_new < t_old
                            T[idx, b] = t_new
                            diff = t_old - t_new
                            diff > local_max && (local_max = diff)

                            # Wake adjacent brick if a boundary voxel improved
                            li == 0 && nb_xm > 0 && T[idx + 7,   nb_xm] > t_new && (brick_active[nb_xm] = 0x01)
                            li == 7 && nb_xp > 0 && T[idx - 7,   nb_xp] > t_new && (brick_active[nb_xp] = 0x01)
                            lj == 0 && nb_ym > 0 && T[idx + 56,  nb_ym] > t_new && (brick_active[nb_ym] = 0x01)
                            lj == 7 && nb_yp > 0 && T[idx - 56,  nb_yp] > t_new && (brick_active[nb_yp] = 0x01)
                            lk == 0 && nb_zm > 0 && T[idx + 448, nb_zm] > t_new && (brick_active[nb_zm] = 0x01)
                            lk == 7 && nb_zp > 0 && T[idx - 448, nb_zp] > t_new && (brick_active[nb_zp] = 0x01)
                        end
                    end
                end
            end
        end

        return local_max
    end

    thread_max_diff = zeros(Float32, Threads.maxthreadid())

    for _ in 1:max_iters
        fill!(thread_max_diff, 0.0f0)

        # 1. Red bricks (completely race-free in parallel; wakes Black face neighbors)
        Threads.@threads :static for r in eachindex(red_bricks)
            @inbounds begin
                b = Int(red_bricks[r])
                if brick_active[b] != 0x00
                    tid = Threads.threadid()
                    diff = _sweep_brick!(b)
                    brick_active[b] = diff > tol ? 0x01 : 0x00
                    diff > thread_max_diff[tid] && (thread_max_diff[tid] = diff)
                end
            end
        end

        # 2. Black bricks (completely race-free in parallel; wakes Red face neighbors)
        Threads.@threads :static for k in eachindex(black_bricks)
            @inbounds begin
                b = Int(black_bricks[k])
                if brick_active[b] != 0x00
                    tid = Threads.threadid()
                    diff = _sweep_brick!(b)
                    brick_active[b] = diff > tol ? 0x01 : 0x00
                    diff > thread_max_diff[tid] && (thread_max_diff[tid] = diff)
                end
            end
        end

        maximum(thread_max_diff) <= tol && break
    end

    return grid
end

@inline function query_sparse_point(grid::SparseBrickGrid, p::Point3f)
    inv_dx = 1.0f0 / Float32(grid.dx)
    nx, ny, nz = grid.voxel_dims
    gi = clamp(floor(Int, (p[1] - grid.origin[1]) * inv_dx) + 1, 1, nx)
    gj = clamp(floor(Int, (p[2] - grid.origin[2]) * inv_dx) + 1, 1, ny)
    gk = clamp(floor(Int, (p[3] - grid.origin[3]) * inv_dx) + 1, 1, nz)

    b = grid.brick_map[cld(gi, BRICK_DIM), cld(gj, BRICK_DIM), cld(gk, BRICK_DIM)]
    b == 0 && return (Inf32, UInt16(0))
    idx = _brick_lin_idx((gi - 1) & 7, (gj - 1) & 7, (gk - 1) & 7)
    @inbounds return (grid.fill_time[idx, b], grid.thick_r_sq[idx, b])
end

"""
    snap_gate_to_sparse_grid(grid::SparseBrickGrid, gp::Point3f; search_radius_vox::Int = 4) -> Point3f

Snaps a world-space gate coordinate `gp` to the nearest occupied voxel inside `grid`
(useful when a gate sits on a 1 mm thin rib or surface face).
"""
function snap_gate_to_sparse_grid(
    grid::SparseBrickGrid,
    gp::Point3f;
    search_radius_vox::Int = 4
)::Point3f
    dx = Float32(grid.dx)
    inv_dx = 1.0f0 / dx
    nx, ny, nz = grid.voxel_dims

    gi0 = clamp(floor(Int, (gp[1] - grid.origin[1]) * inv_dx) + 1, 1, nx)
    gj0 = clamp(floor(Int, (gp[2] - grid.origin[2]) * inv_dx) + 1, 1, ny)
    gk0 = clamp(floor(Int, (gp[3] - grid.origin[3]) * inv_dx) + 1, 1, nz)

    best_d2 = typemax(Int)
    best_ijk = (gi0, gj0, gk0)

    @inbounds for dk in -search_radius_vox:search_radius_vox
        gk = gk0 + dk
        (1 <= gk <= nz) || continue
        bz = cld(gk, BRICK_DIM)
        lk = (gk - 1) & 7

        for dj in -search_radius_vox:search_radius_vox
            gj = gj0 + dj
            (1 <= gj <= ny) || continue
            by = cld(gj, BRICK_DIM)
            lj = (gj - 1) & 7

            for di in -search_radius_vox:search_radius_vox
                gi = gi0 + di
                (1 <= gi <= nx) || continue
                bx = cld(gi, BRICK_DIM)
                b = grid.brick_map[bx, by, bz]
                b == 0 && continue

                li = (gi - 1) & 7
                if _brick_is_occupied(grid.occupancy, b, li, lj, lk)
                    d2 = di * di + dj * dj + dk * dk
                    if d2 < best_d2
                        best_d2 = d2
                        best_ijk = (gi, gj, gk)
                        d2 == 0 && break
                    end
                end
            end
        end
    end

    return Point3f(
        grid.origin[1] + (best_ijk[1] - 0.5f0) * dx,
        grid.origin[2] + (best_ijk[2] - 0.5f0) * dx,
        grid.origin[3] + (best_ijk[3] - 0.5f0) * dx
    )
end

"""
    voxelize_and_solve_sparse(
    meshdata::MeshData,
    gate_points::AbstractVector{Point3f};
    dx::Float32 = 0.5f0,
    max_wall_mm::Float32 = 8.0f0,
    tol::Float32 = 1f-5,
    max_iters::Int = 800
) where {F}

Full sparse-brick pipeline:
1. Sparse BVH/triangle-guided brick allocation (`8×8×8` VDB layout)
2. Separable 3-pass Brick Parabolic EDT (`thick_r_sq`)
3. Block-skipping Red-Black Brick Eikonal solver (`fill_time`)
"""
function voxelize_and_solve_sparse(
    meshdata::MeshData,
    gate_points::AbstractVector{Point3f};
    dx::Float32 = 0.5f0,
    max_wall_mm::Float32 = 8.0f0,
    tol::Float32 = 1f-5,
    max_iters::Int = 800
) 
    min_pt = Point3f(Inf32, Inf32, Inf32)
    max_pt = Point3f(-Inf32, -Inf32, -Inf32)
    @inbounds for v in mesh_verts
        min_pt = min.(min_pt, v)
        max_pt = max.(max_pt, v)
    end

    origin = min_pt - Vec3f(dx, dx, dx)
    span = (max_pt + Vec3f(dx, dx, dx)) - origin
    voxel_dims = (
        max(1, ceil(Int, span[1] / dx)),
        max(1, ceil(Int, span[2] / dx)),
        max(1, ceil(Int, span[3] / dx))
    )

    #TODO this has the wrong signature
    # grid = build_sparse_bricks(mesh_verts, mesh_faces, is_inside_fn, origin, dx, voxel_dims)

    grid = build_sparse_bricks(meshdata, origin, dx, voxel_dims)


    max_wall_radius_vox = max(4, ceil(Int, (0.5f0 * max_wall_mm) / dx) + 2)
    compute_brick_edt!(grid; max_wall_radius_vox = max_wall_radius_vox)

    snapped_gates = [snap_gate_to_sparse_grid(grid, gp; search_radius_vox = 5) for gp in gate_points]
    solve_eikonal_sparse!(grid, snapped_gates; tol = tol, max_iters = max_iters)

    return grid
end

function voxelize_and_solve_sparse(
    nodes,
    elements,
    gate_points::AbstractVector{Point3f};
    kwargs...
)

    meshdata = build_scene_bvh((;nodes, elements))
    return voxelize_and_solve_sparse(
        meshdata,
        gate_points;
        kwargs...
    )
end

"""
    project_to_surface_sparse(grid::SparseBrickGrid, mesh_verts, mesh_faces)

Projects sparse brick `fill_time` and local wall thickness `2 * sqrt(thick_r_sq) * dx`
onto every CAD surface triangle by probing the triangle centroid shifted slightly
inward along the negative face normal (`-0.55 * dx`), preventing exterior air misses.
"""
function project_to_surface_sparse(
    grid::SparseBrickGrid,
    mesh_verts::AbstractVector{Point3f},
    mesh_faces::AbstractVector{NTuple{3, Int}}
)
    n_faces = length(mesh_faces)
    tri_fill_time = Vector{Float32}(undef, n_faces)
    tri_thickness = Vector{Float32}(undef, n_faces)
    dx = Float32(grid.dx)

    Threads.@threads :static for f in 1:n_faces
        @inbounds begin
            i1, i2, i3 = mesh_faces[f]
            p1, p2, p3 = mesh_verts[i1], mesh_verts[i2], mesh_verts[i3]
            centroid = (p1 + p2 + p3) * (1.0f0 / 3.0f0)

            e1 = p2 - p1
            e2 = p3 - p1
            n_vec = Vec3f(
                e1[2] * e2[3] - e1[3] * e2[2],
                e1[3] * e2[1] - e1[1] * e2[3],
                e1[1] * e2[2] - e1[2] * e2[1]
            )
            n_len = sqrt(n_vec[1]^2 + n_vec[2]^2 + n_vec[3]^2)
            inward_n = n_len > 1f-12 ? (n_vec * (-1.0f0 / n_len)) : Vec3f(0.0f0, 0.0f0, 0.0f0)

            # Probe slightly inside the part along -normal first, then centroid, then vertices
            t_val, rsq = query_sparse_point(grid, centroid + inward_n * (0.55f0 * dx))
            if !isfinite(t_val)
                t_val, rsq = query_sparse_point(grid, centroid)
            end
            if !isfinite(t_val)
                for pv in (p1, p2, p3)
                    tv, rv = query_sparse_point(grid, pv + inward_n * (0.55f0 * dx))
                    if isfinite(tv)
                        t_val, rsq = tv, rv
                        break
                    end
                end
            end

            tri_fill_time[f] = t_val
            tri_thickness[f] = rsq > 0 ? (2.0f0 * sqrt(Float32(rsq)) * dx) : 0.0f0
        end
    end

    return tri_fill_time, tri_thickness
end


export SparseBrickGrid,
       build_sparse_bricks,
       compute_brick_edt!,
       solve_eikonal_sparse!,
       voxelize_and_solve_sparse,
       project_to_surface_sparse,
       query_sparse_point


end #module SparseBrickGrid