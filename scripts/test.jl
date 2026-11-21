module TestFig

using GLMakie
using GeometryBasics


"""
Demo of a 3d plot with an overlay following the mouse

"""
function main()



fig = Figure(size = (1000, 800))

# 1. Background 3D Scene
ax3 = Axis3(fig[1, 1], title = "Click & Drag Across the Mesh")

sphere_mesh = normal_mesh(Tesselation(Sphere(Point3f(0), 1.0f0), 64))
verts = coordinates(sphere_mesh)
field_values = [sin(5 * v[1]) * cos(5 * v[2]) + v[3] for v in verts]

mesh_plot = mesh!(ax3, sphere_mesh, color = field_values, colormap = :viridis)

# 2. Dedicated Pixel-Space Overlay Scene
overlay_scene = Scene(fig.scene; clear = false, camera = campixel!)

# Outer card rect (includes padding for title and y-axis labels) and inner Axis bbox
card_rect = Observable(Rect2f(100, 100, 260, 175))
ax_bbox   = @lift(BBox(
    left($card_rect) + 38,
    right($card_rect) - 12,
    bottom($card_rect) + 14,
    top($card_rect) - 28
))

# White backing card drawn in pixel space with overdraw = true (ignores 3D depth buffer)
card_bg = poly!(
    overlay_scene,
    card_rect,
    color = RGBAf(1.0, 1.0, 1.0, 0.96),
    strokecolor = :gray60,
    strokewidth = 1.5,
    overdraw = true
)

ax_float = Axis(
    overlay_scene,
    bbox = ax_bbox,
    backgroundcolor = :white,
    xticklabelsvisible = false,
    yticklabelsvisible = true,
    yticklabelsize = 10,
    yticklabelcolor = :black,
    ytickcolor = :black,
    topspinecolor = :gray60,
    bottomspinecolor = :gray60,
    leftspinecolor = :gray60,
    rightspinecolor = :gray60,
    xgridcolor = RGBAf(0, 0, 0, 0.08),
    ygridcolor = RGBAf(0, 0, 0, 0.08),
    title = "Sampled Profile",
    titlesize = 12,
    titlecolor = :black,
    tellwidth = false,
    tellheight = false
)

for interaction in keys(interactions(ax_float))
    deregister_interaction!(ax_float, interaction)
end

# 3. Observable Data for the Floating Line Plot
x_samples = collect(range(0, 2π, length = 50))
y_samples = Observable(zeros(Float64, 50))

# overdraw = true disables depth testing against the 3D sphere
lines!(ax_float, x_samples, y_samples, color = :royalblue, linewidth = 2.5, overdraw = true)
xlims!(ax_float, 0, 2π)
ylims!(ax_float, -2.5, 2.5)

# Helper to force overdraw = true on all Axis decorations (spines, gridlines, ticks, title)
function set_overdraw_recursive!(scene::Scene)
    for p in scene.plots
        p.overdraw[] = true
    end
    for child in scene.children
        set_overdraw_recursive!(child)
    end
end
set_overdraw_recursive!(ax_float.blockscene)
set_overdraw_recursive!(ax_float.scene)

function set_overlay_visible!(state::Bool)
    card_bg.visible[] = state
    ax_float.blockscene.visible[] = state
    ax_float.scene.visible[] = state
end
set_overlay_visible!(false)

# 4. Mouse Drag & Raycast Logic
is_dragging_mesh = Ref(false)

on(events(fig).mousebutton) do event
    if event.button == Mouse.left
        if event.action == Mouse.press
            plt, idx = Makie.pick(ax3, events(fig).mouseposition[])
            if plt == mesh_plot
                is_dragging_mesh[] = true
                deactivate_interaction!(ax3, :dragrotate)
            end
        elseif event.action == Mouse.release
            if is_dragging_mesh[]
                is_dragging_mesh[] = false
                activate_interaction!(ax3, :dragrotate)
                set_overlay_visible!(false)
            end
        end
    end
end

on(events(fig).mouseposition) do mp
    if is_dragging_mesh[] && ispressed(fig, Mouse.left)
        plt, idx = Makie.pick(ax3, mp)

        if plt == mesh_plot && 1 <= idx <= length(verts)
            hit_pos = verts[idx]
            val = field_values[idx]

            y_samples[] = @. val * sin(x_samples + hit_pos[1] * 3) + cos(x_samples * hit_pos[2] * 3)
            ax_float.title = "Z: $(round(hit_pos[3], digits=2)) | Val: $(round(val, digits=2))"

            w, h = 260, 175
            win_w, win_h = widths(fig.scene)
            x0 = clamp(mp[1] + 20, 10, max(10, win_w - w - 10))
            y0 = clamp(mp[2] + 20, 10, max(10, win_h - h - 10))
            card_rect[] = Rect2f(x0, y0, w, h)

            set_overlay_visible!(true)
        else
            set_overlay_visible!(false)
        end
    end
end

display(fig)
end


export main
end #module