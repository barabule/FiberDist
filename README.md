Mold filling approximation tool, useful to quickly estimate fiber orientations, weld lines etc. from a CAD file.
Based on a Eikonal equation solver on the voxelization of the geometry, followed by a projection of the results on the surface geometry (2.5d analytic approach).
This keeps the solution time interactive.

Gates can be dragged over the mesh geometry(drag after gate highlights) and resized (drag up/down with Shift pressed).
Add / delete gates with 'a' and 'd' (last gate cannot be deleted).
After each drag the system is solved and the colormap of the fill time updates. This can also be animated with the bottom slider and play / pause button.

Geometry can be imported as CAD files (and surface meshed by Gmsh) or directly in a polygonal format (stl, ply, obj etc).
In case of CAD files, these can be remeshed. 

Voxelization resolution along the longest axis can be changed and inspeted in the voxelization panel. Aim to have at least 2 filled voxels on every rib.
In tests the voxel resolution could be increased over 1000 with some slowdowns (and increased memory usage).




Coded with Gemini
