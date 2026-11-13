# Example: 2-triangle unit square
V = [0.0 1.0 1.0 0.0;
     0.0 0.0 1.0 1.0;
     0.0 0.0 0.0 0.0]
F = [1 1;
     2 3;
     3 4]

solver = HeatMethodSolver(V, F)
distances = compute_geodesic(solver, [1]) # Distance from vertex 1 -> ≈ [0.0, 1.0, 1.414, 1.0]
