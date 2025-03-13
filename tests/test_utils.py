import src.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from src.utils import plot_force_disp
import dolfinx
import ufl
import mpi4py

def test_plotter_func():
    mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 10, 10)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1,))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
    utils.plotter_func(u, 2, mesh, "tests/test_plotter_func.png")

def test_distance_points_to_segment():
    """Test the distance_points_to_segment function."""
    # Test case 1: Point on the line segment
    points = [(1, 1)]  # Point on the line segment from (0,0) to (2,2)
    x1, y1, x2, y2 = 0, 0, 2, 2
    distances = utils.distance_points_to_segment(points, x1, y1, x2, y2)
    assert len(distances) == 1
    assert np.isclose(distances[0], 0.0)
    
    # Test case 2: Point directly above the middle of the line segment
    points = [(1, 2)]  # Point above the middle of horizontal line from (0,1) to (2,1)
    x1, y1, x2, y2 = 0, 1, 2, 1
    distances = utils.distance_points_to_segment(points, x1, y1, x2, y2)
    assert np.isclose(distances[0], 1.0)
    
    # Test case 3: Point closest to an endpoint
    points = [(-1, 0)]  # Point closest to the first endpoint of line from (0,0) to (5,0)
    x1, y1, x2, y2 = 0, 0, 5, 0
    distances = utils.distance_points_to_segment(points, x1, y1, x2, y2)
    assert np.isclose(distances[0], 1.0)
    
    # Test case 4: Point beyond the second endpoint
    points = [(7, 1)]  # Point beyond the second endpoint of line from (0,0) to (5,0)
    x1, y1, x2, y2 = 0, 0, 5, 0
    distances = utils.distance_points_to_segment(points, x1, y1, x2, y2)
    assert np.isclose(distances[0], np.sqrt(5))  # sqrt((7-5)² + (1-0)²) = sqrt(5)
    
    # Test case 5: Multiple points
    points = [(1, 1), (0, 2), (3, -1)]  # Multiple points with different positions
    x1, y1, x2, y2 = 0, 0, 2, 0
    distances = utils.distance_points_to_segment(points, x1, y1, x2, y2)
    assert len(distances) == 3
    assert np.isclose(distances[0], 1.0)  # Point (1,1) is 1 unit above the line
    assert np.isclose(distances[1], 2.0)  # Point (0,2) is 2 units above the first endpoint
    assert np.isclose(distances[2], np.sqrt(2))  # Point (3,-1) is sqrt(2) units from the line

def test_plot_force_disp():
    # Create sample data
    B = np.array([
        [-100, 0.1],
        [-200, 0.2],
        [-300, 0.3],
        [-400, 0.4]
    ])
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Call the function
        plot_force_disp(B, "test", temp_dir)
        
        # Check if files were created
        assert os.path.exists(f"{temp_dir}/force_disp_test.png")
        assert os.path.exists(f"{temp_dir}/force_test.txt")
        
        # Check if the saved data matches the input
        saved_data = np.loadtxt(f"{temp_dir}/force_test.txt")
        np.testing.assert_array_equal(saved_data, B)

