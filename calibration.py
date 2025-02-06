import cv2
import numpy as np
from scipy.optimize import minimize


def load_image_and_detect_lines(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10
    )
    return lines, width, height


def collect_points(lines, num_points_per_line=10):
    points_per_line = []
    if lines is None:
        return points_per_line
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_points = []
        for t in np.linspace(0, 1, num_points_per_line):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            line_points.append([x, y])
        points_per_line.append(np.array(line_points))
    return points_per_line


def apply_undistortion(points, params, width, height):
    cx_offset, cy_offset, k1, k2, k3, k4 = params
    cx = width / 2 + cx_offset
    cy = height / 2 + cy_offset

    x = (points[:, 0] - cx) / (width / 2)
    y = (points[:, 1] - cy) / (height / 2)
    r = np.sqrt(x**2 + y**2)

    # Radial distortion correction: r_corrected = r + k1*r^3 + k2*r^5 + k3*r^7 + k4*r^9
    r_corrected = r + k1 * r**3 + k2 * r**5 + k3 * r**7 + k4 * r**9

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.divide(r_corrected, r, where=r != 0)
    x_corrected = x * scale
    y_corrected = y * scale

    x_undistorted = x_corrected * (width / 2) + cx
    y_undistorted = y_corrected * (height / 2) + cy

    return np.column_stack((x_undistorted, y_undistorted))


def objective(params, points_per_line, width, height):
    total_error = 0.0
    for line_points in points_per_line:
        if len(line_points) < 2:
            continue
        undistorted = apply_undistortion(line_points, params, width, height)
        vx, vy, x0, y0 = cv2.fitLine(
            undistorted.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        )
        dx = undistorted[:, 0] - x0
        dy = undistorted[:, 1] - y0
        distances = np.abs(vy * dx - vx * dy) / np.sqrt(vy**2 + vx**2)
        total_error += np.sum(distances**2)
    return total_error


# Configuration
image_path = "/kaggle/input/fisheye8k/Fisheye8K/train/images/camera10_A_0.png"  # Replace with your image path
try:
    lines, width, height = load_image_and_detect_lines(image_path)
except Exception as e:
    print(e)
    exit()

points_per_line = collect_points(lines)

# Initial parameters (scaled for numerical stability)
initial_params = [3.942 / 1000, -3.093 / 1000, 0.339749, -0.031988, 0.048275, -0.007201]

# Optimization with bounds
bounds = [
    (-100, 100),  # cx_offset
    (-100, 100),  # cy_offset
    (-1, 1),  # k1
    (-1, 1),  # k2
    (-1, 1),  # k3
    (-1, 1),  # k4
]

result = minimize(
    objective,
    initial_params,
    args=(points_per_line, width, height),
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": 100},
)

# Get optimized intrinsic parameters
cx_offset, cy_offset, k1, k2, k3, k4 = result.x

# Format output (extrinsic based on typical setup)
output = {
    "extrinsic": {
        "quaternion": [
            0.5946970238045494,
            -0.5837953694518585,
            0.39063952590941586,
            -0.3910488170060691,
        ],
        "translation": [3.7484, 0.0, 0.6577999999999999],
    },
    "intrinsic": {
        "aspect_ratio": 1.0,
        "cx_offset": cx_offset,
        "cy_offset": cy_offset,
        "height": float(height),
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "k4": k4,
        "model": "radial_poly",
        "poly_order": 4,
        "width": float(width),
    },
    "name": "MVR",
}

print(output)
