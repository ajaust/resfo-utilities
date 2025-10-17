import numpy as np


def point_in_hexahedron(
    hexahedron: list[tuple[float, float, float]],
    point: tuple[float, float, float],
    quads: list[tuple[float, float, float, float]],
    eps: float = 1.0e-05,
) -> bool:
    p = np.asarray(point, float)
    vertices = np.asarray(hexahedron, float)
    centroid = vertices.mean(axis=0)

    for q in quads:
        a, b, c, d = vertices[np.array(q, int)]  # quad vertices
        # If the quad is slightly non-planar, the (a,b,c) plane is used.
        n = np.cross(b - a, c - a)  # face normal (direction depends on winding)
        if np.linalg.norm(n) < eps:
            # Degenerate face
            continue
        n = n / np.linalg.norm(n)

        # Ensure normal points outward: face centroid should be farther from centroid along +n
        face_centroid = (a + b + c + d) / 4.0
        if np.dot(n, face_centroid - centroid) < 0:
            n = -n  # flip outward

        # Signed distance of p to the plane
        s = np.dot(n, p - a)

        # For a convex polyhedron, inside means p is **not** outside any face
        if s > eps:
            return False

    return True


def point_in_polygon(
    polygon: list[tuple[float, float]], point: tuple[float, float]
) -> bool:
    x, y = point
    n = len(polygon)

    # Early exit if the polygon has fewer than 3 vertices
    if n < 3:
        return False

    # Check for points on a vertex or edge
    # This loop is crucial for handling edge cases and should be done first
    (x1, y1) = polygon[0]
    for i in range(1, n + 1):
        (x2, y2) = polygon[i % n]

        # Check if point lies exactly on a vertex
        if (x, y) == (x1, y1):
            return True

        # Check if point lies on a boundary line segment
        # Using a small tolerance (epsilon) for floating-point comparisons
        epsilon = 1e-9
        if y1 == y2 == y:  # Horizontal edge
            if min(x1, x2) - epsilon <= x <= max(x1, x2) + epsilon:
                return True
        elif x1 == x2 == x:  # Vertical edge
            if min(y1, y2) - epsilon <= y <= max(y1, y2) + epsilon:
                return True
        elif x1 != x2 and y1 != y2:  # Sloped edge
            # Check if point is collinear with the edge
            # (y - y1) * (x2 - x1) == (x - x1) * (y2 - y1)
            # => (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1) == 0
            if abs((y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)) < epsilon:
                # Check if point is within the segment's bounding box
                if (min(x1, x2) - epsilon <= x <= max(x1, x2) + epsilon) and (
                    min(y1, y2) - epsilon <= y <= max(y1, y2) + epsilon
                ):
                    return True

        x1, y1 = x2, y2

    # Standard ray casting algorithm for interior/exterior check
    intersections = 0
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]

        # The key check for a horizontal ray to the right
        if y > min(y1, y2) and y <= max(y1, y2) and x < max(x1, x2):
            # The intersection x-coordinate
            if y1 != y2:
                x_intersection = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            else:  # Horizontal line case handled previously, but this is a failsafe
                x_intersection = float("inf")

            # Count the intersection if it's to the right of the point
            if x <= x_intersection:
                intersections += 1

        x1, y1 = x2, y2

    # Even-odd rule: odd intersections = inside, even = outside
    if intersections % 2 == 1:
        return True
    else:
        return False
