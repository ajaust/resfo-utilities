import numpy as np
from shapely import Point, Polygon


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
    _polygon = Polygon(polygon)
    _point = Point(point)
    return _polygon.intersects(_point)
