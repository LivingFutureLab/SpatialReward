import numpy as np
from math import sqrt


def normalize(x, y, z):
    unit = sqrt(x * x + y * y + z * z)
    if unit == 0:
        return 0, 0, 0
    return x / unit, y / unit, z / unit


def get_min_max(a, b, c):
    min_val = a
    max_val = a
    if min_val > b:
        min_val = b
    if min_val > c:
        min_val = c
    if max_val < b:
        max_val = b
    if max_val < c:
        max_val = c
    return int(min_val), int(max_val)


def dot_product(a0, a1, a2, b0, b1, b2):
    return a0 * b0 + a1 * b1 + a2 * b2


def cross_product(a0, a1, a2, b0, b1, b2):
    x = a1 * b2 - a2 * b1
    y = a2 * b0 - a0 * b2
    z = a0 * b1 - a1 * b0
    return x, y, z


def generate_faces(triangles, width, height):
    """Draw triangle faces with z-buffer and perspective-correct UV mapping."""
    length = triangles.shape[0]
    zbuffer = {}
    faces = []

    for i in range(length):
        a = triangles[i, 0, 0], triangles[i, 0, 1], triangles[i, 0, 2]
        b = triangles[i, 1, 0], triangles[i, 1, 1], triangles[i, 1, 2]
        c = triangles[i, 2, 0], triangles[i, 2, 1], triangles[i, 2, 2]
        uva = triangles[i, 0, 3], triangles[i, 0, 4]
        uvb = triangles[i, 1, 3], triangles[i, 1, 4]
        uvc = triangles[i, 2, 3], triangles[i, 2, 4]
        minx, maxx = get_min_max(a[0], b[0], c[0])
        miny, maxy = get_min_max(a[1], b[1], c[1])
        pixels = []
        for j in range(minx, maxx + 2):
            for k in range(miny - 1, maxy + 2):
                x = float(j)
                y = float(k)

                m = list(cross_product(
                    c[0] - a[0], b[0] - a[0], a[0] - x,
                    c[1] - a[1], b[1] - a[1], a[1] - y
                ))
                if abs(m[2]) > 0:
                    bcy = m[1] / m[2]
                    bcz = m[0] / m[2]
                    bc = (1 - bcy - bcz, bcy, bcz)
                else:
                    continue

                if bc[0] < -0.00001 or bc[1] < -0.00001 or bc[2] < -0.00001:
                    continue

                z = 1 / (bc[0] / a[2] + bc[1] / b[2] + bc[2] / c[2])

                v = (uva[0] * bc[0] / a[2] + uvb[0] * bc[1] / b[2] + uvc[0] * bc[2] / c[2]) * z * width
                u = height - (uva[1] * bc[0] / a[2] + uvb[1] * bc[1] / b[2] + uvc[1] * bc[2] / c[2]) * z * height

                idx = ((x + y) * (x + y + 1) + y) / 2
                if zbuffer.get(idx) is None or zbuffer[idx] < z:
                    zbuffer[idx] = z
                    pixels.append((i, j, k, int(u) - 1, int(v) - 1))

        faces.append(pixels)
    return faces
