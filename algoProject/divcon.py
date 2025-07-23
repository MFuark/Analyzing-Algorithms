import random
import math
import time

# Custom merge sort (O(n log n))
def merge_sort(points, key=lambda p: p[0]):
    if len(points) <= 1:
        return points
    mid = len(points) // 2
    left = merge_sort(points[:mid], key)
    right = merge_sort(points[mid:], key)
    return merge(left, right, key)

def merge(left, right, key):
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

# Euclidean distance
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Brute-force for small inputs
def brute_force(points):
    min_dist = float('inf')
    best_pair = None
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            d = distance(points[i], points[j])
            if d < min_dist:
                min_dist = d
                best_pair = (points[i], points[j])
    return (min_dist, best_pair)

# Main recursive function
def closest_pair_recursive(Px, Py):
    if len(Px) <= 3:
        return brute_force(Px)

    mid = len(Px) // 2
    Qx = Px[:mid]
    Rx = Px[mid:]
    mid_x = Px[mid][0]

    Qy = [p for p in Py if p[0] <= mid_x]
    Ry = [p for p in Py if p[0] > mid_x]

    (d1, pair1) = closest_pair_recursive(Qx, Qy)
    (d2, pair2) = closest_pair_recursive(Rx, Ry)

    delta = min(d1, d2)
    best_pair = pair1 if d1 < d2 else pair2

    # Check the strip
    strip = [p for p in Py if abs(p[0] - mid_x) < delta]
    for i in range(len(strip)):
        for j in range(i+1, min(i+7, len(strip))):  # Check up to 6 ahead
            d = distance(strip[i], strip[j])
            if d < delta:
                delta = d
                best_pair = (strip[i], strip[j])

    return (delta, best_pair)

# Main wrapper
def closest_pair(points):
    Px = merge_sort(points, key=lambda p: p[0])
    Py = merge_sort(points, key=lambda p: p[1])
    return closest_pair_recursive(Px, Py)

# For testing
if __name__ == "__main__":
    # Generate random points
    def generate_unique_points(n, bound=1000000):
        seen = set()
        points = []
        while len(points) < n:
            x = random.randint(0, bound)
            y = random.randint(0, bound)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append((x, y))
        return points

    n = 100000
    points = generate_unique_points(n)

    start = time.perf_counter()
    dist, pair = closest_pair(points)
    end = time.perf_counter()

    print(f"Closest pair: {pair}")
    print(f"Distance: {dist:.4f}")
    print(f"Time: {(end - start) * 1000:.2f} ms")
