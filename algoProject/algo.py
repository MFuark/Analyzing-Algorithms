import random
import math
import time
from tqdm import tqdm as loading


# for i in mmmm(range(100)):
#     time.sleep(0.05)  # Simulating a task

# --- Sorting (Merge Sort) ---
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

# --- Distance ---
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# --- Brute-Force Algorithm (ALG1) ---
def brute_force_closest_pair(points):
    min_dist = float('inf')
    best_pair = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = distance(points[i], points[j])
            if d < min_dist:
                min_dist = d
                best_pair = (points[i], points[j])
    return (min_dist, best_pair)

# --- Divide-and-Conquer Algorithm (ALG2) ---
def closest_pair_recursive(Px, Py):
    if len(Px) <= 3:
        return brute_force_closest_pair(Px)

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

    # Build strip: points close to the dividing line
    strip = [p for p in Py if abs(p[0] - mid_x) < delta]
    for i in range(len(strip)):
        for j in range(i + 1, min(i + 7, len(strip))):
            d = distance(strip[i], strip[j])
            if d < delta:
                delta = d
                best_pair = (strip[i], strip[j])

    return (delta, best_pair)

def closest_pair(points):
    Px = merge_sort(points, key=lambda p: p[0])
    Py = merge_sort(points, key=lambda p: p[1])
    return closest_pair_recursive(Px, Py)

# --- Random Input Generator ---
def generate_points(n, bound=1000000):
    seen = set()
    points = []
    while len(points) < n:
        x = random.randint(0, bound)
        y = random.randint(0, bound)
        if (x, y) not in seen:
            seen.add((x, y))
            points.append((x, y))
    return points

# --- Comparison/Test Function ---
def compare_algorithms(n, trials=10, bound=1000000):
    bf_times = []
    dc_times = []

    for _ in range(trials):
        points = generate_points(n, bound)

        # Brute-Force
        start_bf = time.perf_counter()
        bf_dist, _ = brute_force_closest_pair(points)
        end_bf = time.perf_counter()
        bf_times.append((end_bf - start_bf) * 1000)

        # Divide-and-Conquer
        start_dc = time.perf_counter()
        dc_dist, _ = closest_pair(points)
        end_dc = time.perf_counter()
        dc_times.append((end_dc - start_dc) * 1000)

        # Optional sanity check:
        assert math.isclose(bf_dist, dc_dist, rel_tol=1e-9), f"Mismatch! BF: {bf_dist}, DC: {dc_dist}"

    avg_bf = sum(bf_times) / trials
    avg_dc = sum(dc_times) / trials

    return avg_bf, avg_dc

# --- Run Experiments ---
if __name__ == "__main__":
    ns = [1000, 2000, 3000, 4000, 5000]  # Increase n only for divide-and-conquer later
    results = []

    print("Running Closest Pair Comparison...\n")

    for n in loading(ns):
        # print(f"Testing n = {n}")
        avg_bf, avg_dc = compare_algorithms(n)
        results.append((n, avg_bf, avg_dc))
        # print(f"  Brute-Force Avg: {avg_bf:.2f} ms")
        # print(f"  Divide-&-Conquer Avg: {avg_dc:.2f} ms")

    # Final Summary
    print("\nSummary:")
    print(f"{'n':>6}  {'BF Time (ms)':>15}  {'DC Time (ms)':>15}")
    for n, bf, dc in results:
        print(f"{n:>6}  {bf:15.2f}  {dc:15.2f}")
