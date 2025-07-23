import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def merge_sort(points, key=lambda p: p):
    if len(points) <= 1:
        return points
    mid = len(points) // 2
    left = merge_sort(points[:mid], key)
    right = merge_sort(points[mid:], key)
    return merge(left, right, key)


def merge(left, right, key):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def analyze_algorithms():
    n_values = [i * 1000 for i in range(1, 11)]
    bf_results = []
    dc_results = []

    for n in tqdm(n_values):
        points = GenerateRandomPoints(n)

        start = time.time()
        bf_p1, bf_p2, bf_dist = brute_force_closest_points(points)
        bf_time = (time.time() - start) * 1000
        theoretical_bf = n ** 2
        ratio_bf = bf_time / theoretical_bf
        bf_results.append((n, theoretical_bf, bf_time, ratio_bf))

        start = time.time()
        dc_p1, dc_p2, dc_dist = divide_and_conquer_closest_pair(points)
        dc_time = (time.time() - start) * 1000
        theoretical_dc = n * math.log2(n)
        ratio_dc = dc_time / theoretical_dc
        dc_results.append((n, theoretical_dc, dc_time, ratio_dc))

    return bf_results, dc_results


def generate_tables(results, label, filename):
    df = pd.DataFrame(results, columns=["n", "TheoreticalRT", "EmpiricalRT (ms)", "Ratio"])
    c = df["Ratio"].max()
    df["PredictedRT"] = df["TheoreticalRT"] * c
    df.to_csv(filename, index=False)
    print(f"Table {label} saved to {filename}")
    return df, c


def plot_graphs(df_bf, df_dc):
    plt.figure(figsize=(10, 6))
    plt.plot(df_bf["n"], df_bf["EmpiricalRT (ms)"], 'o-', label="Brute Force Empirical RT")
    plt.plot(df_dc["n"], df_dc["EmpiricalRT (ms)"], 's-', label="Divide & Conquer Empirical RT")
    plt.xlabel("n")
    plt.ylabel("Empirical Runtime (ms)")
    plt.title("Empirical Runtime Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("Empirical_Comparison.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df_bf["n"], df_bf["EmpiricalRT (ms)"], 'o-', label="Brute Force Empirical RT")
    plt.plot(df_bf["n"], df_bf["PredictedRT"], 'x--', label="Brute Force Predicted RT")
    plt.xlabel("n")
    plt.ylabel("Runtime (ms)")
    plt.title("Brute Force: Empirical vs Predicted RT")
    plt.legend()
    plt.grid(True)
    plt.savefig("Brute_Force_Predicted.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df_dc["n"], df_dc["EmpiricalRT (ms)"], 's-', label="Divide & Conquer Empirical RT")
    plt.plot(df_dc["n"], df_dc["PredictedRT"], 'x--', label="Divide & Conquer Predicted RT")
    plt.xlabel("n")
    plt.ylabel("Runtime (ms)")
    plt.title("Divide and Conquer: Empirical vs Predicted RT")
    plt.legend()
    plt.grid(True)
    plt.savefig("Divide_Conquer_Predicted.png")
    plt.show()


def GenerateRandomPoints(n):
    points = set()
    while len(points) < n:
        point = (random.randint(0, 100000000), random.randint(0, 100000000))
        points.add(point)
    return list(points)


def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def brute_force_closest_points(P):
    n = len(P)
    d_min = float('inf')
    index_1, index_2 = 0, 1
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = euclidean_distance(P[i], P[j])
            if d < d_min:
                d_min = d
                index_1, index_2 = i, j
    return P[index_1], P[index_2], d_min


def closest_pair_recursive(Px, Py):
    n = len(Px)
    if n <= 3:
        return brute_force_closest_points(Px)

    mid = n // 2
    Qx = Px[:mid]
    Rx = Px[mid:]
    midpoint_x = Px[mid][0]

    Qy = [p for p in Py if p[0] <= midpoint_x]
    Ry = [p for p in Py if p[0] > midpoint_x]

    q1, q2, d_q = closest_pair_recursive(Qx, Qy)
    r1, r2, d_r = closest_pair_recursive(Rx, Ry)

    delta = min(d_q, d_r)
    best_pair = (q1, q2) if d_q < d_r else (r1, r2)

    strip = [p for p in Py if abs(p[0] - midpoint_x) < delta]

    for i in range(len(strip)):
        for j in range(i + 1, min(i + 16, len(strip))):
            p, q = strip[i], strip[j]
            dist = euclidean_distance(p, q)
            if dist < delta:
                delta = dist
                best_pair = (p, q)

    return best_pair[0], best_pair[1], delta


def divide_and_conquer_closest_pair(P):
    Px = merge_sort(P, key=lambda p: p[0])
    Py = merge_sort(P, key=lambda p: p[1])
    return closest_pair_recursive(Px, Py)







if __name__ == "__main__":
    print("Running runtime analysis for 10 values of n...")
    bf_results, dc_results = analyze_algorithms()

    print(bf_results)
    df_bf, c1 = generate_tables(bf_results, "ALG1 (Brute Force)", "table_alg1_brute_force.csv")
    df_dc, c2 = generate_tables(dc_results, "ALG2 (Divide and Conquer)", "table_alg2_divide_conquer.csv")

    print(f"\nEstimated c1 (Brute Force): {c1:e}")
    print(f"Estimated c2 (Divide and Conquer): {c2:e}")

    plot_graphs(df_bf, df_dc)
