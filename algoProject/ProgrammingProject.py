import math
import time
import random
from time import sleep
from tqdm import tqdm
from tqdm import trange

import tkinter as tk
from tkinter import ttk

class ClosestPoints:
    
    def __init__(self, master):
        self.master = master
        self.m = 50000
        self.points = []
        
        
        
def GenerateRandomPoint(pointsList, m):
    
    for x in trange(m):
        newPoint = (random.randint(0, 100000000), random.randint(0, 100000000))
        while newPoint not in points:
            newPoint = (random.randint(0, 100000000),random.randint(0, 100000000))
            pointsList += newPoint,
            
    return pointsList
            

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


"""

Algorithm BruteForceClosestPoints(P)
// P is a list of n points, n ≥ 2, P1 = (x 1 , y 1 ),..., Pn = (x n , y n )
// returns the index 1 and index 2 of the closest pair of points
d min = ∞
for i = 1 to n-1
for j = i + 1 to n
if d < d min
d min = d; index 1 = i; index 2 = j
return index 1 , index 2
𝑑𝑑 = (𝑥𝑥𝑖𝑖 − 𝑥𝑥𝑗𝑗 )2+ 𝑦𝑦𝑖𝑖 − 𝑦𝑦𝑗𝑗 2

"""


    
    
# theArray = []

# for i in range(3):
#     randX = random.randint(0, 100)
#     randY = random.randint(0, 100) 

#     theArray += (randX, randY),
    
#     arrayLength = len(theArray)
    
#     for j in range(arrayLength):
#         print("Item", j)
#         print(theArray[j][0])





def brute_force_closest_pair(points):
    min_dist = float('inf')
    pair = None
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(points[i], points[j])
            if dist < min_dist:
                min_dist = dist
                pair = (points[i], points[j])
    return pair, min_dist

def closest_pair_recursive(Px, Py):
    n = len(Px)
    if n <= 3:
        return brute_force_closest_pair(Px)

    mid = n // 2
    Qx = Px[:mid]
    Rx = Px[mid:]
    midpoint_x = Px[mid][0]

    Qy = [p for p in Py if p[0] <= midpoint_x]
    Ry = [p for p in Py if p[0] > midpoint_x]

    (pair_left, dist_left) = closest_pair_recursive(Qx, Qy)
    (pair_right, dist_right) = closest_pair_recursive(Rx, Ry)

    delta = min(dist_left, dist_right)
    best_pair = pair_left if dist_left <= dist_right else pair_right

    # Build strip[]: points within delta of the vertical line
    strip = [p for p in Py if abs(p[0] - midpoint_x) < delta]

    # Check the strip for closer pairs (only 7 ahead in Y-sorted list)
    for i in range(len(strip)):
        for j in range(i+1, min(i+8, len(strip))):  # at most 7 ahead
            p, q = strip[i], strip[j]
            dist = euclidean_distance(p, q)
            if dist < delta:
                delta = dist
                best_pair = (p, q)

    return best_pair, delta

def closest_pair(points):
    Px = sorted(points, key=lambda p: p[0])  # sort by x
    Py = sorted(points, key=lambda p: p[1])  # sort by y
    return closest_pair_recursive(Px, Py)









# Example usage
if __name__ == "__main__":
    
    try:
        root = tk.Tk()
        gui = InterruptGUI(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught! Program exiting gracefully.")

    # Create a progressbar widget
    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress.pack(pady=20)

    # Button to start progress
    start_button = tk.Button(root, text="Start Progress", command=start_progress)
    start_button.pack(pady=10)

    root.mainloop()
    
    # points = [(2, 3), (12, 30), (40, 50), (5, 1), (3, 4), (3, 4)]
    points = [] 
    m = 5000
    points = GenerateRandomPoint(points, m)
        
    
    
    pair, distance = closest_pair(points)
    print(f"Closest pair: {pair}, Distance: {distance:.4f}")








def BruteForceClosestPoints(P):
    pass

def DivideAndConquer(P):
    pass


