import os
import csv
import math
import random
import pandas as pd

random.seed(42)

# =====================================================
# CARGAR INSTANCIA
# =====================================================

def load_instance(filepath):

    points = []

    with open(filepath, newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            points.append(
                (
                    float(row["x"]),
                    float(row["y"])
                )
            )

    return points


# =====================================================
# DISTANCIA
# =====================================================

def dist(a, b):

    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


# =====================================================
# COSTO DEL TOUR
# =====================================================

def tour_cost(route, points):

    cost = 0

    n = len(route)

    for i in range(n):

        a = points[route[i]]

        b = points[route[(i + 1) % n]]

        cost += dist(a, b)

    return cost


# =====================================================
# VECINO MAS CERCANO
# =====================================================

def nearest_neighbor(points):

    n = len(points)

    current = 0

    unvisited = set(range(n))

    unvisited.remove(current)

    route = [current]

    while unvisited:

        nxt = min(
            unvisited,
            key=lambda j:
            dist(points[current], points[j])
        )

        route.append(nxt)

        unvisited.remove(nxt)

        current = nxt

    return route


# =====================================================
# 2 OPT HILL CLIMBING
# =====================================================

def two_opt(route, points):

    route = route[:]

    improved = True

    n = len(route)

    while improved:

        improved = False

        best_cost = tour_cost(
            route,
            points
        )

        for i in range(1, n - 1):

            for j in range(i + 1, n):

                candidate = (
                    route[:i]
                    + route[i:j+1][::-1]
                    + route[j+1:]
                )

                candidate_cost = tour_cost(
                    candidate,
                    points
                )

                if candidate_cost < best_cost:

                    route = candidate

                    best_cost = candidate_cost

                    improved = True

    return route


def tsp_2opt(points):

    start = nearest_neighbor(points)

    best = two_opt(
        start,
        points
    )

    return tour_cost(
        best,
        points
    )


# =====================================================
# SIMULATED ANNEALING
# =====================================================

def tsp_sa(points):

    route = nearest_neighbor(points)

    current = route[:]

    current_cost = tour_cost(
        current,
        points
    )

    best = current[:]

    best_cost = current_cost

    T = 100

    cooling = 0.95

    n = len(route)

    for _ in range(1000):

        i = random.randint(
            1,
            n - 2
        )

        j = random.randint(
            i + 1,
            n - 1
        )

        candidate = (
            current[:i]
            + current[i:j+1][::-1]
            + current[j+1:]
        )

        candidate_cost = tour_cost(
            candidate,
            points
        )

        delta = current_cost - candidate_cost

        if delta > 0:

            current = candidate
            current_cost = candidate_cost

        else:

            p = math.exp(
                delta / max(T, 0.001)
            )

            if random.random() < p:

                current = candidate
                current_cost = candidate_cost

        if current_cost < best_cost:

            best = current[:]
            best_cost = current_cost

        T *= cooling

    return best_cost


# =====================================================
# MST + 2OPT
# =====================================================

def mst_tour(points):

    n = len(points)

    in_tree = [False] * n

    min_edge = [float("inf")] * n

    parent = [-1] * n

    min_edge[0] = 0

    graph = [[] for _ in range(n)]

    for _ in range(n):

        u = min(

            [
                i
                for i in range(n)
                if not in_tree[i]
            ],

            key=lambda x:
            min_edge[x]
        )

        in_tree[u] = True

        if parent[u] != -1:

            graph[u].append(parent[u])

            graph[parent[u]].append(u)

        for v in range(n):

            if not in_tree[v]:

                d = dist(
                    points[u],
                    points[v]
                )

                if d < min_edge[v]:

                    min_edge[v] = d

                    parent[v] = u

    route = []

    visited = [False] * n

    def dfs(u):

        visited[u] = True

        route.append(u)

        for v in graph[u]:

            if not visited[v]:

                dfs(v)

    dfs(0)

    return route


def tsp_mst_2opt(points):

    tour = mst_tour(points)

    best = two_opt(
        tour,
        points
    )

    return tour_cost(
        best,
        points
    )


# =====================================================
# ETIQUETAR DATASET
# =====================================================

def process_dataset(file_name):

    df = pd.read_csv(
        os.path.join(
            "data",
            file_name
        )
    )

    cost_2opt = []
    cost_sa = []
    cost_mst = []

    winners = []

    total = len(df)

    for idx, row in df.iterrows():

        instance_id = row["instance_id"]

        filepath = os.path.join(
            "data",
            "instances",
            f"{instance_id}.csv"
        )

        points = load_instance(filepath)

        c2 = tsp_2opt(points)

        csa = tsp_sa(points)

        cmst = tsp_mst_2opt(points)

        best_algo = min(
            {
                "2opt_hill_climbing": c2,
                "sa_2opt": csa,
                "mst_2opt": cmst
            },
            key=lambda k: {
                "2opt_hill_climbing": c2,
                "sa_2opt": csa,
                "mst_2opt": cmst
            }[k]
        )

        cost_2opt.append(c2)
        cost_sa.append(csa)
        cost_mst.append(cmst)
        winners.append(best_algo)

        print(
            f"[{idx+1}/{total}] {instance_id}"
        )

    df["cost_2opt_hill_climbing"] = cost_2opt
    df["cost_sa_2opt"] = cost_sa
    df["cost_mst_2opt"] = cost_mst
    df["best_algorithm"] = winners

    output_file = file_name.replace(
        ".csv",
        "_labeled.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print()
    print("Guardado:", output_file)


# =====================================================
# MAIN
# =====================================================

process_dataset("train.csv")
process_dataset("test.csv")



