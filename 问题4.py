# -*- coding: gbk -*-
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ����ģ��2

# ================== �������ݶ��� ==================
# �����Ż����ͣ���㲼�֣��������ꡢ�ܳ������� ѡ�����Ͼŵ�
parking_spots = {
    '������': {'x': 13039873.05, 'y': 4058597.558, 'total': 4},
    '����': {'x': 13040439.39, 'y': 4058474.212, 'total': 52},
    '����': {'x': 13039882.45, 'y': 4058168.561, 'total': 3},
    '��Է1��': {'x': 13039630.26, 'y': 4058273.857, 'total': 5},
    '�����ѧԺ': {'x': 13040185.71, 'y': 4058092.577, 'total': 0},
    '��������': {'x': 13040528.62, 'y': 4057938.989, 'total': 59},
    '����': {'x': 13040362.37, 'y': 4057491.841, 'total': 76},
    '��ѧ4¥': {'x': 13039789.83, 'y': 4057363.079, 'total': 72},
    '��ѧ2¥': {'x': 13039829.74, 'y': 4057732.176, 'total': 70},
    '��ʳ��': {'x': 13039440.37, 'y': 4057325.086, 'total': 2},
    '����': {'x': 13039317.06, 'y': 4056875.487, 'total': 19},
    '÷Է1��': {'x': 13039141.26, 'y': 4057377.553, 'total': 30},
    'УҽԺ': {'x': 13039005.96, 'y': 4057544.19, 'total': 3},
    '��ʳ��': {'x': 13039181.9, 'y': 4057737.509, 'total': 35},
    'һʳ��': {'x': 13039538.2, 'y': 4057841.582, 'total': 10},
    'ͼ�������': {'x': 13039762.12, 'y': 4057976.279, 'total': 0},
    '���ӿ�ѧѧԺ': {'x': 13040198.66, 'y': 4057838.718, 'total': 0},
    '�㳡': {'x': 13040045.96, 'y': 4057632.647, 'total': 0},
    '�ʺ���': {'x': 13039334.28, 'y': 4057933.14, 'total': 0},
    'ʵ��¥': {'x': 13039294.34, 'y': 4057011.254, 'total': 0},
    '���޴�':{'x':13040851.7209,'y':4058161.078400001,'total':0}
}

distance=����ģ��2.distance_map

# ��������
failure_rate = 0.06      # ������ 6%
vehicle_capacity = 20    # ���䳵����
speed = 25               # ��ʻ�ٶ� (km/h)
loading_time_per_bike = 1 / 60  # װ��ʱ�� (Сʱ/��)

# �㷨��������
num_ants = 20        # ��������
alpha = 1            # ��Ϣ����Ҫ�̶�����
beta = 2             # ����ʽ����
rho = 0.1           # ��Ϣ�ػӷ�ϵ��
Q = 100             # ��Ϣ��ǿ�ȳ���
num_iterations = 50 # ��������
# ================== ����Ԥ���� ==================
def prepare_data():
    # ���������ϳ�����
    demands = {}
    coordinates = []
    for spot, info in parking_spots.items():
        if spot == '���޴�':
            continue
        demand = int(info['total'] * failure_rate)
        demands[spot] = demand
        coordinates.append([info['x'], info['y']])
    
    # ���޴�����
    depot = [parking_spots['���޴�']['x'], parking_spots['���޴�']['y']]
    
    return demands, np.array(coordinates), depot

# ================== ��Ⱥ�㷨ʵ�� ==================
class AntColonyOptimization:
    def __init__(self, distance_matrix, num_ants, alpha, beta, rho, Q, num_iterations):
        self.distance_matrix = distance_matrix  # �������
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.num_iterations = num_iterations
        self.num_locations = len(distance_matrix)
        self.pheromone_matrix = np.ones((self.num_locations, self.num_locations))  # ��ʼ����Ϣ�ؾ���
        self.best_path = None
        self.best_path_length = float('inf')
    
    def run(self):
        for _ in range(self.num_iterations):
            # ��������·��
            ant_paths = []
            ant_path_lengths = []
            for _ in range(self.num_ants):
                path, path_length = self.generate_ant_path()
                ant_paths.append(path)
                ant_path_lengths.append(path_length)
                
                # ��������·��
                if path_length < self.best_path_length:
                    self.best_path_length = path_length
                    self.best_path = path
            
            # ������Ϣ��
            self.update_pheromones(ant_paths, ant_path_lengths)
        
        return self.best_path, self.best_path_length
    
    def generate_ant_path(self):
        # �Ӽ��޴�����
        start_index = 0  # ������޴��ǵ�һ��λ��
        current_index = start_index
        path = [current_index]
        path_length = 0.0
        
        # ������������λ��
        for _ in range(self.num_locations - 1):
            # ѡ����һ��λ��
            next_index = self.select_next_location(current_index)
            path.append(next_index)
            path_length += self.distance_matrix[current_index][next_index]
            current_index = next_index
        
        # ���ؼ��޴�
        path.append(start_index)
        path_length += self.distance_matrix[current_index][start_index]
        
        return path, path_length
    
    def select_next_location(self, current_index):
        # ����ѡ�����
        probabilities = np.zeros(self.num_locations)
        for i in range(self.num_locations):
            if i not in self.best_path and i != current_index:
                probabilities[i] = (self.pheromone_matrix[current_index][i] ** self.alpha) * \
                                   ((1.0 / self.distance_matrix[current_index][i]) ** self.beta)
        
        # ��һ������
        probabilities /= probabilities.sum()
        
        # ���ѡ����һ��λ��
        return np.random.choice(range(self.num_locations), p=probabilities)
    
    def update_pheromones(self, ant_paths, ant_path_lengths):
        # ��Ϣ�ػӷ�
        self.pheromone_matrix *= (1.0 - self.rho)
        
        # �������Ϣ��
        for path, path_length in zip(ant_paths, ant_path_lengths):
            for i in range(len(path) - 1):
                from_index = path[i]
                to_index = path[i + 1]
                self.pheromone_matrix[from_index][to_index] += self.Q / path_length

# ================== ������ˣ�����Լ����==================

def cluster_points(demands, coordinates, depot):
    # ת��Ϊ�������� (���� + ������)
    features = []
    spots = list(demands.keys())
    for i, spot in enumerate(spots):
        features.append([coordinates[i][0], coordinates[i][1], demands[spot]])
    
    # K-means���ࣨ��������Լ������������
    total_demand = sum(demands.values())
    n_clusters = int(np.ceil(total_demand / vehicle_capacity))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    clusters = kmeans.labels_
    
    # ����ز���֤����
    clusters_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in clusters_dict:
            clusters_dict[cluster_id] = {'spots': [], 'demand': 0}
        
        clusters_dict[cluster_id]['spots'].append(spots[i])
        clusters_dict[cluster_id]['demand'] += demands[spots[i]]
    
    return clusters_dict

# ================== ʹ����Ⱥ�㷨==================
def optimize_route_with_aco(coordinates, depot, spots_in_cluster, distance,**aco_params):
    """ �Ż����·���Ż����� """
    # �����������ʹ��Ԥ�����ɵ�distance_map��
    num_spots = len(spots_in_cluster) + 1  # �������޴�
    
    # ���������б����޴�ʼ��Ϊ��һ���㣩
    all_spots = ['���޴�'] + spots_in_cluster
    
    # ��ʼ���������
    distance_matrix = np.zeros((num_spots, num_spots))
    for i in range(num_spots):
        for j in range(num_spots):
            from_spot = all_spots[i]
            to_spot = all_spots[j]
            # ����ʹ��distance_map�еľ���
            if from_spot in distance and to_spot in distance[from_spot]:
                distance_matrix[i][j] = distance[from_spot][to_spot]
            else:
                # �󱸼��㣨ŷ�Ͼ��룩
                coord_i = parking_spots[from_spot] if from_spot != '���޴�' else {'x': depot[0], 'y': depot[1]}
                coord_j = parking_spots[to_spot] if to_spot != '���޴�' else {'x': depot[0], 'y': depot[1]}
                distance = np.sqrt((coord_j['x']-coord_i['x'])**2 + (coord_j['y']-coord_i['y'])**2)
                distance_matrix[i][j] = distance
    
    # ������Ⱥ�㷨
    aco = AntColonyOptimization(
        distance_matrix=distance_matrix,
        num_ants=aco_params.get('num_ants', 20),
        alpha=aco_params.get('alpha', 1),
        beta=aco_params.get('beta', 2),
        rho=aco_params.get('rho', 0.1),
        Q=aco_params.get('Q', 100),
        num_iterations=aco_params.get('num_iterations', 50)
    )
    
    best_path_indices, best_path_length = aco.run()
    
    # ת������Ϊ����
    best_path = [all_spots[idx] for idx in best_path_indices]
    
    return best_path

# ================== ·���Ż���ʹ��OR-Tools��������Ⱥ�㷨���棩==================

def optimize_route(coordinates, depot, spots_in_cluster):
    # �����������
    def create_distance_matrix(coords):
        num_locs = len(coords)
        dist_matrix = np.zeros((num_locs, num_locs))
        for i in range(num_locs):
            for j in range(num_locs):
                dist_matrix[i][j] = np.linalg.norm(coords[i]-coords[j])
        return dist_matrix
    
    # ��������˳����Ӽ��޴���
    cluster_coords = [depot]  # ���޴���Ϊ���
    spot_names = ['���޴�']
    for spot in spots_in_cluster:
        cluster_coords.append([parking_spots[spot]['x'], parking_spots[spot]['y']]) 
        spot_names.append(spot)
    
    dist_matrix = create_distance_matrix(np.array(cluster_coords))
    
    # ����·��ģ��
    manager = pywrapcp.RoutingIndexManager(len(cluster_coords), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # �������
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    
    # �������
    solution = routing.SolveWithParameters(search_parameters)
    
    # ��ȡ·��
    route = []
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(spot_names[node])
            index = solution.Value(routing.NextVar(index))
        route.append(spot_names[manager.IndexToNode(index)])
    
    return route

# ================== ��ʱ����� ==================
def calculate_time(route, speed, loading_time, parking_spots, distance_map):
    """ �Ż����ʱ����㺯�� """
    total_distance = 0
    total_load_time = 0
    
    for i in range(len(route)-1):
        from_spot = route[i]
        to_spot = route[i+1]
        
        # ʹ��distance_map��ȡ����
        try:
            distance = distance_map[from_spot][to_spot]
        except KeyError:
            # �󱸼��㣨ŷ�Ͼ��룩
            coord_from = parking_spots[from_spot] if from_spot != '���޴�' else {'x': depot[0], 'y': depot[1]}
            coord_to = parking_spots[to_spot] if to_spot != '���޴�' else {'x': depot[0], 'y': depot[1]}
            distance = np.sqrt((coord_to['x']-coord_from['x'])**2 + (coord_to['y']-coord_from['y'])**2)
        
        total_distance += distance
        
        # װ��ʱ�����
        if from_spot != '���޴�':
            total_load_time += parking_spots[from_spot]['total'] * failure_rate * loading_time
    
    travel_time = total_distance / speed  # ��
    return travel_time + total_load_time

def calculate_time_0(route, speed, loading_time):
    total_distance = 0
    total_load_time = 0
    depot = parking_spots['���޴�']
    
    for i in range(len(route)-1):
        from_spot = route[i]
        to_spot = route[i+1]
        
        # �������
        x1, y1 = parking_spots[from_spot]['x'], parking_spots[from_spot]['y']
        x2, y2 = parking_spots[to_spot]['x'], parking_spots[to_spot]['y']
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)/1000
        total_distance += distance
        
        # ����װ��ʱ�䣨���ڷǼ��޴�װ����
        if from_spot != '���޴�':
            total_load_time += parking_spots[from_spot]['total'] * failure_rate * loading_time
    
    travel_time = total_distance / speed  # Сʱ
    total_time = travel_time + total_load_time
    return total_time

# ================== ������ ==================
def main():
    # ����׼��
    demands, coordinates, depot = prepare_data()
    
    distance=����ģ��2.distance_map

    # �������
    clusters = cluster_points(demands, coordinates, depot)
    
    # �ִ��Ż�·��
    total_time = 0
    for cluster_id, cluster_info in clusters.items():
        spots_in_cluster = cluster_info['spots']
        print(f"\n=== �� {cluster_id} ===")
        print(f"����ͣ����: {spots_in_cluster}")
        print(f"�ܹ��ϳ���: {cluster_info['demand']}")
        
        # ʹ����Ⱥ�㷨�Ż�·������Ҫ�޸ĵ㣩
        optimal_route = optimize_route_with_aco(
            coordinates=coordinates,
            depot=depot,
            spots_in_cluster=spots_in_cluster,
            num_ants=num_ants,
            alpha=alpha,
            beta=beta,
            rho=rho,
            Q=Q,
            num_iterations=num_iterations,
            distance=distance
        )
        # ����ʱ�䣨ʹ��ʵ�ʾ����ֵ���㣩
        cluster_time = calculate_time(
            route=optimal_route,
            speed=speed,
            loading_time=loading_time_per_bike,
            parking_spots=parking_spots,
            distance_map=distance  # ��������
        )
        
        
        print("����·��:", " �� ".join(optimal_route))
        
        total_time += cluster_time
        print(f"����ʱ��: {cluster_time:.2f} Сʱ")
    
    print("\n=== ���� ===")
    print(f"Ԥ��������ʱ��: {total_time:.2f} Сʱ")

if __name__ == "__main__":
    main()