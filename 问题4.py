# -*- coding: gbk -*-
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import 调度模型2

# ================== 输入数据定义 ==================
# 假设优化后的停车点布局（包含坐标、总车辆数） 选择晚上九点
parking_spots = {
    '体育部': {'x': 13039873.05, 'y': 4058597.558, 'total': 4},
    '北门': {'x': 13040439.39, 'y': 4058474.212, 'total': 52},
    '网球场': {'x': 13039882.45, 'y': 4058168.561, 'total': 3},
    '菊苑1栋': {'x': 13039630.26, 'y': 4058273.857, 'total': 5},
    '计算机学院': {'x': 13040185.71, 'y': 4058092.577, 'total': 0},
    '工程中心': {'x': 13040528.62, 'y': 4057938.989, 'total': 59},
    '东门': {'x': 13040362.37, 'y': 4057491.841, 'total': 76},
    '教学4楼': {'x': 13039789.83, 'y': 4057363.079, 'total': 72},
    '教学2楼': {'x': 13039829.74, 'y': 4057732.176, 'total': 70},
    '三食堂': {'x': 13039440.37, 'y': 4057325.086, 'total': 2},
    '南门': {'x': 13039317.06, 'y': 4056875.487, 'total': 19},
    '梅苑1栋': {'x': 13039141.26, 'y': 4057377.553, 'total': 30},
    '校医院': {'x': 13039005.96, 'y': 4057544.19, 'total': 3},
    '二食堂': {'x': 13039181.9, 'y': 4057737.509, 'total': 35},
    '一食堂': {'x': 13039538.2, 'y': 4057841.582, 'total': 10},
    '图书馆西门': {'x': 13039762.12, 'y': 4057976.279, 'total': 0},
    '电子科学学院': {'x': 13040198.66, 'y': 4057838.718, 'total': 0},
    '广场': {'x': 13040045.96, 'y': 4057632.647, 'total': 0},
    '彩虹球场': {'x': 13039334.28, 'y': 4057933.14, 'total': 0},
    '实验楼': {'x': 13039294.34, 'y': 4057011.254, 'total': 0},
    '检修处':{'x':13040851.7209,'y':4058161.078400001,'total':0}
}

distance=调度模型2.distance_map

# 参数配置
failure_rate = 0.06      # 故障率 6%
vehicle_capacity = 20    # 运输车容量
speed = 25               # 行驶速度 (km/h)
loading_time_per_bike = 1 / 60  # 装车时间 (小时/辆)

# 算法参数配置
num_ants = 20        # 蚂蚁数量
alpha = 1            # 信息素重要程度因子
beta = 2             # 启发式因子
rho = 0.1           # 信息素挥发系数
Q = 100             # 信息素强度常数
num_iterations = 50 # 迭代次数
# ================== 数据预处理 ==================
def prepare_data():
    # 计算各点故障车辆数
    demands = {}
    coordinates = []
    for spot, info in parking_spots.items():
        if spot == '检修处':
            continue
        demand = int(info['total'] * failure_rate)
        demands[spot] = demand
        coordinates.append([info['x'], info['y']])
    
    # 检修处坐标
    depot = [parking_spots['检修处']['x'], parking_spots['检修处']['y']]
    
    return demands, np.array(coordinates), depot

# ================== 蚁群算法实现 ==================
class AntColonyOptimization:
    def __init__(self, distance_matrix, num_ants, alpha, beta, rho, Q, num_iterations):
        self.distance_matrix = distance_matrix  # 距离矩阵
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.num_iterations = num_iterations
        self.num_locations = len(distance_matrix)
        self.pheromone_matrix = np.ones((self.num_locations, self.num_locations))  # 初始化信息素矩阵
        self.best_path = None
        self.best_path_length = float('inf')
    
    def run(self):
        for _ in range(self.num_iterations):
            # 生成蚂蚁路径
            ant_paths = []
            ant_path_lengths = []
            for _ in range(self.num_ants):
                path, path_length = self.generate_ant_path()
                ant_paths.append(path)
                ant_path_lengths.append(path_length)
                
                # 更新最优路径
                if path_length < self.best_path_length:
                    self.best_path_length = path_length
                    self.best_path = path
            
            # 更新信息素
            self.update_pheromones(ant_paths, ant_path_lengths)
        
        return self.best_path, self.best_path_length
    
    def generate_ant_path(self):
        # 从检修处出发
        start_index = 0  # 假设检修处是第一个位置
        current_index = start_index
        path = [current_index]
        path_length = 0.0
        
        # 访问所有其他位置
        for _ in range(self.num_locations - 1):
            # 选择下一个位置
            next_index = self.select_next_location(current_index)
            path.append(next_index)
            path_length += self.distance_matrix[current_index][next_index]
            current_index = next_index
        
        # 返回检修处
        path.append(start_index)
        path_length += self.distance_matrix[current_index][start_index]
        
        return path, path_length
    
    def select_next_location(self, current_index):
        # 计算选择概率
        probabilities = np.zeros(self.num_locations)
        for i in range(self.num_locations):
            if i not in self.best_path and i != current_index:
                probabilities[i] = (self.pheromone_matrix[current_index][i] ** self.alpha) * \
                                   ((1.0 / self.distance_matrix[current_index][i]) ** self.beta)
        
        # 归一化概率
        probabilities /= probabilities.sum()
        
        # 随机选择下一个位置
        return np.random.choice(range(self.num_locations), p=probabilities)
    
    def update_pheromones(self, ant_paths, ant_path_lengths):
        # 信息素挥发
        self.pheromone_matrix *= (1.0 - self.rho)
        
        # 添加新信息素
        for path, path_length in zip(ant_paths, ant_path_lengths):
            for i in range(len(path) - 1):
                from_index = path[i]
                to_index = path[i + 1]
                self.pheromone_matrix[from_index][to_index] += self.Q / path_length

# ================== 聚类分趟（容量约束）==================

def cluster_points(demands, coordinates, depot):
    # 转换为特征矩阵 (坐标 + 需求量)
    features = []
    spots = list(demands.keys())
    for i, spot in enumerate(spots):
        features.append([coordinates[i][0], coordinates[i][1], demands[spot]])
    
    # K-means聚类（根据容量约束调整簇数）
    total_demand = sum(demands.values())
    n_clusters = int(np.ceil(total_demand / vehicle_capacity))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    clusters = kmeans.labels_
    
    # 分配簇并验证容量
    clusters_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in clusters_dict:
            clusters_dict[cluster_id] = {'spots': [], 'demand': 0}
        
        clusters_dict[cluster_id]['spots'].append(spots[i])
        clusters_dict[cluster_id]['demand'] += demands[spots[i]]
    
    return clusters_dict

# ================== 使用蚁群算法==================
def optimize_route_with_aco(coordinates, depot, spots_in_cluster, distance,**aco_params):
    """ 优化后的路径优化函数 """
    # 构建距离矩阵（使用预先生成的distance_map）
    num_spots = len(spots_in_cluster) + 1  # 包含检修处
    
    # 创建名称列表（检修处始终为第一个点）
    all_spots = ['检修处'] + spots_in_cluster
    
    # 初始化距离矩阵
    distance_matrix = np.zeros((num_spots, num_spots))
    for i in range(num_spots):
        for j in range(num_spots):
            from_spot = all_spots[i]
            to_spot = all_spots[j]
            # 优先使用distance_map中的距离
            if from_spot in distance and to_spot in distance[from_spot]:
                distance_matrix[i][j] = distance[from_spot][to_spot]
            else:
                # 后备计算（欧氏距离）
                coord_i = parking_spots[from_spot] if from_spot != '检修处' else {'x': depot[0], 'y': depot[1]}
                coord_j = parking_spots[to_spot] if to_spot != '检修处' else {'x': depot[0], 'y': depot[1]}
                distance = np.sqrt((coord_j['x']-coord_i['x'])**2 + (coord_j['y']-coord_i['y'])**2)
                distance_matrix[i][j] = distance
    
    # 运行蚁群算法
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
    
    # 转换索引为名称
    best_path = [all_spots[idx] for idx in best_path_indices]
    
    return best_path

# ================== 路径优化（使用OR-Tools，后用蚁群算法代替）==================

def optimize_route(coordinates, depot, spots_in_cluster):
    # 创建距离矩阵
    def create_distance_matrix(coords):
        num_locs = len(coords)
        dist_matrix = np.zeros((num_locs, num_locs))
        for i in range(num_locs):
            for j in range(num_locs):
                dist_matrix[i][j] = np.linalg.norm(coords[i]-coords[j])
        return dist_matrix
    
    # 调整坐标顺序（添加检修处）
    cluster_coords = [depot]  # 检修处作为起点
    spot_names = ['检修处']
    for spot in spots_in_cluster:
        cluster_coords.append([parking_spots[spot]['x'], parking_spots[spot]['y']]) 
        spot_names.append(spot)
    
    dist_matrix = create_distance_matrix(np.array(cluster_coords))
    
    # 创建路由模型
    manager = pywrapcp.RoutingIndexManager(len(cluster_coords), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # 求解设置
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    
    # 求解问题
    solution = routing.SolveWithParameters(search_parameters)
    
    # 提取路径
    route = []
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(spot_names[node])
            index = solution.Value(routing.NextVar(index))
        route.append(spot_names[manager.IndexToNode(index)])
    
    return route

# ================== 总时间计算 ==================
def calculate_time(route, speed, loading_time, parking_spots, distance_map):
    """ 优化后的时间计算函数 """
    total_distance = 0
    total_load_time = 0
    
    for i in range(len(route)-1):
        from_spot = route[i]
        to_spot = route[i+1]
        
        # 使用distance_map获取距离
        try:
            distance = distance_map[from_spot][to_spot]
        except KeyError:
            # 后备计算（欧氏距离）
            coord_from = parking_spots[from_spot] if from_spot != '检修处' else {'x': depot[0], 'y': depot[1]}
            coord_to = parking_spots[to_spot] if to_spot != '检修处' else {'x': depot[0], 'y': depot[1]}
            distance = np.sqrt((coord_to['x']-coord_from['x'])**2 + (coord_to['y']-coord_from['y'])**2)
        
        total_distance += distance
        
        # 装载时间计算
        if from_spot != '检修处':
            total_load_time += parking_spots[from_spot]['total'] * failure_rate * loading_time
    
    travel_time = total_distance / speed  # 秒
    return travel_time + total_load_time

def calculate_time_0(route, speed, loading_time):
    total_distance = 0
    total_load_time = 0
    depot = parking_spots['检修处']
    
    for i in range(len(route)-1):
        from_spot = route[i]
        to_spot = route[i+1]
        
        # 计算距离
        x1, y1 = parking_spots[from_spot]['x'], parking_spots[from_spot]['y']
        x2, y2 = parking_spots[to_spot]['x'], parking_spots[to_spot]['y']
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)/1000
        total_distance += distance
        
        # 计算装车时间（仅在非检修处装车）
        if from_spot != '检修处':
            total_load_time += parking_spots[from_spot]['total'] * failure_rate * loading_time
    
    travel_time = total_distance / speed  # 小时
    total_time = travel_time + total_load_time
    return total_time

# ================== 主程序 ==================
def main():
    # 数据准备
    demands, coordinates, depot = prepare_data()
    
    distance=调度模型2.distance_map

    # 聚类分趟
    clusters = cluster_points(demands, coordinates, depot)
    
    # 分簇优化路径
    total_time = 0
    for cluster_id, cluster_info in clusters.items():
        spots_in_cluster = cluster_info['spots']
        print(f"\n=== 簇 {cluster_id} ===")
        print(f"包含停车点: {spots_in_cluster}")
        print(f"总故障车辆: {cluster_info['demand']}")
        
        # 使用蚁群算法优化路径（重要修改点）
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
        # 计算时间（使用实际距离字典计算）
        cluster_time = calculate_time(
            route=optimal_route,
            speed=speed,
            loading_time=loading_time_per_bike,
            parking_spots=parking_spots,
            distance_map=distance  # 新增参数
        )
        
        
        print("最优路径:", " → ".join(optimal_route))
        
        total_time += cluster_time
        print(f"本趟时间: {cluster_time:.2f} 小时")
    
    print("\n=== 总览 ===")
    print(f"预计总运输时间: {total_time:.2f} 小时")

if __name__ == "__main__":
    main()