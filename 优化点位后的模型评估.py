import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pulp as pl

#LOCS = ["东门", "南门", "北门", "一食堂", "二食堂", "三食堂", "梅苑1栋", "菊苑1栋", "教学2楼", "教学4楼", "计算机学院", "工程中心", "网球场", "体育馆", "校医院"]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体（Windows）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
# 读取Excel文件
file_path = "E:\新建文件夹\大二下\华中杯\B题：校园共享单车的调度与维护问题_1744791842432\附件\附件1-共享单车分布统计表.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")
file_path_d="E:\新建文件夹\大二下\华中杯\新距离.xlsx"
df_distance=pd.read_excel(file_path_d,engine="openpyxl")

# 定义ID到名称的映射关系
destinat_id_to_name = {
    1: "体育部",
    2: "北门",
    3: "网球场",
    4: "菊苑1栋",
    5: "计算机学院",
    6: "工程中心",
    7: "教学4楼",
    8: "教学2楼",
    9: "三食堂",
    10: "南门",
    11: "梅苑1栋",
    12: "校医院",
    13: "二食堂",
    14: "一食堂",
    15: "东门",
    16: "图书馆西门",   
    17: "电子科学学院",
    18: "广场",
    19: "彩虹球场",
    20: "实验楼"
}
print("所有列名:", df_distance.columns.tolist())
# 初始化嵌套字典
distance_map = {}
num=0
Num=0
# 遍历数据构建结构
for _, row in df_distance.iterrows():
    num=num+1
    if num%19==1:
        Num=Num+1

    origin_id = row["ObjectID"]
  
    distance = row["Total_Length"]
    
        # 获取起点名称
    origin_name = destinat_id_to_name.get(Num)
    if not origin_name: continue  # 跳过未定义的起点
    
        # 获取终点名称
        # 按 " - " 分割字符串，取后半部分并去除空格
    destinat_name = row["Name"].split(" - ")[1].strip()
    if not destinat_name: continue  # 跳过未定义的终点
   
        # 添加距离信息（保留6位小数）
    if num%19==1:
        distance_map[origin_name]={}
    distance_map[origin_name][destinat_name] = round(float(distance), 6)
'''
# 补全自身距离为0的条目
for origin in distance_map:
    if origin not in distance_map[origin]:
        distance_map[origin][origin] = 0.0'''

print(distance_map)

# 星期变成小时
df["Unnamed: 0"][0] = 0
df["Unnamed: 0"][8] = 24
df["Unnamed: 0"][16] = 48
df["Unnamed: 0"][24] = 72
df["Unnamed: 0"][29] = 96

# 定义地点列表
locs = ["东门", "南门", "北门", "一食堂", "二食堂", "三食堂", "梅苑1栋", "菊苑1栋", "教学2楼", "教学4楼", "计算机学院", "工程中心", "网球场", "体育馆", "校医院"]
new_locs=["图书馆西门","电子科学学院","广场","彩虹球场","实验楼"]
#locs=优化点位.NEW_LOCS

# 定义分类规则
categories = {
    "校门": ["东门", "南门", "北门"],
    "食堂": ["一食堂", "二食堂", "三食堂"],
    "宿舍": ["梅苑1栋", "菊苑1栋"],
    "教学楼": ["教学2楼", "教学4楼"],
    "其它": ["计算机学院", "工程中心", "网球场", "体育馆", "校医院"]
}

# 处理200+的数据
df.replace('200\+', 200, regex=True, inplace=True)

num = 0
for i in range(34):
    if not pd.isna(float(df["Unnamed: 0"][i])):
        df["Unnamed: 0"][i]
        num = df["Unnamed: 0"][i]
    else:
        df["Unnamed: 0"][i] = num


# 定义时间转换函数
def time_to_hours(t):
    try:
        hours, minutes, sec = map(float, t.split(":"))
        return hours + minutes / 60
    except:
        return np.nan  # 处理无效时间


# 应用转换并生成新列
df["Time"] = pd.to_datetime(df['Unnamed: 1'], format='%H:%M:%S', errors='coerce')
df['time'] = df['Time'].dt.hour + df['Time'].dt.minute / 60
df['time'] = df['time'] + df['Unnamed: 0']
time_column = df.pop('time')
df.insert(0, 'time', time_column)

# 删除原始列
df = df.drop(columns=["Unnamed: 0", "Unnamed: 1", "Time"])

# 把地点分类
# 定义地点列表
#locs = ["东门", "南门", "北门", "一食堂", "二食堂", "三食堂", "梅苑1栋", "菊苑1栋", "教学2楼", "教学4楼", "计算机学院", "工程中心", "网球场", "体育馆", "校医院"]
# 定义预测时间范围（7-23小时）
x_interp = np.linspace(7, 23, 200)

# 对每个地点插值
spline_functions = {}

SUM = np.zeros(200, dtype=float)
for location in df.columns[1:]:

    x = df['time'].to_numpy()
    y = df[location].to_numpy()

    # 填补缺失值为0
    y_filled = np.where(np.isnan(y), 0.0, y)

    # 只保留7-23小时区间
    mask = (x >= 7) & (x <= 23)
    x_filtered = x[mask]
    y_filtered = y_filled[mask]

    # 如果7、23点不在原数据里，手动补上，值取已有7点或23点平均（或者首尾值平均）
    if 7 not in x_filtered:
        x_filtered = np.insert(x_filtered, 0, 7)
        y_filtered = np.insert(y_filtered, 0, y_filtered[0])
    if 23 not in x_filtered:
        x_filtered = np.append(x_filtered, 23)
        y_filtered = np.append(y_filtered, y_filtered[-1])

        # 保证7点=23点
    avg_value = (y_filtered[0] + y_filtered[-1]) / 2
    y_filtered[0] = avg_value
    y_filtered[-1] = avg_value

    # 周期性三次样条插值，首尾相连（7=23）
    cs = CubicSpline(x_filtered, y_filtered, bc_type='periodic')
    spline_functions[location] = lambda t, cs=cs: np.clip(cs(t), 0, None)

    # 三次样条插值预测值（非负）
    y_spline = spline_functions[location](x_interp)


# 定义需求速率函数（取负导数，负值代表用车，非负）
def demand_rate(location, t, h=1e-5):
    dN_dt = (spline_functions[location](t) - spline_functions[location](t - h)) / (2 * h)
    return dN_dt


# 某时间段用车总需求量（积分）
def total_demand(location, t_start, t_end):
    result, _ = quad(lambda t: demand_rate(location, t), t_start, t_end)
    return result


# 调度模型核心代码
# --------------------------
# 步骤1：计算各停车点的供需缺口
# --------------------------
def calculate_supply_demand_gap(locations, peak_start, buffer_hours=1):
    gap_dict = {}
    time_ranges = [7,11,13,17,19]
    for loc in locations:
        if(loc in locs):
            current_count = spline_functions[loc](peak_start)
            future_demand = total_demand(loc, peak_start, peak_start + buffer_hours)
            ideal_inventory = current_count - future_demand * 1.0  # 理想库存为当前数量减去未来需求
            gap = current_count - ideal_inventory  # 供需缺口：正值表示供应过剩，负值表示需求不足
            gap_dict[loc] = gap
        for i in range(5):
            if peak_start==time_ranges[i]:
                if loc=='图书馆西门':
                    gap_dict[loc]=[5,-81,66,66,52][i]
                if loc=='电子科学学院':
                    gap_dict[loc]=[ -17,15,-26,19,-31][i]
                if loc=='广场':
                    gap_dict[loc]=[-16,-88,-49,-53,71][i]
                if loc=='彩虹球场':
                    gap_dict[loc]=[3,-8,-24,-16,85][i]
                if loc=='实验楼':
                    gap_dict[loc]=[-81,11,80,96,35][i]
    return gap_dict

# 示例：早高峰8:00前调度，提前1小时开始准备
peak_time =8
time_ranges = [7,11,13,17,19]
supply_demand_total={}
for i in range(len(time_ranges)):
    supply_demand_gap = calculate_supply_demand_gap(locs+new_locs, time_ranges[i])
    supply_demand_total[time_ranges[i]]=supply_demand_gap
# 打印供需缺口调试信息
#print("\n供需缺口调试信息:")
#for loc, gap in supply_demand_gap.items():
    #print(f"地点: {loc}, 当前车辆数: {spline_functions[loc](peak_time):.2f}, 未来需求: {total_demand(loc, peak_time, peak_time + 1):.2f}, 理想库存: {spline_functions[loc](peak_time) - total_demand(loc, peak_time, peak_time + 1):.2f}, 供需缺口: {gap:.2f}")

# --------------------------
# 步骤2：构建虚拟距离矩阵（使用实际的曲线距离）
# --------------------------
# 手动构建距离字典

# 获取两点间距离
def get_distance(from_loc, to_loc):
    try:
        return distance_map[from_loc][to_loc] / 1000  # 转换为公里
    except KeyError:
        try:
            return distance_map[to_loc][from_loc] / 1000  # 无向图
        except KeyError:
            return 2.0  # 默认距离（若未定义拓扑关系）

# 生成全距离矩阵
distance_matrix = pd.DataFrame(
    index=locs, columns=locs,
    data=[[get_distance(from_loc, to_loc) for to_loc in locs] for from_loc in locs]
)

# 打印距离字典和距离矩阵
print("\n距离字典:")
for from_loc in locs:
    print(f"{from_loc}: {distance_map.get(from_loc, {})}")

print("\n距离矩阵:")
print(distance_matrix)

# --------------------------
# 步骤3：调度算法实现（带时间约束的贪心算法）
# --------------------------
class Scheduler:
    def __init__(self, num_vehicles=3, capacity=20, speed=25):
        self.vehicles = [{
            'id': i,
            'capacity': capacity,
            'speed': speed,
            'current_location': '运维处',  # 初始位置
            'schedule': [],
            'used_time': 0.0
        } for i in range(num_vehicles)]
        # 新增：存储调度流数据 {period: {"调出点->调入点": 数量}}
        self.schedule_flows = {}  

    def find_nearest_vehicle(self, target_loc):
        min_time = float('inf')
        selected_vehicle = None
        for vehicle in self.vehicles:
            if vehicle['capacity'] <= 0:
                continue
            from_loc = vehicle['current_location']
            dist = get_distance(from_loc, target_loc)
            time_cost = dist / vehicle['speed'] * 60  # 分钟
            if time_cost < min_time:
                min_time = time_cost
                selected_vehicle = vehicle
        return selected_vehicle

    def dispatch(self, from_loc, to_loc, amount,peak_time):
        vehicle = self.find_nearest_vehicle(from_loc)
        if vehicle is None:
            print(f"没有可用的车辆可以从 {from_loc} 调度到 {to_loc}")
            return 0

        # 计算运输时间
        pickup_time = get_distance(vehicle['current_location'], from_loc) / vehicle['speed'] * 60
        transport_time = get_distance(from_loc, to_loc) / vehicle['speed'] * 60
        total_time = pickup_time + transport_time + amount * 1  # 装卸时间（1分钟/辆）

        # 检查时间约束（必须在高峰开始前完成）
        if total_time > (peak_time - 7) * 60:  # 假设从7点开始调度
            print(f"调度任务从 {from_loc} 到 {to_loc} 超出时间限制")
            return 0

        # 更新车辆状态
        actual_amount = min(amount, vehicle['capacity'])
        vehicle['capacity'] -= actual_amount
        vehicle['current_location'] = to_loc
        vehicle['used_time'] += total_time
        vehicle['schedule'].append({
            'from': from_loc,
            'to': to_loc,
            'amount': actual_amount,
            'time_cost': total_time
        })
        print(f"调度车辆 {vehicle['id']} 从 {from_loc} 到 {to_loc}: {actual_amount} 辆, 耗时 {total_time:.1f} 分钟")
        return actual_amount

    def batch_dispatch(self, supply_points, demand_points,period,peak_time):
        
        """新增 period 参数表示时段"""
        # 初始化当前时段的调度流
        self.schedule_flows[period] = {}  
        sorted_demand = sorted(demand_points.items(), key=lambda x: -x[1])
        for demand_loc, gap in sorted_demand:
            while gap > 0:
                nearest_supply = min(
                    supply_points.items(),
                    key=lambda x: get_distance(x[0], demand_loc) if x[1] > 0 else float('inf')
                )
                if nearest_supply[1] <= 0:
                    break
                transfer_amount = min(gap, nearest_supply[1], 20)
                # 调用 dispatch 并获取实际调度量
                actual = self.dispatch(nearest_supply[0], demand_loc, transfer_amount,peak_time)
                if actual > 0:
                    
                    flow_key = f"{nearest_supply[0]}->{demand_loc}"
                    self.schedule_flows[period][flow_key] = actual
                    supply_points[nearest_supply[0]] -= actual
                    gap -= actual
                else:
                    break
    def get(self, period):
        """提供外部访问接口"""
        return self.schedule_flows.get(period, {})

# --------------------------
# 步骤4：执行调度并输出结果
# --------------------------
supply_points={}
demand_points={}
for i in time_ranges:
    supply_points[i] = {loc: max(round(gap), 0) for loc, gap in supply_demand_total[i].items() if gap > 0}
    demand_points[i] = {loc: max(round(-gap), 0) for loc, gap in supply_demand_total[i].items() if gap < 0}


schedule_model = {}  # 存储各时段的调度器实例

for col in time_ranges:  # 遍历所有时段（如'7-9'）
    # 解析时段开始时间（如'7-9' -> 7）
    start_hour = col
    
    # 获取当前时段的供需数据
    supply_nodes = supply_points[col]
    demand_nodes = demand_points[col]
    
    # 创建新调度器实例（确保各时段独立）
    scheduler = Scheduler(num_vehicles=3, capacity=20, speed=25)
    scheduler.batch_dispatch(
        supply_points=supply_nodes.copy(),
        demand_points=demand_nodes.copy(),
        period=col,  # 时段标识（如'7-9'）
        peak_time=start_hour  # 时段开始时间（如7）
    )
    schedule_model[col] = scheduler
# ================== 评估模型核心 ==================
class LocationEvaluator:
    def __init__(self, time_weights=None):
        # 时段权重（如早高峰占更高权重）
        self.time_weights = time_weights or {7: 0.4, 11: 0.6,13:0.4,17:0.3,19:0.1}
        
        # 指标权重配置
        self.metric_weights = {
            'supply_util': 0.25,      # 供给利用率（实际调出/总供给）
            'demand_coverage': 0.25, # 需求覆盖率（实际调入/总需求）
            'cost_efficiency': 0.15,  # 调度成本效率（调度量/最大能力）
            'balance_index': 0.2,   # 供需平衡指数（1 - 基尼系数）
            'flow_quality': 0.15      # 流动质量（有效调度占比）
        }
    
    def gini_coefficient(self,values):
        # 确保数据非负且非空
        values = np.array(values)
        if np.any(values < 0):
            raise ValueError("数据必须非负")
        n = len(values)
        if n == 0:
            return 0.0
        # 计算绝对差的总和
        total = 0.0
        for i in range(n):
            for j in range(n):
                total += abs(values[i] - values[j])
        # 基尼系数公式
        mean = np.mean(values)
        return total / (2 * n**2 * mean) if mean != 0 else 0.0
    
    def evaluate_period(self, period, period_supply, period_demand, period_schedule, max_capacity):
        """单时段评估"""
        # ---------- 基础指标 ----------
        total_supply = sum(period_supply.values())
        total_demand = sum(period_demand.values())
        actual_flow = sum(period_schedule.values())
        
        # ---------- 关键指标计算 ----------
        metrics = {
            # 供给利用率 = 实际调出量 / 总供给量 
            'supply_util': actual_flow / total_supply if total_supply else 0,
            
            # 需求覆盖率 = 实际调入量 / 总需求量 
            'demand_coverage': actual_flow / total_demand if total_demand else 0,
            
            # 调度成本效率 = 实际调度量 / 最大调度能力 
            'cost_efficiency': actual_flow / max_capacity,
            
            # 供需平衡指数 = 1 - 基尼系数（供给量分布）
            'balance_index': 1 - self.gini_coefficient(list(period_supply.values())),
            
            # 流动质量 = 有效调度占比（排除小于5辆的微量调度）
            'flow_quality': sum(1 for v in period_schedule.values() if v >=5) / len(period_schedule) if period_schedule else 0
        }
        
        return metrics
    
    def evaluate_location(self, loc_name,schedule_model):
        """评估单个地点"""
        scores = {'periods': {}}
        
        # ---------- 分时段计算 ----------
        for period in supply_demand_total.keys():
            # 从调度模型获取对应时段的调度器
            scheduler = schedule_model.get(period)
            if not scheduler:
                continue
            
           # 获取该地点参与的调度流（支持字符串键）
            # 获取该时段的调度流数据
            period_flows = scheduler.get(period)
            related_flows = {}
            for flow_key, qty in period_flows.items():
            # 分割字符串键（格式："调出点->调入点"）
                if '->' in flow_key:
                    s, d = flow_key.split('->', 1)  # 最多分割一次，避免调入点含"-"
                    s = s.strip()  # 去除前后空格
                    d = d.strip()
                    if s == loc_name or d == loc_name:
                        related_flows[(s, d)] = qty
                else:
                    print(f"警告: 时段 {period} 的调度键 `{flow_key}` 格式错误，已跳过")
            
            # 构造评估参数
            loc_supply = supply_points[period].get(loc_name, 0)
            loc_demand = demand_points[period].get(loc_name, 0)
            metrics = self.evaluate_period(
                period, 
                period_supply={loc_name: loc_supply},
                period_demand={loc_name: loc_demand},
                period_schedule=related_flows,
                max_capacity=300
            )
            
            # 记录分时段结果
            scores['periods'][period] = {
                'metrics': metrics,
                'weighted_score': sum(
                    metrics[k] * self.metric_weights[k] 
                    for k in self.metric_weights
                )
            }
        
        # ---------- 综合评估 ----------
        # 时间加权总分
        total_score = sum(
            score['weighted_score'] * self.time_weights[period]
            for period, score in scores['periods'].items()
        )
        
        # 全局统计数据
        scores['total_supply'] = sum(
            supply_points[p].get(loc_name,0) for p in supply_points
        )
        scores['total_demand'] = sum(
            demand_points[p].get(loc_name,0) for p in demand_points
        )
        scores['overall_score'] = total_score
        
        return scores

# ================== 执行评估 ==================
def generate_report(evaluator,schedule_model):
    """生成评估报告"""
    report = {}
    
    # 获取所有相关地点（供给方+需求方）
    all_locations = set()
    for p in supply_points.values():
        all_locations.update(p.keys())
    for p in demand_points.values():
        all_locations.update(p.keys())
    
    # 评估每个地点
    for loc in all_locations:
        report[loc] = evaluator.evaluate_location(loc,schedule_model)
    
    # 转换为DataFrame
    df_data = []
    for loc, data in report.items():
        row = {
            '地点': loc,
            '总供给': data['total_supply'],
            '总需求': data['total_demand'],
            '综合评分': data['overall_score']
        }
        for period in data['periods']:
            for metric, value in data['periods'][period]['metrics'].items():
                row[f"{period}_{metric}"] = round(value, 3)
        df_data.append(row)
    
    return pd.DataFrame(df_data)

if __name__ == "__main__":
    # 初始化评估器（可自定义时段权重）
    evaluator = LocationEvaluator()
    
    # 生成报告
    report_df = generate_report(evaluator,schedule_model)
    
    # 保存并打印
    report_df.to_excel("停车点评估报告（2）.xlsx", index=False)
    print(report_df.to_markdown(index=False))