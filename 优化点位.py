# -*- coding: gbk -*-
import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pulp as pl
#import 调度模型2

# 手动构建距离字典
distance_map = {
    "体育馆": {
        "体育馆": 0.0,
        "北门": 571.14061,
        "网球场": 1400.763758,
        "菊苑1栋": 1158.809453,
        "计算机学院": 791.174367,
        "工程中心": 1043.335862,
        "教学4楼": 1785.20351,
        "教学2楼": 1689.233101,
        "三食堂": 1453.843243,
        "南门": 1852.624092,
        "梅苑1栋": 1713.87693,
        "校医院": 1698.26569,
        "二食堂": 1400.431131,
        "一食堂": 998.028902,
        "东门": 1855.26
    },
    "北门": {
        "体育馆": 571.14061,
        "北门": 0.0,
        "网球场": 1434.169956,
        "菊苑1栋": 1192.21565,
        "计算机学院": 483.593335,
        "工程中心": 600.148647,
        "教学4楼": 1517.56585,
        "教学2楼": 1439.472415,
        "三食堂": 1487.249441,
        "南门": 1886.030289,
        "梅苑1栋": 1747.283127,
        "校医院": 1731.671887,
        "二食堂": 1433.837328,
        "一食堂": 1031.435099,
        "东门": 1437.11
    },
    "网球场": {
        "体育馆": 1400.763758,
        "北门": 1434.169956,
        "网球场": 0.0,
        "菊苑1栋": 241.954305,
        "计算机学院": 1310.111691,
        "工程中心": 1659.803956,
        "教学4楼": 1602.718534,
        "教学2楼": 1506.748126,
        "三食堂": 1271.358268,
        "南门": 1649.457588,
        "梅苑1栋": 1135.961237,
        "校医院": 1120.349997,
        "二食堂": 822.515438,
        "一食堂": 815.543926,
        "东门": 1221.46
    },
    "菊苑1栋": {
        "体育馆": 1158.809453,
        "北门": 1192.21565,
        "网球场": 241.954305,
        "菊苑1栋": 0.0,
        "计算机学院": 1068.157386,
        "工程中心": 1417.849651,
        "教学4楼": 1360.764229,
        "教学2楼": 1264.79382,
        "三食堂": 1029.403962,
        "南门": 1407.503283,
        "梅苑1栋": 894.006932,
        "校医院": 878.395692,
        "二食堂": 580.561133,
        "一食堂": 573.589621,
        "东门": 1826.99
    },
    "计算机学院": {
        "体育馆": 791.174367,
        "北门": 483.593335,
        "网球场": 1310.111691,
        "菊苑1栋": 1068.157386,
        "计算机学院": 0.0,
        "工程中心": 669.894318,
        "教学4楼": 1354.345358,
        "教学2楼": 1276.251922,
        "三食堂": 1363.191176,
        "南门": 1761.972025,
        "梅苑1栋": 1623.224863,
        "校医院": 1607.613623,
        "二食堂": 1309.779064,
        "一食堂": 907.376835,
        "东门": 1119.24
    },
    "工程中心": {
        "体育馆": 1043.335862,
        "北门": 600.148647,
        "网球场": 1659.803956,
        "菊苑1栋": 1417.849651,
        "计算机学院": 669.894318,
        "工程中心": 0.0,
        "教学4楼": 917.417203,
        "教学2楼": 839.323768,
        "三食堂": 1169.716635,
        "南门": 1506.067314,
        "梅苑1栋": 1843.66907,
        "校医院": 1957.305888,
        "二食堂": 1659.471329,
        "一食堂": 1257.069099,
        "东门": 650.29
    },
    "教学4楼": {
        "体育馆": 1785.20351,
        "北门": 1517.56585,
        "网球场": 1602.718534,
        "菊苑1栋": 1360.764229,
        "计算机学院": 1354.345358,
        "工程中心": 917.417203,
        "教学4楼": 0.0,
        "教学2楼": 343.199458,
        "三食堂": 469.692345,
        "南门": 806.043023,
        "梅苑1栋": 1143.64478,
        "校医院": 1352.485283,
        "二食堂": 1189.576838,
        "一食堂": 787.174608,
        "东门": 701.28
    },
    "教学2楼": {
        "体育馆": 1689.233101,
        "北门": 1439.472415,
        "网球场": 1506.748126,
        "菊苑1栋": 1264.79382,
        "计算机学院": 1276.251922,
        "工程中心": 839.323768,
        "教学4楼": 343.199458,
        "教学2楼": 0.0,
        "三食堂": 658.657778,
        "南门": 1057.438626,
        "梅苑1栋": 1389.474758,
        "校医院": 1391.440988,
        "二食堂": 1093.606429,
        "一食堂": 691.2042,
        "东门": 772.4
    },
    "三食堂": {
        "体育馆": 1453.843243,
        "北门": 1487.249441,
        "网球场": 1271.358268,
        "菊苑1栋": 1029.403962,
        "计算机学院": 1363.191176,
        "工程中心": 1169.716635,
        "教学4楼": 469.692345,
        "教学2楼": 658.657778,
        "三食堂": 0.0,
        "南门": 398.780849,
        "梅苑1栋": 736.382605,
        "校医院": 945.223108,
        "二食堂": 858.216571,
        "一食堂": 455.814342,
        "东门": 1380.72
    },
    "南门": {
        "体育馆": 1852.624092,
        "北门": 1886.030289,
        "网球场": 1649.457588,
        "菊苑1栋": 1407.503283,
        "计算机学院": 1761.972025,
        "工程中心": 1506.067314,
        "教学4楼": 806.043023,
        "教学2楼": 1057.438626,
        "三食堂": 398.780849,
        "南门": 0.0,
        "梅苑1栋": 513.496351,
        "校医院": 722.336854,
        "二食堂": 826.94215,
        "一食堂": 854.59519,
        "东门": 1840.8
    },
    "梅苑1栋": {
        "体育馆": 1713.87693,
        "北门": 1747.283127,
        "网球场": 1135.961237,
        "菊苑1栋": 894.006932,
        "计算机学院": 1623.224863,
        "工程中心": 1843.66907,
        "教学4楼": 1143.64478,
        "教学2楼": 1389.474758,
        "三食堂": 736.382605,
        "南门": 513.496351,
        "梅苑1栋": 0.0,
        "校医院": 208.840503,
        "二食堂": 313.445799,
        "一食堂": 715.848028,
        "东门": 1851.75
    },
    "校医院": {
        "体育馆": 1698.26569,
        "北门": 1731.671887,
        "网球场": 1120.349997,
        "菊苑1栋": 878.395692,
        "计算机学院": 1607.613623,
        "工程中心": 1957.305888,
        "教学4楼": 1352.485283,
        "教学2楼": 1391.440988,
        "三食堂": 945.223108,
        "南门": 722.336854,
        "梅苑1栋": 208.840503,
        "校医院": 0.0,
        "二食堂": 297.834559,
        "一食堂": 700.236788,
        "东门": 2121.3
    },
    "二食堂": {
        "体育馆": 1400.431131,
        "北门": 1433.837328,
        "网球场": 822.515438,
        "菊苑1栋": 580.561133,
        "计算机学院": 1309.779064,
        "工程中心": 1659.471329,
        "教学4楼": 1189.576838,
        "教学2楼": 1093.606429,
        "三食堂": 858.216571,
        "南门": 826.94215,
        "梅苑1栋": 313.445799,
        "校医院": 297.834559,
        "二食堂": 0.0,
        "一食堂": 402.402229,
        "东门": 2213.99
    },
    "一食堂": {
        "体育馆": 998.028902,
        "北门": 1031.435099,
        "网球场": 815.543926,
        "菊苑1栋": 573.589621,
        "计算机学院": 907.376835,
        "工程中心": 1257.069099,
        "教学4楼": 787.174608,
        "教学2楼": 691.2042,
        "三食堂": 455.814342,
        "南门": 854.59519,
        "梅苑1栋": 715.848028,
        "校医院": 700.236788,
        "二食堂": 402.402229,
        "一食堂": 0.0,
        "东门": 1822.03
    },
    "东门": {
        "体育馆": 1855.26,
        "北门": 1437.11,
        "网球场": 1221.46,
        "菊苑1栋": 1826.99,
        "计算机学院": 1119.24,
        "工程中心": 650.29,
        "教学4楼": 701.28,
        "教学2楼": 772.4,
        "三食堂": 1380.72,
        "南门": 1840.8,
        "梅苑1栋": 1851.75,
        "校医院": 2121.3,
        "二食堂": 2213.99,
        "一食堂": 1822.03,
        "东门": 0.0
    }
}

# 读取Excel文件
#file_path = "E:\\新建文件夹\\大二下\\华中杯\\评估\\停车点评估报告（1）.xlsx"
file_path = "E:\\新建文件夹\\大二下\\停车点评估报告（3）.xlsx"
report_df = pd.read_excel(file_path, engine="openpyxl")

def add_virtual_nodes(low_score_locs, distance_map):
    #为高需求低效区域添加虚拟停车点
    new_nodes = []
    for loc in low_score_locs:
            # 在距原节点500米范围内添加虚拟点
        virtual_loc = f"{loc}_虚拟点"
        distance_map[virtual_loc] = {
            k: v + 500 for k, v in distance_map[loc].items()
        }
        distance_map[virtual_loc][virtual_loc] = 0
        new_nodes.append(virtual_loc)
    return new_nodes

def validate_optimization(old_report, new_report, threshold=0.15):
    """验证优化效果"""
    improvement = {}
    for loc in new_report['地点']:
        old_score = old_report[old_report['地点'] == loc]['综合评分'].values[0]
        new_score = new_report[new_report['地点'] == loc]['综合评分'].values[0]
        improvement[loc] = new_score - old_score
    
    # 筛选显著改进点
    significant = [k for k, v in improvement.items() if v > threshold]
    print(f"显著改进点位（提升>{threshold}）：{significant}")

# 从评估报告中筛选低效停车点（综合评分低于阈值）
low_score_locations = report_df[report_df['综合评分'] < 0.6]['地点'].tolist()
print(f"需优先优化的停车点：{low_score_locations}")

#for loc in low_score_locations:
    #loc_data = report_df[report_df['地点'] == loc].iloc[0]
new_nodes = add_virtual_nodes(low_score_locations, distance_map)
print(f"新增虚拟停车点：{new_nodes}")

#调度模型2.locs=调度模型2.locs+new_nodes
#print(调度模型2.locs)
print(distance_map)
# 对比优化前后报告
#validate_optimization(original_report, optimized_report)
