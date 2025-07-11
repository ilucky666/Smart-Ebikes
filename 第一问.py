import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.integrate import quad

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体（Windows）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常
# 读取Excel文件
file_path = "E:\\新建文件夹\\大二下\\华中杯\\B题：校园共享单车的调度与维护问题_1744791842432\\附件\\附件1-共享单车分布统计表.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")

#星期变成小时
df["Unnamed: 0"][0]=0
df["Unnamed: 0"][8]=24
df["Unnamed: 0"][16]=48
df["Unnamed: 0"][24]=72
df["Unnamed: 0"][29]=96

#处理200+的数据
df.replace('200\+', 200, regex=True, inplace=True)

num=0
for i in range(34):
    if not pd.isna(float(df["Unnamed: 0"][i])):
        df["Unnamed: 0"][i]
        num= df["Unnamed: 0"][i]
    else:
        df["Unnamed: 0"][i]=num

#定义时间转换函数
def time_to_hours(t):
    try:
        hours, minutes, sec = map(float,t.split(":"))
        return hours + minutes/60
    except:
        return np.nan  # 处理无效时间
#应用转换并生成新列
df["Time"] = pd.to_datetime(df['Unnamed: 1'],format='%H:%M:%S',errors='coerce')
df['time']=df['Time'].dt.hour+df['Time'].dt.minute/60
df['time']=df['time']+df['Unnamed: 0']
time_column = df.pop('time')
df.insert(0, 'time', time_column)

#删除原始列
df = df.drop(columns=["Unnamed: 0", "Unnamed: 1","Time"])

#把地点分类
# 定义地点列表
locs = ["东门", "南门", "北门", "一食堂", "二食堂", "三食堂", "梅苑1栋", "菊苑1栋", "教学2楼", "教学4楼", "计算机学院", "工程中心", "网球场", "体育馆", "校医院"]

# 定义分类规则
categories = {
    "校门": [{"name":"东门","y":[]}, 
            {"name":"南门","y":[]}, 
            {"name":"北门","y":[]}],
    "食堂": [{"name": "一食堂","y": []},
            {"name": "二食堂","y": []}],
    "宿舍": [{"name":"梅苑1栋", "y":[]},
           {"name":"菊苑1栋","y":[]}],
    "教学楼": [{"name":"教学2楼","y":[]},
            {"name":"教学4楼","y":[]}],
    "其它": [{"name":"计算机学院","y":[]},
           {"name":"工程中心", "y":[]},
           {"name":"网球场", "y":[]},
           {"name":"体育馆", "y":[]},
           {"name":"校医院","y":[]}]
}

#存放每一张图的数据
plot_map=np.empty((0,200),dtype=np.float64)

# 计算所有地点每日平均曲线，存成字典结构
def average_daily_profiles_all_locations(spline_functions, days=5):
    hours = np.arange(7, 23, 0.01)
    avg_profiles = {}

    for location, spline_func in spline_functions.items():
        avg_values = []
        for h in hours:
            values = []
            for d in range(days):
                value = spline_func(h + 24 * d)
                values.append(value)
            avg_values.append(np.mean(values))
        avg_profiles[location] = avg_values

    return hours, avg_profiles

def find_and_update_y(category_data, target_name, new_y):
    """
    在嵌套字典结构中查找指定name，并更新其y值为新数组
    :param category_data: 嵌套字典结构（格式：{父级key: [{'name': str, 'x': list, 'y': list}, ...]）
    :param target_name: 要查找的目标name字符串
    :param new_y: 要赋予的新y值（建议传入list类型）
    :return: 成功返回父级key，失败返回None
    """
    # 遍历每个分类（父级key）
    for category, locations in category_data.items():
        # 遍历该分类下的每个地点字典
        for location in locations:
            try:
                # 检查当前地点名称是否匹配目标
                if location['name'] == target_name:
                    # 更新y值（确保传入的是可迭代对象，这里不做强制类型转换）
                    location['y'] = new_y
                    return category  # 返回所属父级key
            except KeyError:
                # 处理字典缺少'name'键的异常情况
                print(f"警告：分类 '{category}' 下的某个地点缺少'name'键")
                continue
    # 遍历完成未找到
    print(f"未找到名称 '{target_name}' 对应的地点")
    return None

    # 定义保存路径
save_path = r"E:\\新建文件夹\\大二下\\华中杯\\拟合曲线2\\新建文件夹"
os.makedirs(save_path, exist_ok=True)  # 如果文件夹不存在就创建

    # 定义预测时间范围（7-23小时）
x_interp = np.linspace(7, 23, 200)

    # 对每个地点插值
spline_functions = {}

SUM=np.zeros(200,dtype=float)
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
    for i in range(len(x_interp)):
        SUM[i]=SUM[i]+y_spline[i]
        
    plot_map=np.vstack((plot_map,y_spline))
    
    #预测停车数量
    predict_time=[7,9,12,14,18,21,23]
    columns_out=['7','9','12','14','18','21','23']
    #predict_bike=np.zeros((16,7),dtype=float)
    values=[0,0,0,0,0]
    row=0
    day=0
    19
    for p_time in predict_time:
        for k in range(5):
            y_spline1 = spline_functions[location](p_time+24*k)
            #values[k]= combined_model(p_time+24*k, *popt)
            values[k]=y_spline1
            #value=predict_bikes(location, p_time)
            #Sum[predict_time.index(p_time)]=Sum[predict_time.index(p_time)]+value
            if values[k]<0:
                values[k]=0.0
        print(int(sum(values)/5))
        
    print("\n")
print(np.max(SUM))   

plt.figure(figsize=(12, 6))  # 设置画布大小

# 遍历每行数据
for i in range(3):
    plt.plot(plot_map[i], label=f"{locs[i]}")  # 绘制曲线并添加标签

# 添加图表元素
plt.title("校门")
plt.xlabel("时间（小时）")
plt.ylabel("停车数量")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
plt.grid(True, linestyle='--', alpha=0.6)

 # 保存图片
filename = os.path.join(save_path, "校门.png")
plt.savefig(filename,bbox_inches='tight')
plt.close()


# 遍历每行数据
for i in range(3,6):
    plt.plot(plot_map[i], label=f"{locs[i]}")  # 绘制曲线并添加标签

# 添加图表元素
plt.title("食堂")
plt.xlabel("时间（小时）")
plt.ylabel("停车数量")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
plt.grid(True, linestyle='--', alpha=0.6)

# 保存图片
filename = os.path.join(save_path, "食堂.png")
plt.savefig(filename,bbox_inches='tight')
plt.close()

# 遍历每行数据
for i in range(6,8):
    plt.plot(plot_map[i], label=f"{locs[i]}")  # 绘制曲线并添加标签

# 添加图表元素
plt.title("宿舍")
plt.xlabel("时间（小时）")
plt.ylabel("停车数量")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
plt.grid(True, linestyle='--', alpha=0.6)

# 保存图片
filename = os.path.join(save_path, "宿舍.png")
plt.savefig(filename,bbox_inches='tight')
plt.close()

# 遍历每行数据
for i in range(8,10):
    plt.plot(plot_map[i], label=f"{locs[i]}")  # 绘制曲线并添加标签

# 添加图表元素
plt.title("教学楼")
plt.xlabel("时间（小时）")
plt.ylabel("停车数量")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
plt.grid(True, linestyle='--', alpha=0.6)

# 保存图片
filename = os.path.join(save_path, "教学楼.png")
plt.savefig(filename,bbox_inches='tight')
plt.close()

# 遍历每行数据
for i in range(10,15):
    plt.plot(plot_map[i], label=f"{locs[i]}")  # 绘制曲线并添加标签

# 添加图表元素
plt.title("其它")
plt.xlabel("时间（小时）")
plt.ylabel("停车数量")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
plt.grid(True, linestyle='--', alpha=0.6)

# 保存图片
filename = os.path.join(save_path, "其它.png")
plt.savefig(filename,bbox_inches='tight')
plt.close()
