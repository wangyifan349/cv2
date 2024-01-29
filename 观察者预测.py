import threading
import random
import time
from collections import Counter
import numpy as np
from scipy.stats import norm
class LongRandomProcess:
    def __init__(self, duration):
        self.duration = duration
        self.data = []
        self.lock = threading.Lock()
        self.running = True
        self.observed_data = []  # 存储观察到的数据
        self.miss_chance = 0.3  # 30% 的概率错过数据

    def random_process(self):
        # 记录函数开始执行的时间
        start_time = time.time()
    
        # 循环直到达到指定的持续时间
        while time.time() - start_time < self.duration:
            # 使用锁来确保数据添加操作的线程安全
            # 这是必要的，因为可能有其他线程（如观察者线程）尝试同时访问或修改self.data
            with self.lock:
                # 向self.data列表中添加一个随机整数，0或1
                # random.randint(0, 1)函数生成一个包含0和1的随机整数，包括0和1
                # 这里的随机性来源于Python的随机数生成器，它基于某种算法和初始种子值来生成伪随机数序列
                self.data.append(random.randint(0, 1))
        
            # 随机暂停一段时间，介于0.1到0.5秒之间
            # 这里的随机性同样来源于Python的随机数生成器
            # random.uniform(0.1, 0.5)生成一个指定范围内的随机浮点数
            # 这个暂停模拟了在现实世界中数据生成过程可能的不规律性
            time.sleep(random.uniform(0.1, 0.5))
    
        # 当循环结束，即过程运行时间达到指定的持续时间后，将running标志设置为False
        # 这表示随机过程已经结束
        self.running = False

    def observer(self):
        while self.running:
            time.sleep(random.uniform(1, 3))
            with self.lock:
                if self.data and random.random() > self.miss_chance:
                    self.observed_data.append(self.data[-1])

    def evaluate_using_average(self):
        if self.observed_data:
            observed_average = sum(self.observed_data) / len(self.observed_data)
            print(f"平均数策略 - 观察者猜测的数据平均值: {observed_average}")
        else:
            print("平均数策略 - 没有观察到数据。")

    def evaluate_using_median(self):
        if self.observed_data:
            sorted_data = sorted(self.observed_data)
            n = len(sorted_data)
            median = sorted_data[n // 2] if n % 2 != 0 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
            print(f"中位数策略 - 观察者猜测的数据中位数: {median}")
        else:
            print("中位数策略 - 没有观察到数据。")

    def evaluate_using_mode(self):
        if self.observed_data:
            data_counter = Counter(self.observed_data)
            mode = data_counter.most_common(1)[0][0]
            print(f"模式策略 - 观察者猜测的数据模式: {mode}")
        else:
            print("模式策略 - 没有观察到数据。")

    def evaluate_using_normal_distribution(self):
        if self.observed_data:
            # 计算观察到的数据的平均值和标准差
            mean = np.mean(self.observed_data)
            std = np.std(self.observed_data)
            # 假设观察到的数据符合正态分布，使用平均值和标准差作为分布参数
            # 这里我们可以使用这些参数来描述整个数据集的分布情况
            print(f"正态分布策略 - 观察到的数据平均值: {mean}, 标准差: {std}")
            # 作为示例，我们可以计算下一个数据点落在某个特定区间内的概率
            # 例如，计算下一个数据点落在平均值±1标准差区间内的概率
            prob = norm.cdf(mean + std, mean, std) - norm.cdf(mean - std, mean, std)
            print(f"下一个数据点落在平均值±1标准差区间内的概率: {prob}")
        else:
            print("正态分布策略 - 没有观察到数据。")



# 模拟持续时间（秒）
duration = 10

# 创建模拟实例
process = LongRandomProcess(duration)

# 创建并启动随机过程线程
thread_process = threading.Thread(target=process.random_process)
thread_process.start()

# 创建并启动观察者线程
thread_observer = threading.Thread(target=process.observer)
thread_observer.start()

# 等待所有线程完成
thread_process.join()
thread_observer.join()

# 使用不同的策略评估观察到的数据
process.evaluate_using_average()
process.evaluate_using_median()
process.evaluate_using_mode()
process.evaluate_using_normal_distribution()  # 调用新的观察方法
