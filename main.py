import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import math
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用SimHei（黑体）
T_095 = {
    3:4.30,
    4:3.18,
    5:2.78,
    6:2.57,
    7:2.45,
    8:2.36,
    9:2.31,
    10:2.26,
    15:2.14,
    20:2.09
}

class PhysicsExperimentBasic:
    def __init__(self):
        self.__data__ = None
        self.xlike = None
        self.ylike = None
        self.r_sq = None
        self.regression = None
        self.regression_function_str = None
    
    # data_input:
    def read_csv(self, csvfile_path, x_col=0, y_col=1, header=0):
        """
        从CSV文件加载数据
        
        :param file_path: CSV文件路径
        :param x_col: 自变量列名或列索引, 默认为第1列
        :param y_col: 因变量列名或列索引, 默认为第2列
        :param header: 表头行, 默认为0

        :return: boolean, 是否读取成功
        """
        try:
            self.__data__ = pd.read_csv(csvfile_path, header=header)
            self.xlike = self.__data__[x_col] if isinstance(x_col, str) else self.__data__.iloc[:, x_col]
            self.ylike = self.__data__[y_col] if isinstance(y_col, str) else self.__data__.iloc[:, y_col]
            return True
        except Exception as e:
            print(f"读取数据时出错: {e}")
            return False
    
    def read_from_keyboard(self):
        """
        从键盘中读取x, y数据, 可以为任意浮点数值, 以end结束输入
        """
        print("请输入x的值")
        measurements_x = []
        while True:
            try:
                data = input("请输入一个测量值（输入'end'结束输入）：")
                if data.lower() == 'end':
                    break
                measurements_x.append(float(data))
            except ValueError:
                print("输入无效，请输入数字！")
        
        if len(measurements_x) == 0:
            print("没有输入任何测量值，结束输入")
        
        print("请输入y的值")
        measurements_y = []
        while True:
            try:
                data = input("请输入一个测量值（输入'end'结束输入）：")
                if data.lower() == 'end':
                    break
                measurements_y.append(float(data))
            except ValueError:
                print("输入无效，请输入数字！")
        
        if len(measurements_y) == 0:
            print("没有输入任何测量值，结束输入")
            return
        self.ylike=measurements_y
    
    # preprocessor:
    def purge(self):
        """除去数据中的NaN值"""
        mask = ~(np.isnan(self.xlike) | np.isnan(self.ylike))
        self.xlike = self.xlike[mask]
        self.ylike = self.ylike[mask]

    # regression:
    def linear(self):
        """执行线性回归拟合"""
        if self.__data__ is None:
            print("请先加载数据")
            return False
        
        # 使用scipy的stats进行线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.xlike, self.ylike)

        def linear_regression_instance(x):
            return slope * x + intercept
        
        self.r_sq = r_value**2
        self.regression = linear_regression_instance
        self.regression_function_str = f"y = {slope:.4f}x + {intercept:.4f}"
        
        print("=====线性拟合结果=====\n")
        print(f"回归方程: {self.regression_function_str}")
        print(f"相关系数 R: {r_value:.4f}")
        print(f"决定系数R²: {r_value**2:.4f}")
        print(f"P值: {p_value:.4f}")
        print(f"标准误差: {std_err:.4f}")
        
        return True

    def polynomial(self, degree : int = 2):
        """执行多项式拟合

        :param degree: 使用的多项式的次数
        """
        if len(self.xlike) == 0:
            raise ValueError("数据中没有有效的数值")
        
        def target_polynomial(x, *coefficients):
            result = 0
            for i, coef in enumerate(coefficients):
                result += coef * (x ** i)
            return result
        
        # 初始猜测参数（全设为1）
        initial_guess = np.ones(degree + 1)
        
        try:
            # 使用curve_fit进行多项式拟合
            popt, _ = curve_fit(target_polynomial, self.xlike, self.ylike, p0=initial_guess)
            
            y_pred = target_polynomial(self.xlike, *popt)
            ss_res = np.sum((self.ylike - y_pred) ** 2)
            ss_tot = np.sum((self.ylike - np.mean(self.ylike)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            def fit_function(x):
                return target_polynomial(x, *popt)
            
            self.r_sq = r_squared
            self.regression = fit_function
            
            print(f"====={degree}次多项式拟合结果=====\n")
            function_str = "y = "
            equation_parts = []
            for i, coef in enumerate(popt):
                if i == 0:
                    equation_parts.append(f"{coef:.6f}")
                else:
                    equation_parts.append(f"{coef:.6f} * x^{i}")
            function_str += " + ".join(equation_parts)
            self.regression_function_str = function_str
            print(f"拟合方程{self.regression_function_str}")
            print(f"决定系数R²: {r_squared:.6f}")
            
        except Exception as e:
            print(f"拟合过程中出现错误: {e}")

    def quadratic(self):
        """使用二次多项式拟合"""
        return self.polynomial(degree=2)
    
    def sinusoidal(self, function_type='basic', initial_guess=None, maxfev=10000):
        """
        进行正弦型函数拟合
        
        :param function_type: 正弦函数类型, 可选 'basic', 'damped', 'multi_freq'
        :param initial_guess: 初始参数猜测的列表, 如果为None则自动估计
        :param maxfev: 估计次数，如果出现拟合失败`Number of calls to function has reached maxfev`可以调高该参数 
        """
        def basic_sinusoidal(x, A, f, phi, offset):
            """基本正弦函数: A * sin(2π * f * x + phi) + offset"""
            return A * np.sin(2 * np.pi * f * x + phi) + offset
        
        def damped_sinusoidal(x, A, f, phi, decay, offset):
            """衰减正弦函数: A * sin(2π * f * x + phi) * exp(-decay * x) + offset"""
            return A * np.sin(2 * np.pi * f * x + phi) * np.exp(-decay * x) + offset
        
        def multi_freq_sinusoidal(x, A1, f1, phi1, A2, f2, phi2, offset):
            """双频率正弦函数: A1 * sin(2π * f1 * x + phi1) + A2 * sin(2π * f2 * x + phi2) + offset"""
            return A1 * np.sin(2 * np.pi * f1 * x + phi1) + A2 * np.sin(2 * np.pi * f2 * x + phi2) + offset
        
        if function_type == 'basic':
            fit_function = basic_sinusoidal
            param_names = ['振幅(A)', '频率(f)', '相位(φ)', '偏移量']
            if initial_guess is None:
                amplitude_guess = (np.max(self.ylike) - np.min(self.ylike)) / 2
                freq_guess = 1.0 / (np.max(self.xlike) - np.min(self.xlike)) if np.max(self.xlike) > np.min(self.xlike) else 0.1
                phase_guess = 0
                offset_guess = np.mean(self.ylike)
                initial_guess = [amplitude_guess, freq_guess, phase_guess, offset_guess]
        elif function_type == 'damped':
            fit_function = damped_sinusoidal
            param_names = ['振幅(A)', '频率(f)', '相位(φ)', '衰减系数', '偏移量']
            if initial_guess is None:
                amplitude_guess = (np.max(self.ylike) - np.min(self.ylike)) / 2
                freq_guess = 1.0 / (np.max(self.xlike) - np.min(self.xlike)) if np.max(self.xlike) > np.min(self.xlike) else 0.1
                phase_guess = 0
                decay_guess = 0.1
                offset_guess = np.mean(self.ylike)
                initial_guess = [amplitude_guess, freq_guess, phase_guess, decay_guess, offset_guess]
        elif function_type == 'multi_freq':
            fit_function = multi_freq_sinusoidal
            param_names = ['振幅1(A1)', '频率1(f1)', '相位1(φ1)', '振幅2(A2)', '频率2(f2)', '相位2(φ2)', '偏移量']
            if initial_guess is None:
                amplitude_guess = (np.max(self.ylike) - np.min(self.ylike)) / 4
                freq_guess = 1.0 / (np.max(self.xlike) - np.min(self.xlike)) if np.max(self.xlike) > np.min(self.xlike) else 0.1
                phase_guess = 0
                offset_guess = np.mean(self.ylike)
                initial_guess = [amplitude_guess, freq_guess, phase_guess, 
                            amplitude_guess/2, freq_guess*2, phase_guess, 
                            offset_guess]
        else:
            raise ValueError("不支持的函数类型，请检查你的输入")
        
        try:
            # 使用curve_fit进行正弦拟合
            popt, _ = curve_fit(fit_function, self.xlike, self.ylike, p0=initial_guess, maxfev=maxfev)
            
            y_pred = fit_function(self.xlike, *popt)
            ss_res = np.sum((self.ylike - y_pred) ** 2)
            ss_tot = np.sum((self.ylike - np.mean(self.ylike)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            def fitted_function(x):
                return fit_function(x, *popt)
            
            self.r_sq = r_squared
            self.regression = fitted_function
            
            print(f"\n=== {function_type} 正弦拟合结果 ===")
            print(f"决定系数R²: {r_squared:.6f}")
            print("拟合参数:")
            for name, value in zip(param_names, popt):
                print(f"  {name}: {value:.6f}")

            if function_type == 'basic':
                self.regression_function_str=f"y = {popt[0]} sin ( 2π * {popt[1]} x + {popt[2]} ) + {popt[3]}"
            elif function_type == "damped":
                self.regression_function_str=f"y = {popt[0]} sin ( 2π * {popt[1]} x + {popt[2]} ) exp ( - {popt[3]} x ) + {popt[4]}"
            elif function_type == "multi_freq":
                self.regression_function_str=f"y = {popt[0]} sin ( 2π * {popt[1]} x + {popt[2]} ) + {popt[3]} sin ( 2π {popt[4]} x + {popt[5]} ) + {popt[6]}"

            print(f"拟合方程: {self.regression_function_str}")

        except Exception as e:
            print(f"拟合过程中出现错误: {e}")
            print("请尝试调整初始猜测参数或使用不同的函数类型")

    # visualizing
    def plot(
            self, 
            title="拟合效果图", 
            xlabel="X", 
            ylabel="Y", 
            show_ui=False, 
            show_equation=True, 
            show_stat=False, 
            save=False, 
            file_path=None
        ):
        """
        绘制散点图和拟合直线
        
        :param title: 图表标题
        :param xlabel: X轴标签
        :param ylabel: Y轴标签
        :param show_ui: 是否直接显示该图片
        :param show_equation: 是否在图上显示回归方程
        :param show_stat: 是否在图片中显示数据点的坐标
        :param save: 是否保存png文件到目录中
        :param file_path: 保存路径，此参数在`save`参数为`False`时不起作用
        """
        if self.regression is None:
            print("请先执行拟合")
            return
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.xlike, self.ylike, alpha=0.7, color='blue', label='数据点')
        
        regression_line = self.regression(self.xlike)
        plt.plot(self.xlike, regression_line, color='red', linewidth=2, label='拟合直线')
        
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if show_equation:
            equation_text = f'y = {self.slope:.4f}x + {self.intercept:.4f}\nR² = {self.r_value**2:.4f}'
            plt.annotate(equation_text, xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        fontsize=12, ha='left')
        
        if show_stat:
            for i in range(len(self.xlike)):
                if i%2==0:
                    plt.text(self.xlike[i], self.ylike[i], f'({self.xlike[i]}, {self.ylike[i]})', ha='left', va='bottom', fontsize=10, color='blue')
                else:
                    plt.text(self.xlike[i], self.ylike[i], f'({self.xlike[i]}, {self.ylike[i]})', ha='left', va='top', fontsize=10, color='blue')

        plt.legend()
        plt.tight_layout()
        if save:
            if os.path.exists(file_path): 
                raise FileExistsError("There is a file with the same name.")
            else:
                plt.savefig(file_path)

        if show_ui:
            plt.show()

    def plot_2(self, xlabel="X", ylabel="Y"):
        pass

    # predict:
    def predict(self, x):
        """
        使用拟合的模型进行预测
        
        :param x_values: 要预测的x值
        
        :return: 预测的y值
        """
        if self.regression is None:
            print("请先执行拟合")
            return None
        
        return self.regression(x)
    
    # uncertainty analyze
    def calculate_uncertainty(self, data = "y"):
        if data == "x":
            measurements = self.xlike
        elif data == "y":
            measurements = self.ylike
        else:
            raise ValueError
        average = sum(measurements) / len(measurements)
        print(f"测量值的平均值为：{average:.6f}")
        
        # \Delta_A:
        if len(measurements) > 1:
            variance = sum((x - average) ** 2 for x in measurements) / (len(measurements) - 1)
            variance_std = math.sqrt(variance)
            print(f"贝塞尔标准偏差为：{variance_std:.6f}")
            u_a = T_095[len(measurements)]*math.sqrt(variance) / math.sqrt(len(measurements))
            print(f"在95%的置信水平下, A类不确定度(统计方法)为: {u_a:.6f}")
        else:
            print("只输入了一个测量值, 无法计算A类不确定度。")
            u_a = 0
        
        # \Delta_B
        try:
            u_b = float(input("请输入B类不确定度（非统计方法）："))
        except ValueError:
            print("输入无效，B类不确定度设为0。")
            u_b = 0

        # 计算合成不确定度
        u_c = math.sqrt(u_a ** 2 + u_b ** 2)
        print(f"合成不确定度为：{u_c:.6f}")
        u_r = u_c/average*100
        print(f"相对不确定度为：{u_r:.4f} %")
        
        # 计算扩展不确定度（置信水平95%）
        print(f"在95%的置信水平下，测量结果表示为：{average:.6f} ± {u_c:.6f}")

    # utils
    def get_status(self, verbose=True):
        if self.__data__ is None:
            print("没有存储任何数据。")
        else:
            if verbose:
                print(f"x: {self.xlike[0,5]}...")
                print(f"y: {self.ylike[0,5]}...")
                print(f"拟合方程: {self.regression_function_str}")
                print(f"使用此方程的决定系数: {self.r_sq}")

    def __call__(self, *args, **kwds):
        return self.get_status(*args, **kwds)
