import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pathlib import Path
from scipy import stats
import os
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用SimHei（黑体）

class PhysicsExperimentBasic:
    def __init__(self):
        self.__data__ = None
        self.slope = None # deprecating...
        self.intercept = None # deprecating...
        self.r_value = None
        self.p_value = None
        self.std_err = None
        self.xlike = None
        self.ylike = None
        self.yhat = None
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
        从键盘中读取数据
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
        self.slope, self.intercept, self.r_value, self.p_value, self.std_err = stats.linregress(self.xlike, self.ylike)

        def linear_regression_instance(x):
            return self.slope * x + self.intercept
        
        self.regression = linear_regression_instance
        self.regression_function_str = f"y = {self.slope:.4f}x + {self.intercept:.4f}"
        
        print("=====线性拟合结果=====\n")
        print(f"回归方程: {self.regression_function_str}")
        print(f"相关系数 R: {self.r_value:.4f}")
        print(f"决定系数R²: {self.r_value**2:.4f}")
        print(f"P值: {self.p_value:.4f}")
        print(f"标准误差: {self.std_err:.4f}")
        
        return True

    def polynomial(self, degree : int = 2):
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
            
            # 计算R²分数
            y_pred = target_polynomial(self.xlike, *popt)
            ss_res = np.sum((self.ylike - y_pred) ** 2)
            ss_tot = np.sum((self.ylike - np.mean(self.ylike)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            def fit_function(x):
                return target_polynomial(x, *popt)
            
            self.regression = fit_function
            
            print(f"====={degree}次多项式拟合结果=====\n")
            #print("拟合方程: y = ", end="")
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
        return self.polynomial(degree=2)
    
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

    # utils
    def __call__(self, *args, **kwds):
        return self.linear()
