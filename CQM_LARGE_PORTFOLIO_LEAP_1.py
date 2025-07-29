import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neal import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.system import LeapHybridSampler
from pyqubo import Array, Placeholder
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import datetime
from dimod import ConstrainedQuadraticModel, Binary, quicksum
from dwave.system import LeapHybridCQMSampler

budget = 1000000000
#target_return = 0.25
#target_volatility = 0.10
#gamma_values = [10]
#lambda_values = [1]
#theta_values = [1]
gamma_values = np.arange(1, 2, 1) 
lambda_values = np.arange(1, 2, 1) 
theta_values = np.arange(1, 2, 1) 
function_times = {}

def timeit(func):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        function_times[func.__name__] = end_time - start_time
        return result
    return timed

# 从yahoo finance下载数据 <class 'pandas.core.frame.DataFrame'>
@timeit
def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)  # 保留所有列
    data = data['Adj Close']  # 使用调整后收盘价计算收益率
    return data

# 从yahoo finance下载数据 <class 'pandas.core.frame.DataFrame'>
@timeit
def calculate_nikkei_metrics(start_date, end_date):
    nikkei_data = yf.download('^N225', start=start_date, end=end_date, auto_adjust=False)['Close']
    nikkei_returns = nikkei_data.pct_change().dropna()
    nikkei_annual_return = nikkei_returns.mean() * 252
    nikkei_annual_volatility = nikkei_returns.std() * np.sqrt(252)
    return nikkei_annual_return, nikkei_annual_volatility

# 计算财务指标
@timeit
def calculate_metrics(data):
    # returns:[m rows x n columns], m = trading days, n = stocks, <class 'pandas.core.frame.DataFrame'>
    returns = data.pct_change().dropna()
    # annual_returns: dtype: float64, <class 'pandas.core.series.Series'>, annual trading days = 252
    annual_returns = returns.mean() * 252
    # annual_volatility: dtype: float64, <class 'pandas.core.series.Series'>, annual trading days = 252
    annual_volatility = returns.var() * 252
    # cov_matrix: <class 'pandas.core.frame.DataFrame'>
    cov_matrix = returns.cov() * 252
    return annual_returns, annual_volatility, cov_matrix

# 基于budget进行归一化
@timeit
def fractional_formulation(data, annual_returns, annual_volatility, cov_matrix):
     # 获取最后一个交易日的收盘价
    last_closing_prices = data.iloc[-1]
    fractional_price = last_closing_prices / budget
    fractional_return = fractional_price * annual_returns
    # fractional_volatility似乎没有计算的必要
    fractional_volatility = fractional_price * annual_volatility
    # 外积矩阵
    fractional_price_hadamar = np.outer(fractional_price, np.ones_like(fractional_price))
    # 将协方差矩阵调整为分数形式
    fractional_cov_matrix = (fractional_price_hadamar * cov_matrix).transpose() * fractional_price_hadamar
    return last_closing_prices, fractional_price, fractional_return, fractional_volatility, fractional_cov_matrix

@timeit
def Quantum_formulation(last_closing_prices):
    # 计算给定预算下单只股票的最大购数量, dtype: float64, <class 'pandas.core.series.Series'>
    nmax = budget / last_closing_prices
    # 计算给定预算下单只股票的最大购数量，向下取整, dtype: int32, <class 'pandas.core.series.Series'>
    nmax_int = np.floor(budget / last_closing_prices).astype(int)
    # 计算表示每只股票最大购买数量所需的比特数, 这个比特数表示在量子模型中, 计算表示每只股票最大购买数量所需的比特数
    demanded_bits = nmax_int.apply(lambda x: x.bit_length())
    return nmax, nmax_int, demanded_bits

# 构建一个二进制转换矩阵 matrix_c, 将股票的购买数量表示为二进制形式。每个股票的数量需要通过二进制编码表示，这个矩阵将帮助将这些数量从二进制形式转换为实际购买的数量
@timeit
def binary_conversion_matrix(last_closing_prices):
    # 向下取整，<class 'list'>
    dimensions = [int(np.floor(np.log2(budget / p))) for p in last_closing_prices]
    # 计算转换矩阵 matrix_c 的列数，每个股票的比特位数会有一个额外的位，表示 2^0
    total_columns = sum(dimensions) + len(dimensions)
    # 初始化一个全零矩阵 matrix_c，行数等于股票数量，列数等于之前计算的 total_columns
    matrix_c = np.zeros((len(last_closing_prices), total_columns))
    # 初始化列索引，用于矩阵的填充，用来追踪当前要填充的列位置
    column_index = 0
    # 遍历每只股票，逐行填充矩阵，通过循环对每只股票的比特位进行处理，构建矩阵
    for i in range(len(last_closing_prices)):
        # 对每只股票的每个二进制位填充转换系数，对于第 i 个股票，dimensions[i] 决定了需要多少个比特位。每个比特位用相应的二进制系数填充，即2^j，例如，第一位对应2^0,
        # 第二位对应2^1，依次类推, 假设 dimensions[i] = 2，则矩阵的这一行将填充 [2^0, 2^1, 2^2]，即 [1, 2, 4]
        for j in range(dimensions[i] + 1):
            # 将每个比特位的值填入矩阵 matrix_c
            matrix_c[i, column_index + j] = 2 ** j
        # 更新 column_index，为下一个股票的比特位填充位置。
        column_index += dimensions[i] + 1
    return matrix_c

# 通过将投资组合的二进制表示与收益和风险矩阵结合，计算出基于二进制位表示的投资组合的收益（portfolio_bits_return）和
# 基于二进制位表示的投资组合的波动性（portfolio_bits_volatility）
@timeit
def portfolio_bits_formulation(matrix_c, fractional_return, fractional_cov_matrix):
    # 计算投资组合基于二进制位表示的预期收益, fractional_return 是一个向量，表示每只股票按分数计算的年化收益率。
    # matrix_c 是一个二进制转换矩阵，将二进制位转换为股票的实际数量。
    # matrix_c 矩阵中的每一行表示将某个资产的二进制位转换为实际投资数量。通过乘以 fractional_return，可以得到每个二进制位对应的预期收益
    portfolio_bits_return = np.dot(matrix_c.transpose(), fractional_return)
    # 计算投资组合基于二进制位表示的波动率, fractional_cov_matrix 是分数化后的协方差矩阵，表示资产间的相关性以及每个资产的波动性。
    # 通过矩阵乘法，matrix_c 的转置、协方差矩阵、matrix_c 连续相乘，计算出组合在二进制表示下的波动性。
    # 将二进制位映射到资产的实际数量，再用这些数量去计算投资组合的整体风险。最终得到的是以二进制位形式表示的投资组合的波动性
    portfolio_bits_volatility = np.dot(np.dot(matrix_c.transpose(), fractional_cov_matrix), matrix_c)
    return portfolio_bits_return, portfolio_bits_volatility

@timeit
# 构建 CQM 模型，包含目标函数和三类约束（预算、收益、风险）
def build_cqm(portfolio_bits_return, portfolio_bits_volatility, matrix_c,
              fractional_price, fractional_return, target_return, target_volatility,
              last_closing_prices): 

    # 定义二进制变量 b[i]，用于表示投资决策（选中或不选中每个组合项）
    b = [Binary(f"b[{i}]") for i in range(len(portfolio_bits_return))]

    # 目标函数：最大化组合收益 -> 最小化负收益
    objective = - quicksum(b[i] * portfolio_bits_return[i] for i in range(len(b)))

    # 初始化 CQM 模型
    cqm = ConstrainedQuadraticModel()
    cqm.set_objective(objective)

    # 添加预算约束：组合所花成本不能超过总预算（原始数值，不做归一）
    budget_expr = quicksum((matrix_c @ b)[i] * last_closing_prices.iloc[i] for i in range(len(last_closing_prices)))
    cqm.add_constraint(budget_expr <= budget, label="BudgetConstraint")


    # 添加收益约束：组合期望收益需达到目标值
    return_expr = quicksum((matrix_c @ b)[i] * fractional_return.iloc[i] for i in range(len(fractional_return)))
    cqm.add_constraint(return_expr >= target_return, label="ReturnConstraint")

    # 添加风险约束：组合波动率需小于设定阈值（平方）
    vol_expr = quicksum(
        b[i] * portfolio_bits_volatility[i, j] * b[j]
        for i in range(len(b)) for j in range(len(b))
    )
    cqm.add_constraint(vol_expr <= target_volatility ** 2, label="RiskConstraint")

    return cqm, b

@timeit
# 使用 D-Wave CQM 混合求解器进行一次求解，返回最优二进制解和耗时
def solve_cqm_model(cqm, b_vars):
    sampler = LeapHybridCQMSampler()

    start_time = time.time()
    sampleset = sampler.sample_cqm(cqm, label="Single CQM Portfolio Optimization", time_limit=40)
    end_time = time.time()
    cqm_solution_time = end_time - start_time

    # 提取可行解（满足所有约束）
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
    if len(feasible_sampleset) == 0:
        print("No feasible solution found.")
        return None, cqm_solution_time

    # 提取能量最低的可行解作为最优解
    best_sample = feasible_sampleset.first.sample
    b_solution = np.array([best_sample[f"b[{i}]"] for i in range(len(b_vars))])
    return b_solution, cqm_solution_time

@timeit
def solve_multiple_cqm_models(cqm, b_vars, gamma_values, lambda_values, theta_values):
    solutions = []
    solution_times = []
    total_solutions = 0

    for gamma_value in gamma_values:
        for lambda_value in lambda_values:
            for theta_value in theta_values:
                # 打印模型变量与约束信息
                print(f"[模型调试] 当前变量数量: {len(b_vars)}")
                print(f"[模型调试] 当前约束数量: {len(cqm.constraints)}")
                try:
                    from dimod import cqm_to_bqm
                    bqm = cqm_to_bqm(cqm)
                    print(f"[模型调试] 转换为BQM后变量数量: {len(bqm.variables)}")
                except Exception as e:
                    print(f"[模型调试] 无法转换为BQM: {e}")

                b_solution, solve_time = solve_cqm_model(cqm, b_vars)

                if b_solution is None:
                    print(f"No feasible solution for gamma={gamma_value}, lambda={lambda_value}, theta={theta_value}")
                    continue

                print(f"CQM solve time (gamma={gamma_value}, lambda={lambda_value}, theta={theta_value}): {solve_time:.4f} seconds")

                solutions.append((gamma_value, lambda_value, theta_value, b_solution))
                solution_times.append((gamma_value, lambda_value, theta_value, solve_time))
                total_solutions += 1

    return solutions, solution_times, total_solutions

@timeit
# 分析求解结果并返回 DataFrame
def result(matrix_c, solutions, tickers, fractional_return, last_closing_prices, cov_matrix, annual_returns, budget):
    results_list = []
    analysis_data = []
    risk_free_rate = 0.0

    for gamma_value, lambda_value, theta_value, solution in solutions:
        if solution.shape[0] != matrix_c.shape[1]:
            raise ValueError(f"解的维度 {solution.shape} 与 matrix_c 列数 {matrix_c.shape[1]} 不一致")

        binary_weight_matrix = np.zeros_like(matrix_c)
        flat_index = 0
        for i in range(matrix_c.shape[0]):
            for j in range(matrix_c.shape[1]):
                if matrix_c[i, j] != 0:
                    binary_weight_matrix[i, j] = solution[flat_index]
                    flat_index += 1

        ticker_quantities = np.sum(binary_weight_matrix * matrix_c, axis=1)
        asset_values = np.dot(matrix_c, solution) * last_closing_prices
        asset_weights = asset_values / np.sum(asset_values)

        portfolio_volatility = np.sqrt(np.dot(asset_weights.T, np.dot(cov_matrix, asset_weights)))
        portfolio_return = np.dot(asset_weights, annual_returns)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 1e-6 else 0.0

        return_contribution = asset_weights * annual_returns
        capital_consumed = ticker_quantities * last_closing_prices
        total_capital_consumed = np.sum(capital_consumed)
        capital_diff = abs(budget - total_capital_consumed)
        capital_diff_percent = (capital_diff / budget) * 100

        analysis_data.append({
            'gamma': gamma_value,
            'lambda': lambda_value,
            'theta': theta_value,
            'total_return': portfolio_return,
            'total_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'capital_diff': capital_diff,
            'capital_diff_percent': capital_diff_percent
        })

        ticker_quantities_df = pd.DataFrame({
            'Ticker': tickers,
            'Quantity': ticker_quantities,
            'Return Contribution': return_contribution,
            'Capital Consumed': capital_consumed
        })

        total_contributions_df = pd.DataFrame({
            'Gamma': [gamma_value],
            'Lambda': [lambda_value],
            'Theta': [theta_value],
            'Total Return': [portfolio_return],
            'Total Volatility': [portfolio_volatility],
            'Sharpe Ratio': [sharpe_ratio],
            'Total Capital Consumed': [total_capital_consumed]
        })

        results_list.append((gamma_value, lambda_value, theta_value, ticker_quantities_df, total_contributions_df))

    return results_list, pd.DataFrame(analysis_data)


# 可视化分析参数与目标函数指标关系（Total Return、Volatility、Sharpe Ratio）
@timeit
def plot_analysis(analysis_df, solver_name):
    plt.figure(figsize=(18, 12))

    metrics = [
        ('total_return', 'Total Return'),
        ('total_volatility', 'Total Volatility'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('capital_diff', 'Capital Difference')
    ]
    params = ['gamma', 'lambda', 'theta']

    plot_idx = 1
    for metric, metric_label in metrics:
        for param in params:
            if plot_idx > 12:
                break
            plt.subplot(4, 3, plot_idx)
            analysis_df.groupby(param)[metric].mean().plot(title=f'{metric_label} vs {param.capitalize()}')
            plt.xlabel(param.capitalize())
            plt.ylabel(metric_label)
            plot_idx += 1

    plt.tight_layout()

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f'analysis_plots_{solver_name}_{current_time}.png'
    plt.savefig(plot_file)
    plt.show()

    return plot_file

# 功能是基于给定的资产权重、收益率和协方差矩阵，计算一个投资组合的预期收益和波动率（标准差）。
# 计算投资组合的期望收益和波动率（标准差）
@timeit
def portfolio_performance(weights, returns, covariance):
    portfolio_return = np.dot(weights, returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
    return portfolio_return, portfolio_std


# 定义了一个用于最小化投资组合方差（即最小化组合的波动率）的目标函数 min_variance，并将其作为优化过程的一部分。
# 最小化投资组合波动率（作为目标函数）
@timeit
def min_variance(weights, returns, covariance):
    # 返回 portfolio_performance 的波动率部分（索引 1）
    return portfolio_performance(weights, returns, covariance)[1]

# 使用 SLSQP 方法优化投资组合，目标为最小化波动率并返回最优组合和 Sharpe 比率
@timeit
def optimize_portfolio(returns, covariance):
    num_assets = len(returns)  # 资产数量
    args = (returns, covariance)  # 参数打包传递

    # 限制权重和为 1（不允许做空）
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets]

    # 最小化组合的波动率
    result = minimize(min_variance, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    optimized_weights = result.x
    portfolio_return, portfolio_volatility = portfolio_performance(optimized_weights, returns, covariance)
    sharpe_ratio = portfolio_return / portfolio_volatility  # 假设无风险利率为 0

    return result, sharpe_ratio

@timeit
def plot_efficient_frontier_with_qubo(returns, covariance, qubo_solutions, matrix_c, 
                                      last_closing_prices, solver_name, num_portfolios=10000):
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(returns))
        weights /= np.sum(weights)
        ret, vol = portfolio_performance(weights, returns, covariance)
        results[0, i] = vol
        results[1, i] = ret
        results[2, i] = ret / vol if vol != 0 else 0

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]

    print(f"Minimum Volatility: {sdp_min:.4f}")
    print(f"Return at Minimum Volatility: {rp_min:.4f}")

    plt.figure(figsize=(10, 7))
    plt.scatter(results[0], results[1], c=results[2], cmap='YlGnBu', alpha=0.3)
    plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Max Sharpe (SLSQP)')
    plt.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Min Volatility')

    for i, (gamma, lambd, theta, solution) in enumerate(qubo_solutions):
        asset_values = np.dot(matrix_c, solution) * last_closing_prices
        weights = asset_values / np.sum(asset_values)
        qubo_vol = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        qubo_ret = np.dot(weights, returns)
        label = f'QUBO (γ={gamma}, λ={lambd}, θ={theta})' if i < 5 else None
        plt.scatter(qubo_vol, qubo_ret, marker='x', s=200, label=label)

    plt.title('Efficient Frontier with QUBO Solutions')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.tight_layout()

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f'efficient_frontier_{solver_name}_{current_time}.png'
    plt.savefig(plot_file)
    plt.show()

    return plot_file, sdp_min, rp_min



@timeit
def visualize_data(annual_returns, annual_volatility, cov_matrix):
    import matplotlib.ticker as ticker

    # 可视化年化收益率与波动率
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    annual_returns.plot(kind='bar', title='Annual Returns', color='skyblue')
    plt.ylabel('Annual Return')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    annual_volatility.plot(kind='bar', title='Annual Volatility', color='lightcoral')
    plt.ylabel('Annual Volatility')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # 可视化协方差矩阵
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cov_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar(im)

    tick_labels = cov_matrix.columns if hasattr(cov_matrix, 'columns') else range(len(cov_matrix))
    plt.xticks(ticks=range(len(tick_labels)), labels=tick_labels, rotation=90)
    plt.yticks(ticks=range(len(tick_labels)), labels=tick_labels)

    plt.title('Covariance Matrix')
    plt.tight_layout()
    plt.show()


@timeit
def save_to_excel(data, annual_returns, annual_volatility, cov_matrix, nmax, nmax_int, demanded_bits, 
                  last_closing_prices, fractional_price, fractional_return, fractional_volatility, fractional_cov_matrix, matrix_c, 
                  portfolio_bits_return, portfolio_bits_volatility, results_list, output_path, best_sharpe_ratio=None, 
                  target_return=None, target_volatility=None, min_volatility=None, min_vol_return=None):
    
    with pd.ExcelWriter(output_path) as writer:
        # 保存之前的sheet数据
        data.to_excel(writer, sheet_name='Historical Prices')
        annual_returns.to_excel(writer, sheet_name='Annual Returns')
        annual_volatility.to_excel(writer, sheet_name='Annual Volatility')

        qf_df = pd.DataFrame({
            'Max N': nmax,
            'Max Integral N': nmax_int,
            'Bit Length': demanded_bits,
        })
        qf_df.to_excel(writer, sheet_name='QF')
        
        ff_df = pd.DataFrame({
            'Fractional Price': fractional_price.values,
            'Fractional Return': fractional_return.values,
            'Fractional Volatility': fractional_volatility.values
        }, index=fractional_price.index)
        ff_df.to_excel(writer, sheet_name='Fractional Data', startrow=0)
        
        fractional_cov_matrix_df = pd.DataFrame(fractional_cov_matrix, index=fractional_price.index, columns=fractional_price.index)
        fractional_cov_matrix_df.to_excel(writer, sheet_name='Fractional Data', startrow=len(ff_df) + 2)
        matrix_c_df = pd.DataFrame(matrix_c)
        matrix_c_df.to_excel(writer, sheet_name='Matrix C')
        
        portfolio_bits_return_df = pd.DataFrame(portfolio_bits_return.flatten(), columns=['Portfolio Bits Return'])
        portfolio_bits_volatility_df = pd.DataFrame(portfolio_bits_volatility.flatten(), columns=['Portfolio Bits Volatility'])
        
        #portfolio_bits_return_df.to_excel(writer, sheet_name='Portfolio Bits Return')
        #portfolio_bits_volatility_df.to_excel(writer, sheet_name='Portfolio Bits Volatility')

        combined_results_df = pd.DataFrame()
        for i, (gamma_value, lambda_value, theta_value, ticker_quantities_df, total_contributions_df) in enumerate(results_list):
            ticker_quantities_df.to_excel(writer, sheet_name=f'QUBO_{i}_Opt_Info')
            total_contributions_df.to_excel(writer, sheet_name=f'QUBO_{i}_Results')
            
            summary_row = total_contributions_df.copy()
            summary_row['Gamma'] = gamma_value
            summary_row['Lambda'] = lambda_value
            summary_row['Theta'] = theta_value
            # 计算资本消耗与预算差额的绝对值和百分比
            capital_diff = abs(budget - total_contributions_df['Total Capital Consumed'].iloc[0])
            capital_diff_percent = (capital_diff / budget) * 100
            summary_row['Capital Difference (Abs)'] = capital_diff
            summary_row['Capital Difference (%)'] = capital_diff_percent
            
            combined_results_df = pd.concat([combined_results_df, summary_row])

        combined_results_df.to_excel(writer, sheet_name='Summary of Results', index=False)

        # 新增"Optimize Portfolio Summary"页保存Sharpe Ratio信息
        if best_sharpe_ratio is not None and target_return is not None and target_volatility is not None:
            opt_summary_df = pd.DataFrame({
                'Best Sharpe Ratio': [best_sharpe_ratio],
                'Target Return': [target_return],
                'Target Volatility': [target_volatility]
            })
            opt_summary_df.to_excel(writer, sheet_name='Optimize Portfolio Summary', index=False)
        
        # 新增"Minimum Volatility"页保存最小波动率和对应的收益率
        if min_volatility is not None and min_vol_return is not None:
            min_vol_df = pd.DataFrame({
                'Minimum Volatility': [min_volatility],
                'Return at Minimum Volatility': [min_vol_return]
            })
            min_vol_df.to_excel(writer, sheet_name='Minimum Volatility', index=False)

def main():
    global target_return, target_volatility

    tickers = [
    '1332.T', '1333.T', '1376.T', '1721.T', '1801.T', '1802.T', '1803.T', '1808.T', '1812.T', 
    '1925.T', '1928.T', '1963.T', '2002.T', '2264.T', '2282.T', '2413.T', '2432.T', '2768.T', 
    '2801.T', '2802.T', '2871.T', '2914.T', '3101.T', '3103.T', '3401.T', '3402.T', '3861.T', 
    '3863.T', '4004.T', '4005.T', '4021.T', '4042.T', '4043.T', '4061.T', '4151.T', '4183.T', 
    '4188.T', '4208.T', '4272.T', '4307.T', '4452.T', '4502.T', '4503.T', '4506.T', '4507.T', 
    '4519.T', '4523.T', '4568.T', '4578.T', '4689.T', '4704.T', '4751.T', '4755.T', '4901.T', 
    '4902.T', '4911.T', '5020.T', '5101.T', '5108.T', '5201.T', '5202.T', '5214.T', '5232.T', 
    '5233.T', '5301.T', '5332.T', '5333.T', '5401.T', '5406.T', '5411.T', '5541.T', '5631.T', 
    '5703.T', '5706.T', '5711.T', '5713.T', '5714.T', '5801.T', '5802.T', '5803.T', '5901.T', 
    '6103.T', '6113.T', '6301.T', '6302.T', '6305.T', '6326.T', '6361.T', '6367.T', '6471.T', 
    '6472.T', '6473.T', '6479.T', '6501.T', '6503.T', '6504.T', '6506.T', '6645.T', '6674.T', 
    '6701.T', '6702.T', '6703.T', '6724.T', '6752.T', '6758.T', '6762.T', '6770.T', '6841.T', 
    '6857.T', '6902.T', '6952.T', '6954.T', '6971.T', '6981.T', '6988.T', '7003.T', '7011.T', 
    '7012.T', '7201.T', '7202.T', '7203.T', '7205.T', '7211.T', '7261.T', '7267.T', '7269.T', 
    '7270.T', '7272.T', '7731.T', '7733.T', '7741.T', '7751.T', '7752.T', '7762.T', '7911.T', 
    '7912.T', '7951.T', '7974.T', '8001.T', '8002.T', '8015.T', '8031.T', '8035.T', '8058.T', 
    '8233.T', '8252.T', '8253.T', '8267.T', '8304.T', '8306.T', '8308.T', '8316.T', '8331.T', 
    '8354.T', '8411.T', '8630.T', '8697.T', '8766.T', '8795.T', '8801.T', '8802.T', '8804.T', 
    '8830.T', '9001.T', '9005.T', '9007.T', '9008.T', '9009.T', '9020.T', '9021.T', '9022.T', 
    '9064.T', '9101.T', '9104.T', '9107.T', '9202.T', '9301.T', '9412.T', '9432.T', '9433.T', 
    '9434.T', '9531.T', '9532.T', '9602.T', '9613.T', '9706.T', '9735.T', '9766.T', '9983.T', 
    '9984.T', '9989.T'
    ]
    start_date = '2019-01-01'
    end_date = '2023-12-31'

    data = download_data(tickers, start_date, end_date)
    data.index = data.index.tz_localize(None)
    print("数据下载完成")

    annual_returns, annual_volatility, cov_matrix = calculate_metrics(data)
    print("指标计算完成")

    best_sharpe_ratio = None

    target_choice = int(input("请选择目标收益率和波动率设定方式 (1 = optimize_portfolio, 2 = 自定义, 3 = Nikkei 225): "))

    if target_choice == 1:
        opt_result, best_sharpe_ratio, optimized_weights = optimize_portfolio(annual_returns, cov_matrix)
        target_return, target_volatility = portfolio_performance(optimized_weights, annual_returns, cov_matrix)
        target_method = "optimize_portfolio"
        print(f"目标收益率更新为: {target_return}")
        print(f"目标波动率更新为: {target_volatility}")
        print(f"Optimize Portfolio 得到的最佳 Sharpe Ratio 为: {best_sharpe_ratio:.4f}")

    elif target_choice == 2:
        target_return = float(input("请输入自定义的目标收益率（如 0.25）: "))
        target_volatility = float(input("请输入自定义的目标波动率（如 0.10）: "))
        target_method = "custom"
        print(f"自定义目标收益率: {target_return}, 波动率: {target_volatility}")

    elif target_choice == 3:
        target_return, target_volatility = calculate_nikkei_metrics(start_date, end_date)
        target_method = "nikkei225"
        print(f"Nikkei 225 年化收益率: {target_return}, 年化波动率: {target_volatility}")

    else:
        print("无效选择，请输入 1、2 或 3")
        return

    last_closing_prices, fractional_price, fractional_return, fractional_volatility, fractional_cov_matrix = fractional_formulation(
        data, annual_returns, annual_volatility, cov_matrix)
    print("分数计算完成")

    nmax, nmax_int, demanded_bits = Quantum_formulation(last_closing_prices)
    print("量子形式计算完成")

    matrix_c = binary_conversion_matrix(last_closing_prices)
    portfolio_bits_return, portfolio_bits_volatility = portfolio_bits_formulation(matrix_c, fractional_return, fractional_cov_matrix)
    print("投资组合二进制形式计算完成")

    # 构建 CQM 模型并记录构建时间
    start_build = time.time()
    cqm, b_vars = build_cqm(
        portfolio_bits_return, portfolio_bits_volatility, matrix_c,
        fractional_price, fractional_return, target_return, target_volatility,
        last_closing_prices
    )
    end_build = time.time()
    print(f"构建 CQM 模型耗时: {end_build - start_build:.4f} 秒")
    
        
    print("开始求解CQM")
    cqm_solutions, solution_times, total_solutions = solve_multiple_cqm_models(
        cqm, b_vars, gamma_values, lambda_values, theta_values
    )

    solver_name = "Leap_Hybrid_CQM"

    print(f"共求解 {total_solutions} 个组合")
    for gamma_value, lambda_value, theta_value, solve_time in solution_times:
        print(f"(gamma={gamma_value}, lambda={lambda_value}, theta={theta_value}) 求解时间: {solve_time:.4f}s")

    results_list, analysis_df = result(matrix_c, cqm_solutions, tickers, fractional_return,
                                       last_closing_prices, cov_matrix, annual_returns, budget)

    plot_file = plot_analysis(analysis_df, solver_name)
    print(f"分析图保存为: {plot_file}")

    plot_file, sdp_min, rp_min = plot_efficient_frontier_with_qubo(annual_returns, cov_matrix,
                                                                   cqm_solutions, matrix_c, last_closing_prices, solver_name)


    visualize_data(annual_returns, annual_volatility, cov_matrix)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = 'C:\\QPO'
    output_file = os.path.join(
        output_folder,
        f"QPO_{solver_name}_Target_{target_method}_Return_{target_return:.2f}_Volatility_{target_volatility:.2f}_TotalSolutions_{total_solutions}_{current_time}.xlsx")

    save_to_excel(data, annual_returns, annual_volatility, cov_matrix, nmax, nmax_int, demanded_bits,
                  last_closing_prices, fractional_price, fractional_return, fractional_volatility, fractional_cov_matrix, matrix_c,
                  portfolio_bits_return, portfolio_bits_volatility, results_list, output_file,
                  best_sharpe_ratio=best_sharpe_ratio, target_return=target_return, target_volatility=target_volatility,
                  min_volatility=sdp_min, min_vol_return=rp_min)

    print(f"数据保存到: {output_file}")

    for func_name, duration in function_times.items():
        print(f"{func_name} 耗时: {duration:.4f} 秒")


if __name__ == "__main__":
    main()
