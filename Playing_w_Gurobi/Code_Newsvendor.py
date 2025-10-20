import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

##### Problem Setup #####
# Newsvendor Problem: Determine optimal order quantity to maximize expected profit
# Trade-off between ordering too much (excess inventory) or too little (lost sales)

# Demand parameters (assume normally distributed demand)
mu = 100          # Mean demand (expected units sold)
sigma = 30        # Standard deviation of demand (demand variability)

s = 2             # Selling price per unit (revenue per sold unit)
c = 0.3           # Unit cost (cost per unit ordered)
r = 0.05          # Salvage value per unit (recovery value for unsold inventory)

NoScenarios = 5000 # Number of Monte Carlo scenarios to simulate

##### Generate Demand Scenarios #####
# Generate random demand samples from normal distribution N(mu, sigma)
np.random.seed(111) # Set seed for reproducibility

D = np.random.normal(mu, sigma, NoScenarios)

##### Visualize Demand Distribution #####
# Plot histogram of simulated demand against theoretical normal distribution
_, bins, _ = plt.hist(D, 200, density=1, label='data', color = 'blue') # Histogram of simulated data
y = stats.norm.pdf(bins, mu, sigma)                     # Theoretical normal PDF
plt.plot(bins, y, label='best fit', color = 'red',  linewidth = 2)
plt.xlabel('Demand (units)')
plt.ylabel('Probability Density')
plt.title('Simulated Demand Distribution vs Normal Distribution')
plt.legend()
plt.show()

##### Define Profit Calculation Function #####
def ComputeAverageProfit(buy):
    """
    Compute average profit across all scenarios for a given order quantity
    
    Parameters:
    buy - Order quantity (units to purchase at beginning of period)
    
    Returns:
    average_profit - Expected profit = E[revenue - cost]
    
    Calculation:
    - sell = min(demand, order_quantity) = units actually sold
    - salvage = order_quantity - sell = units left over unsold
    - revenue = s*sell + r*salvage = selling price*sold + salvage_value*unsold
    - profit = revenue - c*buy = total_revenue - ordering_cost
    - average_profit = mean of profits across all scenarios
    """
    
    # Units actually sold: min(demand, ordered quantity)
    sell = np.minimum(D, buy)
    
    # Units not sold (leftover inventory)
    salvage = buy * np.ones(NoScenarios) - sell
    
    # Total revenue: selling revenue + salvage value
    revenue = s * sell + r * salvage
    
    # Profit: total revenue - ordering cost
    profit = revenue - c * buy * np.ones(NoScenarios)
    
    # Average profit across all scenarios
    average_profit = np.mean(profit)
    
    return average_profit

##### Solution 1: Mean Value Solution #####
# Naive approach: Order quantity = expected demand (ignores demand uncertainty)
buy_MV = mu
print("Mean Value Solution (deterministic):")
print("Order quantity = %s units" % buy_MV)
print("Average profit = %.2f" % ComputeAverageProfit(buy_MV))
print()

##### Solution 2: Stochastic Solution #####
# Optimal approach: Use critical fractile formula from newsvendor theory
# Optimal order quantity q* = F_inverse((s-c)/(s-r))
# where F is the CDF of demand distribution
# Critical fractile = (s-c)/(s-r) represents optimal service level

critical_fractile = (s - c) / (s - r)
buy_Stochastic = stats.norm.ppf(critical_fractile, loc = mu, scale = sigma)

print("Stochastic Solution (optimal):")
print("Critical fractile (service level) = %.4f" % critical_fractile)
print("Order quantity = %.2f units" % buy_Stochastic)
print("Average profit = %.2f" % ComputeAverageProfit(buy_Stochastic))
print()

##### Solution Comparison #####
# Compare performance of naive vs optimal approach
profit_MV = ComputeAverageProfit(buy_MV)
profit_Stochastic = ComputeAverageProfit(buy_Stochastic)
profit_improvement = ((profit_Stochastic - profit_MV) / profit_MV) * 100

print("Solution Comparison:")
print("Profit improvement from optimal solution: %.2f%%" % profit_improvement)
print("Additional expected profit per period: %.2f" % (profit_Stochastic - profit_MV))