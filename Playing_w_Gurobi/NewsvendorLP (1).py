import numpy as np
from gurobipy import Model, GRB, quicksum

##### Problem Setup #####
# Newsvendor Problem (Stochastic LP Formulation)
# Determine optimal purchase quantity to maximize expected profit under demand uncertainty
# Trade-off: ordering too much (excess salvage) vs too little (lost sales)

# Demand parameters (assume normally distributed demand)
mu = 100          # Mean demand (expected units sold)
sigma = 30        # Standard deviation of demand (demand variability)

s = 2             # Selling price per unit (revenue per sold unit)
c = 0.3           # Cost per unit purchased (unit purchase cost)
r = 0.05          # Salvage value per unit (recovery value for unsold inventory)
NoScenarios = 5000 # Number of Monte Carlo demand scenarios

##### Generate Demand Scenarios #####
# Generate random demand samples from truncated normal distribution
np.random.seed(111) # Set seed for reproducibility
D = np.random.normal(mu, sigma, NoScenarios) # Normal distribution samples
D = np.maximum(D, 0) # Truncate negative values to 0 (demand cannot be negative)
p = np.ones(NoScenarios) / NoScenarios # Equal probability for each scenario

print('Demand Statistics:')
print(f'  Mean demand: {np.mean(D):.2f}')
print(f'  Std dev demand: {np.std(D):.2f}')
print(f'  Min demand: {np.min(D):.2f}')
print(f'  Max demand: {np.max(D):.2f}')
print()

##### Build Gurobi Model #####
# Stochastic linear program with explicit scenario enumeration
model = Model("Newsvendor_Stochastic_LP")

##### Decision Variables #####
# First-stage variable: purchase decision (made before demand is realized)
x = model.addVar(lb=0, name="x")  # Order quantity: units to purchase upfront

# Second-stage variables: recourse decisions (made after demand is revealed)
y = model.addVars(NoScenarios, lb=0, name="y")  # Units sold in scenario k
z = model.addVars(NoScenarios, lb=0, name="z")  # Units salvaged (unsold) in scenario k

##### Objective Function #####
# Maximize expected profit = expected revenue - purchase cost
# Profit = -c*x (purchase cost, fixed across all scenarios)
#        + E[s*y_k + r*z_k] (expected revenue from sales and salvage)
model.setObjective(
    -c * x + quicksum(p[k] * (s * y[k] + r * z[k]) for k in range(NoScenarios)), 
    GRB.MAXIMIZE
)

##### Constraints #####

# Demand constraints: cannot sell more than demand
# y_k <= d_k for all scenarios k
# Meaning: units sold in scenario k cannot exceed demand d_k
model.addConstrs(
    (y[k] <= D[k] for k in range(NoScenarios)), 
    "Demand"
)

# Balance constraints: ordered units = units sold + units salvaged
# x = y_k + z_k for all scenarios k
# Meaning: in each scenario, purchased units are split between sales and salvage
model.addConstrs(
    (x == y[k] + z[k] for k in range(NoScenarios)), 
    "Balance"
)

##### Solve the Stochastic LP #####
# Solves the newsvendor problem considering all demand scenarios
model.optimize()

##### Display Results #####
# Print optimal solution and performance metrics
print('OPTIMAL STOCHASTIC SOLUTION:')
print('='*50)

if model.status == GRB.OPTIMAL:
    optimal_purchase = x.x
    optimal_profit = model.objVal
    
    print(f'Optimal order quantity (x): {optimal_purchase:.2f} units')
    print(f'Optimal expected profit: {optimal_profit:.2f}')
    print()
    
    # Compute scenario-specific metrics
    print('Scenario Performance:')
    total_sales = sum(y[k].x for k in range(NoScenarios))
    total_salvage = sum(z[k].x for k in range(NoScenarios))
    avg_sales = total_sales / NoScenarios
    avg_salvage = total_salvage / NoScenarios
    stockout_count = sum(1 for k in range(NoScenarios) if y[k].x < D[k] - 1e-6)
    stockout_prob = stockout_count / NoScenarios
    
    print(f'  Average units sold per scenario: {avg_sales:.2f}')
    print(f'  Average units salvaged per scenario: {avg_salvage:.2f}')
    print(f'  Stockout probability (y_k < d_k): {stockout_prob:.4f}')
    print()
    
    # Compare to theoretical newsvendor solution
    critical_fractile = (s - c) / (s - r)
    print(f'Theoretical Analysis:')
    print(f'  Critical fractile (service level): {critical_fractile:.4f}')
    print(f'  (% of time demand is met without stockout)')
    
else:
    print("No optimal solution found.")
    print(f"Model status: {model.status}")

##### Note on Demand Truncation #####
"""
IMPORTANT: Truncated Normal Distribution

The code uses D = np.maximum(D, 0) to handle negative demand samples.

Why this is necessary:
- Normal distribution can generate negative values by chance
- Demand cannot be negative in reality
- Without truncation, constraint y_k <= D[k] may be infeasible with y_k >= 0

Alternative approaches:
1. Truncated normal distribution: scipy.stats.truncnorm
2. Lognormal distribution: naturally non-negative
3. Exponential distribution: suitable for non-negative demands

Example using scipy truncated normal:
  from scipy.stats import truncnorm
  a = (0 - mu) / sigma  # standardized lower bound
  D = truncnorm.rvs(a, np.inf, loc=mu, scale=sigma, size=NoScenarios)
"""