"""
================================================================================
Two-Stage Stochastic Inventory Distribution Problem
================================================================================

Problem Description:
  A central depot must decide first-stage shipments to cities before demand is known.
  After demand is revealed, second-stage recourse actions (additional shipments) are taken.
  Goal: Minimize total expected cost (first-stage plus expected second-stage cost).

Decision Variables:
  First-Stage: x[n] = initial shipment quantity to city n (before demand known)
  Second-Stage: y[n,k] = additional shipment to city n under scenario k (after demand known)
  Slack Variables: leftover[n,k] = excess inventory, shortage[n,k] = unmet demand

Objective:
  Minimize: first-stage shipping cost + expected recourse cost
  Where recourse cost includes second-stage shipping, leftover penalties, and shortage penalties

Key Parameters:
  theta[n] = first-stage cost per unit to city n
  theta_s[n] = second-stage cost per unit to city n
  h = penalty cost per unit of leftover inventory
  g = penalty cost per unit of shortage
  I = total central depot inventory
  Yn[n] = initial inventory already at city n
  demand[(n,k)] = demand at city n under scenario k

================================================================================
"""

import gurobipy as gp
from gurobipy import GRB

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================
"""
Load problem data from external modules.
ReadData.py provides all parameters and scenario information.
Assumes equal probability for each scenario.
"""

import ReadData

cities = ReadData.cities           # list of city identifiers
scenarios = ReadData.scenarios     # list of scenario identifiers
theta = ReadData.theta             # first-stage shipping cost per unit by city
theta_s = ReadData.theta_s         # second-stage shipping cost per unit by city
h = ReadData.h                     # leftover inventory penalty cost per unit
g = ReadData.g                     # shortage penalty cost per unit
I = ReadData.I                     # total central depot inventory available
Yn = ReadData.Yn                   # initial inventory at each city (dictionary)
demand = ReadData.demand           # demand dictionary indexed by (city, scenario)
prob = 1.0 / len(scenarios)        # uniform scenario probability

# ============================================================================
# SECTION 2: MODEL SETUP
# ============================================================================
"""
Create and configure the optimization model.
OutputFlag equals 0 suppresses solver console output.
"""

model = gp.Model("TwoStageStochasticInventory")
model.Params.OutputFlag = 0

# ============================================================================
# SECTION 3: DECISION VARIABLES
# ============================================================================
"""
First-Stage Variables:
  x[n] = quantity shipped to city n before demand realization
  Non-negative, continuous
  
Second-Stage Variables (per scenario k):
  y[n,k] = additional quantity shipped to city n in scenario k
           (recourse action after demand is known)
  leftover[n,k] = excess inventory at city n in scenario k (slack for oversupply)
  shortage[n,k] = unmet demand at city n in scenario k (slack for undersupply)
"""

x = model.addVars(cities, name="x", lb=0)
y = model.addVars(cities, scenarios, name="y", lb=0)
leftover = model.addVars(cities, scenarios, name="leftover", lb=0)
shortage = model.addVars(cities, scenarios, name="shortage", lb=0)

# ============================================================================
# SECTION 4: CONSTRAINTS
# ============================================================================
"""
Constraint 1: First-Stage Inventory Limit
  Total first-stage shipments cannot exceed available central depot inventory.
"""
model.addConstr(
    gp.quicksum(x[n] for n in cities) <= I,
    name="FirstStageInvLimit"
)

"""
Constraint 2: Second-Stage Inventory Limit (per scenario)
  For each scenario, total second-stage shipments cannot exceed
  remaining inventory after first-stage allocation.
"""
total_first_stage = gp.quicksum(x[n] for n in cities)
for k in scenarios:
    model.addConstr(
        gp.quicksum(y[n, k] for n in cities) <= I - total_first_stage,
        name=f"CenterInvLimit_{k}"
    )

"""
Constraint 3: Leftover Definition (per city and scenario)
  Leftover equals amount of inventory exceeding demand.
  If available inventory is more than demand, leftover is positive.
  Otherwise leftover equals 0 (non-negativity).
  
  Available inventory at city n in scenario k equals:
    Yn[n] (initial) + x[n] (first-stage) + y[n,k] (second-stage)
"""
for k in scenarios:
    for n in cities:
        model.addConstr(
            leftover[n, k] >= Yn[n] + x[n] + y[n, k] - demand[(n, k)],
            name=f"LeftoverDef_{n}_{k}"
        )

"""
Constraint 4: Shortage Definition (per city and scenario)
  Shortage equals unmet demand.
  If available inventory is less than demand, shortage is positive.
  Otherwise shortage equals 0 (non-negativity).
"""
for k in scenarios:
    for n in cities:
        model.addConstr(
            shortage[n, k] >= demand[(n, k)] - (Yn[n] + x[n] + y[n, k]),
            name=f"ShortageDef_{n}_{k}"
        )

# ============================================================================
# SECTION 5: OBJECTIVE FUNCTION
# ============================================================================
"""
Total Cost equals First-Stage Cost plus Expected Second-Stage Cost.

First-Stage Cost:
  sum over n of theta[n] times x[n]
  (Cost of initial shipments to all cities)

Expected Second-Stage Cost:
  sum over scenarios k and cities n of:
    prob times (theta_s[n] times y[n,k]     (recourse shipment cost)
              plus h times leftover[n,k]     (excess inventory penalty)
              plus g times shortage[n,k])    (shortage penalty)
  
  prob equals 1/number_of_scenarios (uniform probability)
"""

first_stage_cost = gp.quicksum(theta[n] * x[n] for n in cities)

second_stage_cost = gp.quicksum(
    prob * (theta_s[n] * y[n, k] + h * leftover[n, k] + g * shortage[n, k])
    for n in cities
    for k in scenarios
)

model.setObjective(first_stage_cost + second_stage_cost, GRB.MINIMIZE)

# ============================================================================
# SECTION 6: SOLVE AND OUTPUT
# ============================================================================
"""
Optimize the model and display results.
"""

model.optimize()

# Check optimization status
if model.Status == GRB.OPTIMAL:
    opt_obj = model.ObjVal
    print(f"\nOptimal objective value: {opt_obj:.2f}")
    print("\nOptimal first-stage decisions (initial shipments to cities):")
    for n in cities:
        print(f"  x[{n}] equals {x[n].X:.2f}")
    print(f"\nSolution time: {model.Runtime:.4f} seconds")
else:
    print(f"Optimization failed. Status code: {model.Status}")