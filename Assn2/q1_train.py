"""
================================================================================
Two-Stage Stochastic Airline Revenue Management Problem
================================================================================

Problem Overview:
  An airline must decide how many seats to allocate to Economy (E), Business (B),
  and First Class (F) before demand is revealed. After demand is known for a
  specific scenario, the airline sells seats up to allocated capacity.

Decision Variables:
  First-Stage (here-and-now): x_E, x_B, x_F = seat allocations
  Second-Stage (wait-and-see): y_E, y_B, y_F = actual seat sales per scenario

Constraints:
  Capacity: x_E + 1.5*x_B + 2*x_F <= 200 (seat capacity in terms of Economy units)
  No overselling: y_c <= x_c for each class c (cannot sell more than allocated)
  Demand satisfaction: y_c <= demand_c (cannot sell more than demanded)

Objective:
  Maximize expected profit = sum over scenarios of (probability * revenue for that scenario)
  Revenue per class: Economy = 1.0, Business = 2.0, First = 3.0 (in profit units per seat)

Three Models to Compare:
  1. Full Two-Stage: Optimal solution using all scenario information
  2. Mean-Value (MV): Uses average demands (ignores variability)
  3. Perfect Information (PI): Optimal if demand known in advance

Value Metrics:
  Value of Stochastic Solution (VSS) = Full Two-Stage - MV Evaluated
    Higher VSS indicates stochasticity is important
  
  Expected Value of Perfect Information (EVPI) = Perfect Information - Full Two-Stage
    Higher EVPI indicates demand uncertainty has significant impact

================================================================================
"""

import gurobipy as gb
from gurobipy import GRB

# ============================================================================
# SECTION 1: SCENARIO DATA
# ============================================================================
"""
Scenarios: Dictionary with probability and demand for each class per scenario

Each scenario represents a possible demand realization:
  Scenario 1: High demand (40 percent probability)
  Scenario 2: Medium-high demand (30 percent probability)
  Scenario 3: Medium-low demand (20 percent probability)
  Scenario 4: Low demand (10 percent probability)

Demands for Economy, Business, and First class vary by scenario.
"""

scenarios = {
    1: {"prob": 0.4, "demE": 200, "demB": 60,  "demF": 25},
    2: {"prob": 0.3, "demE": 180, "demB": 40,  "demF": 20},
    3: {"prob": 0.2, "demE": 175, "demB": 25,  "demF": 10},
    4: {"prob": 0.1, "demE": 150, "demB": 10,  "demF": 5}
}

# Profit per seat for Economy class (Business gets 2x, First gets 3x)
r_E = 1.0

# ============================================================================
# SECTION 2: FULL TWO-STAGE STOCHASTIC MODEL
# ============================================================================
"""
Full Two-Stage Model:
  - Solves complete problem considering all scenarios
  - First-stage: Allocate seats to each class
  - Second-stage: For each scenario, sell seats up to allocation and demand
  - Optimal solution accounts for uncertainty

Variables:
  x_E, x_B, x_F: First-stage seat allocations (same for all scenarios)
  y_E[xi], y_B[xi], y_F[xi]: Second-stage sales per class per scenario xi
"""

m_full = gb.Model("Full_2Stage")
m_full.setParam("OutputFlag", 0)

# First-stage variables: seat allocations
xE = m_full.addVar(lb=0, vtype=GRB.CONTINUOUS, name="xE")
xB = m_full.addVar(lb=0, vtype=GRB.CONTINUOUS, name="xB")
xF = m_full.addVar(lb=0, vtype=GRB.CONTINUOUS, name="xF")

# Second-stage variables: sales per scenario
yE = {}
yB = {}
yF = {}
for xi in scenarios:
    yE[xi] = m_full.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"yE_{xi}")
    yB[xi] = m_full.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"yB_{xi}")
    yF[xi] = m_full.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"yF_{xi}")

# Capacity constraint: total seats (in Economy-equivalent units)
m_full.addConstr(xE + 1.5*xB + 2*xF <= 200, "Capacity")

# Scenario-specific constraints
for xi, data in scenarios.items():
    # No overselling: cannot sell more than allocated
    m_full.addConstr(yE[xi] <= xE, name=f"NoOversellE_{xi}")
    m_full.addConstr(yB[xi] <= xB, name=f"NoOversellB_{xi}")
    m_full.addConstr(yF[xi] <= xF, name=f"NoOversellF_{xi}")
    
    # Demand constraints: cannot sell more than demanded
    m_full.addConstr(yE[xi] <= data["demE"], name=f"DemE_{xi}")
    m_full.addConstr(yB[xi] <= data["demB"], name=f"DemB_{xi}")
    m_full.addConstr(yF[xi] <= data["demF"], name=f"DemF_{xi}")

# Objective: Maximize expected revenue
obj_full = gb.LinExpr()
for xi, data in scenarios.items():
    p_xi = data["prob"]
    # Revenue is profit per seat times number of seats sold
    obj_full += p_xi * (r_E*yE[xi] + 2*r_E*yB[xi] + 3*r_E*yF[xi])
m_full.setObjective(obj_full, GRB.MAXIMIZE)
m_full.optimize()

# Display full model results
print("\n" + "=" * 70)
print("FULL TWO-STAGE STOCHASTIC MODEL RESULTS")
print("=" * 70)

if m_full.status == GRB.OPTIMAL:
    print(f"\nOptimal Expected Profit: {m_full.objVal:.4f}")
    print(f"\nOptimal Seat Allocations (First-Stage Decisions):")
    print(f"  Economy seats allocated:    {xE.X:.4f}")
    print(f"  Business seats allocated:   {xB.X:.4f}")
    print(f"  First seats allocated:      {xF.X:.4f}")
    
    print(f"\nSecond-Stage Sales by Scenario:")
    for xi in sorted(scenarios):
        print(f"\n  Scenario {xi} (probability equals {scenarios[xi]['prob']}):")
        print(f"    Economy seats sold:     {yE[xi].X:.4f}")
        print(f"    Business seats sold:    {yB[xi].X:.4f}")
        print(f"    First seats sold:       {yF[xi].X:.4f}")
else:
    print("Error: No optimal solution found for full model")

vS = m_full.objVal

# ============================================================================
# SECTION 3: MEAN-VALUE (DETERMINISTIC) MODEL
# ============================================================================
"""
Mean-Value Model:
  - Ignores uncertainty: uses average (expected) demand for each class
  - Treats as deterministic optimization problem
  - Representative of naive approach that ignores variability

Why Compare to Mean-Value?
  - Simpler to solve
  - May perform poorly when variance is high
  - Difference from full model indicates value of accounting for uncertainty (VSS)
"""

# Compute expected demands across all scenarios
dE_MV = sum(data["prob"] * data["demE"] for data in scenarios.values())
dB_MV = sum(data["prob"] * data["demB"] for data in scenarios.values())
dF_MV = sum(data["prob"] * data["demF"] for data in scenarios.values())

print("\n" + "=" * 70)
print("MEAN-VALUE MODEL (IGNORING UNCERTAINTY)")
print("=" * 70)
print(f"\nExpected Demands (weighted average):")
print(f"  Economy demand:             {dE_MV:.4f}")
print(f"  Business demand:            {dB_MV:.4f}")
print(f"  First demand:               {dF_MV:.4f}")

# Build and solve mean-value model
m_MV = gb.Model("MV_Model")
m_MV.setParam("OutputFlag", 0)

xE_MV = m_MV.addVar(lb=0, vtype=GRB.CONTINUOUS, name="xE_MV")
xB_MV = m_MV.addVar(lb=0, vtype=GRB.CONTINUOUS, name="xB_MV")
xF_MV = m_MV.addVar(lb=0, vtype=GRB.CONTINUOUS, name="xF_MV")
yE_MV = m_MV.addVar(lb=0, vtype=GRB.CONTINUOUS, name="yE_MV")
yB_MV = m_MV.addVar(lb=0, vtype=GRB.CONTINUOUS, name="yB_MV")
yF_MV = m_MV.addVar(lb=0, vtype=GRB.CONTINUOUS, name="yF_MV")

# Constraints
m_MV.addConstr(xE_MV + 1.5*xB_MV + 2*xF_MV <= 200, "Capacity_MV")
m_MV.addConstr(yE_MV <= xE_MV, "NoOversellE_MV")
m_MV.addConstr(yB_MV <= xB_MV, "NoOversellB_MV")
m_MV.addConstr(yF_MV <= xF_MV, "NoOversellF_MV")
m_MV.addConstr(yE_MV <= dE_MV, "DemE_MV")
m_MV.addConstr(yB_MV <= dB_MV, "DemB_MV")
m_MV.addConstr(yF_MV <= dF_MV, "DemF_MV")

# Objective using average demands
obj_MV = r_E*yE_MV + 2*r_E*yB_MV + 3*r_E*yF_MV
m_MV.setObjective(obj_MV, GRB.MAXIMIZE)
m_MV.optimize()

# Extract mean-value solution
xE_MV_val = xE_MV.X
xB_MV_val = xB_MV.X
xF_MV_val = xF_MV.X

print(f"\nMean-Value Optimal Seat Allocations:")
print(f"  Economy seats:              {xE_MV_val:.4f}")
print(f"  Business seats:             {xB_MV_val:.4f}")
print(f"  First seats:                {xF_MV_val:.4f}")

# Evaluate mean-value solution on actual scenarios
"""
This step uses the mean-value allocations and evaluates them against
all possible scenarios. This shows how the mean-value approach performs
when actual demand differs from expected demand.
"""
print(f"\nEvaluating Mean-Value Solution Across All Scenarios:")

vMV_eval = 0.0
for xi, data in scenarios.items():
    # With fixed allocation, actual sales equals minimum of allocation and demand
    yE_val = min(xE_MV_val, data["demE"])
    yB_val = min(xB_MV_val, data["demB"])
    yF_val = min(xF_MV_val, data["demF"])
    
    # Profit for this scenario
    profit_xi = r_E*yE_val + 2*r_E*yB_val + 3*r_E*yF_val
    vMV_eval += data["prob"] * profit_xi
    
    print(f"  Scenario {xi}: Revenue equals {profit_xi:.4f}")

print(f"\nMean-Value Expected Profit: {vMV_eval:.4f}")

# ============================================================================
# SECTION 4: PERFECT INFORMATION MODEL
# ============================================================================
"""
Perfect Information Model:
  - Assumes demand is known before seat allocation decisions
  - Optimal if we could predict demand perfectly
  - Provides upper bound on achievable profit

Why Include This?
  - Benchmark for best possible case
  - Difference from full model (EVPI) indicates cost of uncertainty
  - Shows how much better we could do with perfect forecasts
"""

print("\n" + "=" * 70)
print("PERFECT INFORMATION MODEL (DEMAND KNOWN IN ADVANCE)")
print("=" * 70)

vPI = 0.0
for xi, data in scenarios.items():
    # For each scenario, solve allocation problem knowing demand
    m_PI = gb.Model(f"PI_scenario_{xi}")
    m_PI.setParam("OutputFlag", 0)
    
    # Decision variables for this scenario
    yE_PI = m_PI.addVar(lb=0, vtype=GRB.CONTINUOUS, name="yE_PI")
    yB_PI = m_PI.addVar(lb=0, vtype=GRB.CONTINUOUS, name="yB_PI")
    yF_PI = m_PI.addVar(lb=0, vtype=GRB.CONTINUOUS, name="yF_PI")
    
    # Constraints
    m_PI.addConstr(yE_PI + 1.5*yB_PI + 2*yF_PI <= 200, "Capacity_PI")
    m_PI.addConstr(yE_PI <= data["demE"], "DemE_PI")
    m_PI.addConstr(yB_PI <= data["demB"], "DemB_PI")
    m_PI.addConstr(yF_PI <= data["demF"], "DemF_PI")
    
    # Objective
    m_PI.setObjective(r_E*yE_PI + 2*r_E*yB_PI + 3*r_E*yF_PI, GRB.MAXIMIZE)
    m_PI.optimize()
    
    if m_PI.status == GRB.OPTIMAL:
        profit_pi = m_PI.objVal
        vPI += data["prob"] * profit_pi
        print(f"  Scenario {xi}: Optimal revenue equals {profit_pi:.4f}")

print(f"\nPerfect Information Expected Profit: {vPI:.4f}")

# ============================================================================
# SECTION 5: VALUE METRICS (VSS AND EVPI)
# ============================================================================
"""
Value of Stochastic Solution (VSS):
  Measures benefit of using stochastic optimization vs deterministic approach
  Formula: VSS equals Full Two-Stage Profit minus Mean-Value Evaluated Profit
  Interpretation:
    - Positive VSS means accounting for uncertainty is beneficial
    - Larger VSS indicates higher value of stochastic approach
    - Small or zero VSS means deterministic approach is adequate

Expected Value of Perfect Information (EVPI):
  Measures cost of demand uncertainty
  Formula: EVPI equals Perfect Information Profit minus Full Two-Stage Profit
  Interpretation:
    - Positive EVPI means uncertainty costs money
    - Larger EVPI indicates higher value of improved forecasting
    - Indicates maximum benefit from reducing forecast error

Relationship:
  vPI greater than or equal to vS greater than or equal to vMV_eval
  So VSS greater than or equal to 0 and EVPI greater than or equal to 0
"""

VSS = vS - vMV_eval
EVPI = vPI - vS

print("\n" + "=" * 70)
print("VALUE METRICS SUMMARY")
print("=" * 70)
print(f"\nFull Two-Stage Expected Profit:        {vS:.4f}")
print(f"Mean-Value Evaluated Profit:           {vMV_eval:.4f}")
print(f"Perfect Information Expected Profit:   {vPI:.4f}")
print(f"\nValue of Stochastic Solution (VSS):   {VSS:.4f}")
print(f"  Interpretation: Using stochastic optimization instead of")
print(f"                  deterministic approach gains {VSS:.4f} in expected profit")
print(f"\nExpected Value of Perfect Information (EVPI): {EVPI:.4f}")
print(f"  Interpretation: Uncertainty costs {EVPI:.4f} in expected profit")
print(f"\nTotal Potential Gain from Both Benefits: {VSS + EVPI:.4f}")
print("=" * 70)