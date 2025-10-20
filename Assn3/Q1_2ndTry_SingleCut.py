"""
================================================================================
Assignment 3 Q1 - Single-Cut Benders Decomposition
================================================================================

ALGORITHM OVERVIEW:
-------------------
This script implements the single-cut version of Benders decomposition to solve
a two-stage stochastic inventory distribution problem. Unlike the multi-cut approach,
this algorithm aggregates dual information from ALL scenarios and adds ONE consolidated
Benders cut per iteration.

BENDERS DECOMPOSITION CONCEPT:
------------------------------
Instead of solving the full problem at once, Benders decomposition iteratively:
  - Solves the master problem to get a lower bound (LB) and candidate solution x*
  - Solves each scenario's subproblem to evaluate recourse costs
  - Aggregates dual information from all scenarios into a single cut
  - Adds the aggregated cut back to the master to refine approximation
  - Repeats until convergence (no new cut added)

SINGLE-CUT vs MULTI-CUT COMPARISON:
------------------------------------
Single-Cut Approach (This Script):
  - Variables: ONE epigraph variable theta_var for total recourse cost
  - Cuts per iteration: ONE cut aggregating all scenarios
  - Master size: Smaller (fewer constraints)
  - Convergence: Potentially slower (less information per iteration)
  - Advantage: Computationally efficient, suitable for problems with many scenarios
  
Multi-Cut Approach (Alternative):
  - Variables: ONE epigraph variable n_vars[k] per scenario k
  - Cuts per iteration: UP TO |scenarios| cuts (one per scenario)
  - Master size: Larger (more constraints)
  - Convergence: Potentially faster (more detailed information)
  - Advantage: Better bounds earlier, useful when scenarios << variables

AGGREGATION MECHANISM:
----------------------
In single-cut Benders:
  1. Solve subproblem for each scenario k, get duals (gamma_k, pi_k)
  2. Weight duals by scenario probability: prob(k)
  3. Aggregate: aggregated_coeff[c] = sum_k [prob(k) * (gamma_k + pi_k[c])]
  4. Form ONE cut using aggregated coefficients
  5. Evaluate: if theta_var < aggregated_rhs, add cut; else stop

KNOWN ISSUES (documented in original code):
--------------------------------------------
1. Primal solution printing may display incorrect values due to RHS computation
2. Issue stems from how duals are used in cut formulation
3. Fix would involve reformulating the cut RHS calculation
4. Not implemented due to time constraints - algorithm still functions correctly
5. Recommendation: Pre-solve master without scenario cuts for better initial bounds

================================================================================
"""

import gurobipy as gp
from gurobipy import GRB, LinExpr, quicksum
import time
import ReadData as Data

# ============================================================================
# SECTION 1: DATA LOADING AND PARAMETERS
# ============================================================================
"""
Load problem data from external module and initialize algorithm parameters.

Two-Stage Stochastic Program Structure:
----------------------------------------
Stage 1 (First-Stage/Here-and-Now):
  - Decision: x[c] = units to allocate to city c from central depot
  - Cost: theta[c] per unit (first-stage distribution cost)
  - Constraint: sum of x[c] <= I (total budget constraint)
  - Timing: Made BEFORE uncertainty is revealed

Stage 2 (Second-Stage/Wait-and-See):
  - Uncertainty: Demand is revealed as demand[(c,k)] for city c in scenario k
  - Decisions: Recourse variables (u, v, z, s) that adjust after demand known
    * u[c] = additional supply obtained from center (adjustment variable)
    * v[c] = additional demand served to city c (adjustment variable)
    * z[c] = unused inventory at city c (slack/waste)
    * s[c] = shortage/unmet demand at city c (penalty)
  - Cost: theta_s[c] per unit adjustment + h*z[c] + g*s[c]
  - Timing: Made AFTER demand uncertainty is revealed

Key Difference from Multi-Cut:
  - Single-cut aggregates recourse costs across all scenarios into one variable
  - theta_var approximates E_k[Q_k(x)] instead of individual Q_k(x) values
"""

# Load all required data structures from ReadData module
cities    = Data.cities          # List of city identifiers (e.g., ['C0', 'C1', ..., 'C9'])
scenarios = Data.scenarios       # List of scenario identifiers (e.g., ['S0', 'S1', ...])
theta     = Data.theta           # Dict: first-stage cost coefficient for each city
theta_s   = Data.theta_s         # Dict: second-stage adjustment cost for each city
h         = Data.h               # Scalar: penalty cost per unit of unused inventory (z)
g         = Data.g               # Scalar: penalty cost per unit of shortage (s)
I         = Data.I               # Scalar: total available inventory at central depot
Yn        = Data.Yn              # Dict: initial inventory already at each city
demand    = Data.demand          # Dict: demand[(city_c, scenario_k)] for each city-scenario pair
prob      = Data.prob            # Scalar: probability per scenario (assumes equal probability 1/|S|)

# Algorithm tolerances and limits
CutViolationTolerance = 1e-4  # Tolerance for determining if aggregated cut is violated
                              # If theta_var < aggregated_rhs - eps, cut is considered violated
max_iters = 200               # Maximum allowed Benders iterations (safety limit)

# ============================================================================
# SECTION 2: MASTER PROBLEM FORMULATION (SINGLE-CUT VERSION)
# ============================================================================
"""
Master Problem (First-Stage Problem) - Single-Cut Formulation:
---------------------------------------------------------------
The master problem represents first-stage decisions and an aggregate approximation
of the recourse cost. Unlike multi-cut, there is only ONE epigraph variable for
the total expected recourse cost, not one per scenario.

Variables:
  - x[c]: Units to allocate to city c (first-stage decision, per city)
  - theta_var: Single lower bound approximation on TOTAL expected recourse cost
               Represents: E_k[Q_k(x)] = sum_k [prob(k) * Q_k(x)]

Objective:
  Minimize: sum_c [theta[c] * x[c]]           (first-stage cost)
          + theta_var                          (expected second-stage cost approximation)

Constraints:
  1. Sum of allocations <= I                  (budget constraint)
  2. Aggregated Benders cuts (added dynamically):
     theta_var - sum_c[(aggregated_coeff[c] * x[c])] >= aggregated_rhs
     (one cut per iteration, aggregated over all scenarios)

Interpretation:
  - Master seeks first-stage decisions x that minimize total cost
  - Approximate total expected recourse using single theta_var
  - Single cut per iteration provides lower bound on E_k[Q_k(x)]
  - Cut coefficients are probability-weighted averages of scenario duals

Computational Implications:
  - Fewer constraints than multi-cut (advantage for large |scenarios|)
  - Less detailed information per iteration (disadvantage for convergence speed)
  - More suitable for many-scenario problems
"""

# Create master problem model
MP = gp.Model("Master_Singlecut")
MP.Params.OutputFlag = 0  # Suppress Gurobi solver console output for clarity

# Set optimization sense (minimize)
MP.modelSense = GRB.MINIMIZE

# Decision variable x[c]: first-stage allocation to each city
# Each x[c] has lower bound 0 and is included in objective with cost theta[c]
x = {c: MP.addVar(lb=0, obj=theta[c], name=f"x_{c}") for c in cities}

# Constraint: total first-stage allocation cannot exceed central depot budget
# Interpretation: sum of units sent out <= total available inventory I
MP.addConstr(quicksum(x[c] for c in cities) <= I, name="CenterInventory")

# Epigraph variable theta_var: single lower bound on total expected recourse cost
# This variable approximates E_k[Q_k(x)] = sum_k [prob(k) * Q_k(x)]
# Initial lower bound is 0 (will be tightened by aggregated Benders cuts)
# Coefficient 1.0 in objective (no probability weighting - already in cut derivation)
theta_var = MP.addVar(lb=0, obj=1.0, name="theta")

# Update model to reflect all variables and constraints
MP.update()

# ============================================================================
# SECTION 3: SUBPROBLEM TEMPLATE DEFINITION
# ============================================================================
"""
Subproblem (Second-Stage Problem for Each Scenario):
-----------------------------------------------------
For each scenario k, a subproblem solves the recourse (second-stage) problem
given a fixed first-stage decision x. The subproblem is solved multiple times
as x changes throughout the Benders iterations.

Variables (created per subproblem):
  - u[c]: Additional supply obtained from center to city c
  - v[c]: Additional demand served (adjustment to supply sent to city c)
  - z[c]: Unused/excess inventory at city c (waste)
  - s[c]: Shortage (unmet demand) at city c

Objective (for scenario k with fixed first-stage x):
  Minimize: sum_c [theta_s[c] * (u[c] + v[c])]   (adjustment costs)
          + h * sum_c [z[c]]                       (excess inventory penalty)
          + g * sum_c [s[c]]                       (shortage penalty)

Constraints (added per iteration with updated x):
  1. Capacity constraint: I + sum_c(u[c]) >= sum_c(v[c]) + sum_c(xsol[c])
     Interpretation: Available capacity >= demand for recourse + first-stage allocation
  
  2. For each city c, demand balance:
     Yn[c] + xsol[c] + v[c] + s[c] = demand[(c,k)] + z[c] + u[c]
     
     Left side (what's available):  initial_inventory + first_stage + recourse_supply + shortage
     Right side (what's needed):    demand + unused_inventory + adjustment
     
     Interpretation: Material balance - inflow must equal outflow + adjustments

Duals (Used for Single-Cut Aggregation):
  - gamma_k: Shadow price of capacity constraint for scenario k
  - pi_k[c]: Shadow price of city c demand constraint for scenario k
  
  These are aggregated (probability-weighted) across scenarios to form the single cut.

Single-Cut Aggregation Formula:
  aggregated_coeff[c] = sum_k [prob(k) * (gamma_k + pi_k[c])]
  aggregated_rhs = sum_k [prob(k) * (Q_k(x*) - sum_c[(gamma_k + pi_k[c]) * x*[c]])]
  
  Single cut: theta_var >= aggregated_rhs + sum_c[aggregated_coeff[c] * x[c]]
"""

def build_subproblem(model_name):
    """
    Factory function to create an empty subproblem model for a scenario.
    
    The subproblem is created with decision variables but WITHOUT constraints
    (constraints are added in solve_subproblem() before each solve, as they
    depend on the current first-stage solution xsol).
    
    Arguments:
      model_name (str): Name for the Gurobi model (e.g., "SP_S0")
    
    Returns:
      sp (gp.Model): Empty subproblem model with variables but no constraints
      u_vars (dict): Dictionary of u[c] variables (additional supply)
      v_vars (dict): Dictionary of v[c] variables (additional demand)
      z_vars (dict): Dictionary of z[c] variables (unused inventory)
      s_vars (dict): Dictionary of s[c] variables (shortage)
    
    Variable Definitions:
      u[c]: Non-negative adjustment variable representing additional supply from center
      v[c]: Non-negative adjustment variable representing additional units to serve
      z[c]: Non-negative slack variable for unused/excess inventory at city c
      s[c]: Non-negative slack variable for shortage (unmet demand) at city c
    
    Objective:
      Minimizes the recourse cost = adjustment costs + inventory penalties + shortage penalties
      This objective is the same across all scenarios; constraints change with demand.
    """
    sp = gp.Model(model_name)
    sp.Params.OutputFlag = 0  # Suppress output
    
    # Create four groups of decision variables (one per city)
    u_vars = {c: sp.addVar(lb=0, name=f"u_{c}") for c in cities}
    v_vars = {c: sp.addVar(lb=0, name=f"v_{c}") for c in cities}
    z_vars = {c: sp.addVar(lb=0, name=f"z_{c}") for c in cities}
    s_vars = {c: sp.addVar(lb=0, name=f"s_{c}") for c in cities}
    
    # Set objective function (without constraints)
    # Cost = adjustment cost + excess penalty + shortage penalty
    sp.setObjective(
        quicksum(theta_s[c]*(u_vars[c] + v_vars[c]) for c in cities) +  # Adjustment costs
        h * quicksum(z_vars[c] for c in cities) +                        # Excess inventory penalty
        g * quicksum(s_vars[c] for c in cities),                         # Shortage penalty
        GRB.MINIMIZE
    )
    sp.update()
    
    return sp, u_vars, v_vars, z_vars, s_vars

# Create one subproblem model for each scenario (to be reused across iterations)
SP = {}          # Dictionary to store subproblem models indexed by scenario key
sub_vars = {}    # Dictionary to store variable dictionaries indexed by scenario key

for k in scenarios:
    sp, u_vars, v_vars, z_vars, s_vars = build_subproblem(f"SP_{k}")
    SP[k] = sp
    sub_vars[k] = (u_vars, v_vars, z_vars, s_vars)

# ============================================================================
# SECTION 4: SUBPROBLEM SOLUTION FUNCTION
# ============================================================================
"""
Function: solve_subproblem(k, xsol)
-----------------------------------
Solves the recourse subproblem for scenario k given the current first-stage solution xsol.

This function is called in each Benders iteration for each scenario.
It:
  1. Removes old constraints from the subproblem (from previous iteration)
  2. Adds new constraints with the updated xsol values
  3. Solves the subproblem
  4. Extracts the optimal objective (recourse cost Q_k(xsol))
  5. Extracts dual values (shadow prices) for aggregating into the single cut

Constraints Added:
  (A) Capacity: I + sum(u[c]) >= sum(v[c]) + sum(xsol[c])
      Meaning: Total available capacity (initial + additional supply) must cover
               total recourse supply + first-stage allocation sent to cities
  
  (B) Demand Balance (for each city c):
      Yn[c] + xsol[c] + v[c] + s[c] = demand[(c,k)] + z[c] + u[c]
      
      Meaning: Material balance at city c:
      - Available: initial inventory Yn[c] + first-stage xsol[c] + recourse v[c] + shortage s[c]
      - Needed:    demand + unused inventory z[c] + supply adjustment u[c]

Dual Variables Extracted (for Single-Cut Aggregation):
  - gamma_k: Shadow price of capacity constraint
            Interpretation: Cost per unit of additional capacity in scenario k
            Sign: Non-negative (by LP duality)
  
  - pi_k[c]: Shadow price of city c demand constraint in scenario k
            Interpretation: Cost per unit of inventory at city c in scenario k
            Sign: Can be positive or negative depending on problem structure

Single-Cut Aggregation:
  These duals are weighted by prob(k) and summed across scenarios:
    aggregated_coeff[c] = sum_k [prob(k) * (gamma_k + pi_k[c])]
    aggregated_rhs += prob(k) * (Q_k(x*) - sum_c[(gamma_k + pi_k[c]) * x*[c]])
  
  Result: ONE consolidated cut representing expected recourse cost across all scenarios
"""

def solve_subproblem(k, xsol):
    """
    Solve the recourse subproblem for scenario k with current first-stage solution xsol.
    
    Arguments:
      k (str): Scenario identifier (e.g., "S0")
      xsol (dict): Current first-stage solution {city_c: value_x_c}
    
    Returns:
      SPobj (float): Optimal objective value (recourse cost Q_k(xsol))
      gamma (float): Dual multiplier for capacity constraint (shadow price of budget in scenario k)
      pi (dict): Dictionary of dual multipliers {city_c: dual_value} (shadow prices for each city)
      
    Returns (None, None, None) if subproblem is infeasible.
    
    Process:
      1. Remove all constraints from previous iteration
      2. Add capacity constraint (depends on xsol)
      3. Add demand balance constraints for each city (depends on xsol and demand[(c,k)])
      4. Optimize the subproblem
      5. Extract optimal value and dual solutions
      6. Return results for aggregation into single cut
    """
    sp = SP[k]
    u_vars, v_vars, z_vars, s_vars = sub_vars[k]
    
    # STEP 1: Remove all previous constraints to make room for new ones
    # (Constraints change each iteration because xsol changes and scenario is fixed)
    sp.remove(sp.getConstrs())
    sp.update()
    
    # STEP 2: Add capacity constraint (Constraint A)
    # Interpretation: Total available capacity must cover total demand and first-stage allocation
    # I + sum_c(u[c]) >= sum_c(v[c]) + sum_c(xsol[c])
    # 
    # Rearranged: I + sum_c(u[c]) - sum_c(v[c]) - sum_c(xsol[c]) >= 0
    cap_constr = sp.addConstr(
        I + quicksum(u_vars[c] for c in cities) >= 
        quicksum(v_vars[c] for c in cities) + sum(xsol[c] for c in cities),
        name="Capacity"
    )
    
    # STEP 3: Add demand balance constraints for each city (Constraint B)
    # For each city c:
    # Yn[c] + xsol[c] + v[c] + s[c] = demand[(c,k)] + z[c] + u[c]
    # 
    # Interpretation: Material balance at city c in scenario k
    # LHS (What's available): initial inventory + first-stage + recourse + shortage
    # RHS (What's needed): demand + excess inventory + adjustment
    demand_constr = {}
    for c in cities:
        demand_constr[c] = sp.addConstr(
            Yn[c] + xsol[c] + v_vars[c] + s_vars[c] == 
            demand[(c, k)] + z_vars[c] + u_vars[c],
            name=f"Demand_{c}"
        )
    sp.update()
    
    # STEP 4: Solve the subproblem
    sp.optimize()
    
    # Check if subproblem is feasible
    if sp.status != GRB.OPTIMAL:
        print(f"[ERROR] Subproblem for scenario {k} is infeasible!")
        return None, None, None
    
    # STEP 5: Extract optimal value and dual solutions
    SPobj = sp.objVal  # Optimal recourse cost Q_k(xsol)
    
    # Extract dual multipliers (shadow prices) from constraints
    # These represent the marginal cost of relaxing each constraint in this scenario
    gamma = cap_constr.Pi      # Shadow price of capacity constraint (cost of budget in scenario k)
    pi = {c: demand_constr[c].Pi for c in cities}  # Shadow price of each city's balance in scenario k
    
    # STEP 6: Return results
    return SPobj, gamma, pi


# ============================================================================
# SECTION 5: MAIN BENDERS DECOMPOSITION LOOP (SINGLE-CUT VERSION)
# ============================================================================
"""
Main Algorithm Loop - Single-Cut Benders:
-------------------------------------------
This loop executes the single-cut Benders decomposition algorithm.

Key Difference from Multi-Cut:
  In multi-cut: Add (up to |scenarios|) cuts per iteration, one per scenario
  In single-cut: Add 1 aggregated cut per iteration, using probability-weighted duals

Iteration Structure:
  Repeat until convergence or max_iters reached:
    1. SOLVE MASTER PROBLEM
       - Get lower bound (LB) and candidate solution x*
    
    2. EVALUATE ALL SUBPROBLEMS
       - For each scenario k, solve Q_k(x*)
       - Accumulate expected recourse cost for upper bound
       - Accumulate probability-weighted duals
    
    3. GENERATE SINGLE AGGREGATED BENDERS CUT
       - Use weighted-average duals from all scenarios
       - Form ONE constraint using aggregated coefficients
       - Add cut to master for next iteration
    
    4. CHECK CONVERGENCE
       - If aggregated cut not violated: solution is optimal
       - If cut violated: repeat with refined master problem

Key Variables:
  - LB (Lower Bound): Value of master problem objective
  - UB (Upper Bound): First-stage cost + actual expected recourse costs
  - theta_val: Current value of the single epigraph variable
  - aggregated_coeff: Probability-weighted sum of (gamma + pi[c]) across scenarios
  - aggregated_rhs: Probability-weighted sum of cut RHS values across scenarios

Aggregation Process:
  For each scenario k solved:
    aggregated_coeff[c] += prob(k) * (gamma_k + pi_k[c])
    aggregated_rhs += prob(k) * (Q_k(x*) - sum_c[(gamma_k + pi_k[c]) * x*[c]])
  
  Result: Single cut with aggregated information from all scenarios

Convergence Condition:
  When theta_val >= aggregated_rhs (cut not violated), it means:
  - Single variable theta_var is tight (equal to its cut value)
  - Expected recourse cost is accurately represented in master
  - Master solution is optimal for the stochastic program
"""

# Initialize tracking variables
TotalCutsAdded = 0      # Total aggregated Benders cuts added across all iterations
iteration = 0           # Iteration counter
BestUB = float('inf')   # Best (minimum) upper bound found so far
start_time = time.time()  # Start timer for total runtime

# Main Benders loop
while iteration < max_iters:
    iteration += 1
    MP.update()
    MP.optimize()
    
    # Check if master problem solved successfully
    if MP.status != GRB.OPTIMAL:
        print("[ERROR] Master problem is infeasible!")
        break
    
    # Extract current solution values from master problem
    LB = MP.objVal              # Lower bound (master objective value)
    xsol = {c: x[c].X for c in cities}  # Current first-stage decisions
    theta_val = theta_var.X     # Current single recourse approximation
    
    # Initialize upper bound calculation with first-stage cost
    # UB = first-stage cost + expected recourse cost
    UB = sum(theta[c]*xsol[c] for c in cities)
    
    # ========== ITERATION HEADER OUTPUT ==========
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration}")
    print(f"{'='*70}")
    print(f"Lower Bound (LB) from Master: {LB:.2f}")
    print(f"Current Recourse Approximation (theta_var): {theta_val:.2f}")
    print(f"\nCurrent First-Stage Solution x*:")
    for c in cities:
        print(f"  x[{c}] = {xsol[c]:.4f}")
    
    # ========== INITIALIZE AGGREGATION VARIABLES ==========
    # These will accumulate dual information across all scenarios
    # for the single aggregated cut
    
    aggregated_rhs = 0.0              # Accumulate probability-weighted RHS
    aggregated_coeff = {c: 0.0 for c in cities}  # Accumulate probability-weighted dual coefficients
    
    # ========== PROCESS EACH SCENARIO AND AGGREGATE DUALS ==========
    print(f"\n{'─'*70}")
    print(f"Evaluating Recourse Subproblems and Aggregating Duals:")
    print(f"{'─'*70}")
    
    for k in scenarios:
        # Solve recourse subproblem for this scenario at current x*
        SPobj, gamma, pi = solve_subproblem(k, xsol)
        
        # Check for infeasibility
        if SPobj is None:
            continue
        
        # Add weighted recourse cost to upper bound
        # UB = first-stage + sum_k [prob(k) * Q_k(x*)]
        UB += prob * SPobj
        
        # Print scenario evaluation results
        print(f"\nScenario {k}:")
        print(f"  Recourse Cost Q_{k}(x*) = {SPobj:.2f}")
        print(f"  Dual for Capacity in scenario {k}: γ_{k} = {gamma:.4f}")
        for c in cities:
            print(f"    π_{k}[{c}] = {pi[c]:.4f}", end="  ")
            if (cities.index(c) + 1) % 2 == 0:
                print()
        
        # ========== AGGREGATION: ACCUMULATE DUALS WEIGHTED BY PROBABILITY ==========
        # Single-cut aggregates dual information from all scenarios
        # Weight each scenario's contribution by its probability
        
        # Accumulate probability-weighted RHS contribution from this scenario
        # RHS_k = Q_k(x*) - sum_c[(gamma_k + pi_k[c]) * x*[c]]
        # aggregated_rhs += prob(k) * RHS_k
        aggregated_rhs += prob * (SPobj - sum((pi[c] + gamma)*xsol[c] for c in cities))
        
        # Accumulate probability-weighted dual coefficients from this scenario
        # coeff_k[c] = gamma_k + pi_k[c]
        # aggregated_coeff[c] += prob(k) * coeff_k[c]
        for c in cities:
            aggregated_coeff[c] += prob * (pi[c] + gamma)
    
    # ========== FORM AND ADD SINGLE AGGREGATED BENDERS CUT ==========
    """
    The aggregated cut takes the form:
      theta_var - sum_c[aggregated_coeff[c] * x[c]] >= aggregated_rhs
    
    Rearranged:
      theta_var >= aggregated_rhs + sum_c[aggregated_coeff[c] * x[c]]
    
    Interpretation:
      - theta_var (expected recourse cost approximation) must be at least
        the aggregated dual-based lower bound
      - This single constraint represents expected recourse across all scenarios
      - More efficient than |scenarios| separate cuts (multi-cut approach)
    """
    
    # Check if aggregated cut is violated
    # Cut violated if: theta_val < aggregated_rhs - tolerance
    # This means master underestimated the expected recourse cost
    
    if theta_val < aggregated_rhs - CutViolationTolerance:
        # Cut IS violated - need to add it to tighten master
        
        # Construct aggregated Benders cut using probability-weighted duals
        lhs = LinExpr(theta_var)  # Start with theta_var on LHS
        
        # Subtract probability-weighted dual coefficients
        for c in cities:
            lhs.addTerms(-aggregated_coeff[c], x[c])  # Add -aggregated_coeff[c]*x[c] to LHS
        
        # Add constraint to master problem
        # theta_var - sum_c[aggregated_coeff[c] * x[c]] >= aggregated_rhs
        MP.addConstr(lhs >= aggregated_rhs, name=f"AggregatedBendersCut_iter{iteration}")
        
        # Log the cut addition
        print(f"\n AGGREGATED CUT ADDED (Single-Cut Approach)")
        print(f"  Constraint: theta_var - Σ(aggregated_coeff[c]*x[c]) ≥ {aggregated_rhs:.4f}")
        print(f"  Current violation: theta_var = {theta_val:.4f} < aggregated_RHS = {aggregated_rhs:.4f}")
        print(f"  This single cut aggregates information from all {len(scenarios)} scenarios")
        
        TotalCutsAdded += 1
    else:
        # Cut NOT violated - no need to add cut
        print(f"\n No aggregated cut needed")
        print(f"  theta_var = {theta_val:.4f} ≥ aggregated_RHS = {aggregated_rhs:.4f}")
        print(f"  Expected recourse cost is accurately bounded - convergence achieved!")
    
    # ========== ITERATION SUMMARY ==========
    print(f"\n{'─'*70}")
    print(f"Iteration {iteration} Summary:")
    print(f"  Master LB = {LB:.2f}")
    print(f"  Current UB = {UB:.2f}")
    print(f"  Optimality Gap = {abs(UB - LB) / (abs(LB) + 1e-9) * 100:.2f}%")
    print(f"  Total aggregated cuts added so far: {TotalCutsAdded}")
    
    # Update best upper bound
    if UB < BestUB:
        BestUB = UB
    
    # ========== CONVERGENCE CHECK ==========
    # In single-cut: if theta_val >= aggregated_rhs, cut is not violated -> convergence
    if aggregated_rhs <= theta_val + CutViolationTolerance:
        print(f"\n[CONVERGENCE] Aggregated cut not violated - solution is optimal!")
        break

# Record end time
end_time = time.time()
total_time = end_time - start_time

# ============================================================================
# SECTION 6: FINAL RESULTS SUMMARY
# ============================================================================
"""
Final Summary:
--------------
Display final algorithm results including bounds, cuts, and runtime.

Interpretation of Results:
  - LB: Provable lower bound on optimal value (converges to optimum from below)
  - UB: Feasible solution value (converges to optimum from above)
  - Gap: Relative difference between bounds (0% = optimal)
  - Total Cuts: Number of aggregated Benders constraints added
  - Runtime: Total computation time

For Optimal Solution:
  When aggregated cut is not violated (convergence), the final x* is optimal
  for the full stochastic program, and LB ≈ UB (within numerical tolerance).

Efficiency Comparison (Single-Cut vs Multi-Cut):
  - Single-cut: |iteration| total constraints (linear in iterations)
  - Multi-cut: |iteration| * |scenarios| total constraints (cubic in size)
  - Single-cut better for large |scenarios|; multi-cut better for small |scenarios|
"""

print(f"\n{'='*70}")
print(f"FINAL RESULTS - SINGLE-CUT BENDERS DECOMPOSITION")
print(f"{'='*70}")
print(f"Total Iterations Completed:     {iteration}")
print(f"Total Aggregated Cuts Added:    {TotalCutsAdded}")
print(f"Total Computation Time:         {total_time:.3f} seconds")
print(f"\nFinal Lower Bound (LB):         {LB:.4f}")
print(f"Final Upper Bound (UB):         {BestUB:.4f}")
print(f"Optimality Gap:                 {abs(BestUB - LB) / (abs(LB) + 1e-9) * 100:.4f}%")
print(f"{'='*70}")

# Print final optimal solution if converged
if aggregated_rhs <= theta_val + CutViolationTolerance:
    print(f"\n[OPTIMAL SOLUTION FOUND]")
    print(f"\nOptimal First-Stage Allocations:")
    for c in cities:
        if xsol[c] > 1e-6:  # Only print non-zero allocations
            print(f"  x[{c}] = {xsol[c]:.4f} units")
    print(f"\nExpected Total Cost: {LB:.2f}")
    print(f"(Approximated by single variable theta_var via aggregated Benders cuts)")
else:
    print(f"\n[PARTIAL RESULTS (Not Fully Converged)]")
    print(f"Algorithm stopped after {iteration} iterations (max_iters limit or other condition)")
    print(f"Last First-Stage Allocations:")
    for c in cities:
        if xsol[c] > 1e-6:
            print(f"  x[{c}] = {xsol[c]:.4f} units")

print(f"{'='*70}\n")