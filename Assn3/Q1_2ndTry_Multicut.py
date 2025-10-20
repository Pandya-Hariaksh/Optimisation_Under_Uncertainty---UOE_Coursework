"""
================================================================================
Assignment 3 Q1 - Multi-Cut Benders Decomposition
================================================================================

ALGORITHM OVERVIEW:
-------------------
This script implements the multi-cut version of Benders decomposition to solve
a two-stage stochastic inventory distribution problem. The algorithm decomposes
the full stochastic program into:
  1. A master (first-stage) problem that decides initial allocations x[c]
  2. Multiple subproblems (one per scenario k) that compute recourse costs Q_k(x)

BENDERS DECOMPOSITION CONCEPT:
------------------------------
Instead of solving the full problem at once, Benders decomposition iteratively:
  - Solves the master problem to get a lower bound (LB) and candidate solution x*
  - Solves each scenario's subproblem to evaluate recourse costs
  - If any recourse cost exceeds its approximation, generates a "Benders cut"
  - Adds cuts back to master to refine the approximation
  - Repeats until no new cuts are needed (convergence)

MULTI-CUT vs SINGLE-CUT:
------------------------
  - Multi-cut: ONE cut per scenario per iteration (this approach)
    Advantage: More information per iteration, faster convergence
    Disadvantage: More constraints, larger master problem
  
  - Single-cut: ONE cut aggregated over all scenarios per iteration
    Advantage: Fewer constraints, smaller master problem
    Disadvantage: Less information, slower convergence

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
CutViolationTolerance = 1e-4  # Tolerance for determining if cut is violated (eps > 0)
                              # If n_k < Q_k(x) - eps, then cut is considered violated
max_iters = 200               # Maximum allowed Benders iterations (safety limit)

# ============================================================================
# SECTION 2: MASTER PROBLEM FORMULATION
# ============================================================================
"""
Master Problem (First-Stage Problem):
--------------------------------------
The master problem represents the first-stage decisions and approximate recourse costs.
It is solved repeatedly, with new Benders cuts added in each iteration.

Variables:
  - x[c]: Units to allocate to city c (first-stage decision)
  - n_vars[k]: Lower bound approximation on recourse cost Q_k(x) for scenario k
               (called "epigraph variable" - approximates the recourse function)

Objective:
  Minimize: sum_c [theta[c] * x[c]]           (first-stage cost)
          + sum_k [prob * n_vars[k]]           (expected second-stage cost)

Constraints:
  1. Sum of allocations <= I                  (budget constraint)
  2. Benders cuts (added dynamically):        (tighten recourse approximation)
     n_vars[k] >= dual-based lower bound      (one cut per scenario per iteration)

Interpretation:
  - Master seeks first-stage decisions x that minimize total cost
  - Approximate second-stage cost using n_vars variables
  - Benders cuts tighten the approximation by ensuring n_vars >= true recourse cost
"""

# Create master problem model
MP = gp.Model("Master_Multicut")
MP.Params.OutputFlag = 0  # Suppress Gurobi solver console output for clarity

# Set optimization sense (minimize)
MP.modelSense = GRB.MINIMIZE

# Decision variable x[c]: first-stage allocation to each city
# Each x[c] has lower bound 0 and is included in objective with cost theta[c]
x = {c: MP.addVar(lb=0, obj=theta[c], name=f"x_{c}") for c in cities}

# Constraint: total first-stage allocation cannot exceed central depot budget
# Interpretation: sum of units sent out <= total available inventory I
MP.addConstr(quicksum(x[c] for c in cities) <= I, name="CenterInventory")

# Epigraph variables n_vars[k]: approximate lower bounds on recourse costs
# Each n_vars[k] represents an approximation of Q_k(x) for scenario k
# Initial lower bound is 0 (will be tightened by Benders cuts)
# Included in objective with coefficient prob (expected value weighting)
n_vars = {k: MP.addVar(lb=0, obj=prob, name=f"n_{k}") for k in scenarios}

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

Duals:
  - gamma: Shadow price of capacity constraint (cost of additional budget)
  - pi[c]: Shadow price of city c demand constraint (cost per unit inventory mismatch)
  
  These duals are used to generate Benders cuts (explained in cut generation section).
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
  5. Extracts dual values (shadow prices) for generating Benders cuts

Constraints Added:
  (A) Capacity: I + sum(u[c]) >= sum(v[c]) + sum(xsol[c])
      Meaning: Total available capacity (initial + additional supply) must cover
               total recourse supply + first-stage allocation sent to cities
  
  (B) Demand Balance (for each city c):
      Yn[c] + xsol[c] + v[c] + s[c] = demand[(c,k)] + z[c] + u[c]
      
      Meaning: Material balance at city c:
      - Available: initial inventory Yn[c] + first-stage xsol[c] + recourse v[c] + shortage s[c]
      - Needed:    demand + unused inventory z[c] + supply adjustment u[c]

Dual Variables Extracted:
  - gamma (γ): Shadow price of capacity constraint
              Interpretation: Cost per unit of additional capacity
              Sign: Non-negative (by LP duality)
  
  - pi[c] (π_c): Shadow price of city c demand constraint
                Interpretation: Cost per unit of inventory at city c
                Sign: Can be positive or negative depending on problem structure

Benders Cut Generation (Uses These Duals):
  The dual solution (gamma, pi) is used to construct a lower bound on the recourse cost:
  
  Q_k(x) >= gamma * I + sum_c[pi[c] * (demand[(c,k)] - Yn[c])] - sum_c[(gamma + pi[c]) * x[c]]
  
  This is rearranged into the constraint:
  n_vars[k] - sum_c[(gamma + pi[c]) * x[c]] >= constant_term
  
  which is added to the master problem in the next section.
"""

def solve_subproblem(k, xsol):
    """
    Solve the recourse subproblem for scenario k with current first-stage solution xsol.
    
    Arguments:
      k (str): Scenario identifier (e.g., "S0")
      xsol (dict): Current first-stage solution {city_c: value_x_c}
    
    Returns:
      SPobj (float): Optimal objective value (recourse cost Q_k(xsol))
      gamma (float): Dual multiplier for capacity constraint
      pi (dict): Dictionary of dual multipliers {city_c: dual_value}
      
    Returns (None, None, None) if subproblem is infeasible.
    
    Process:
      1. Remove all constraints from previous iteration
      2. Add capacity constraint (depends on xsol)
      3. Add demand balance constraints for each city (depends on xsol)
      4. Optimize the subproblem
      5. Extract optimal value and dual solutions
      6. Return results for Benders cut generation
    """
    sp = SP[k]
    u_vars, v_vars, z_vars, s_vars = sub_vars[k]
    
    # STEP 1: Remove all previous constraints to make room for new ones
    # (Constraints change each iteration because xsol changes)
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
    # Interpretation: Material balance at city c
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
    # These represent the marginal cost of relaxing each constraint
    gamma = cap_constr.Pi      # Shadow price of capacity constraint (cost of budget)
    pi = {c: demand_constr[c].Pi for c in cities}  # Shadow price of each city's balance
    
    # STEP 6: Return results
    return SPobj, gamma, pi


# ============================================================================
# SECTION 5: MAIN BENDERS DECOMPOSITION LOOP
# ============================================================================
"""
Main Algorithm Loop:
--------------------
This loop executes the core Benders decomposition algorithm.

Iteration Structure:
  Repeat until convergence or max_iters reached:
    1. SOLVE MASTER PROBLEM
       - Get lower bound (LB) and candidate solution x*
    
    2. EVALUATE SUBPROBLEMS
       - For each scenario k, solve Q_k(x*)
       - Accumulate expected recourse cost for upper bound
    
    3. GENERATE BENDERS CUTS
       - For each scenario k where Q_k(x*) > n_vars[k]:
         * Compute cut using dual solution
         * Add constraint to master for next iteration
    
    4. CHECK CONVERGENCE
       - If no cuts added: solution is optimal
       - If cuts added: repeat with refined master problem

Key Variables:
  - LB (Lower Bound): Value of master problem objective
  - UB (Upper Bound): First-stage cost + actual recourse costs
  - TotalCutsAdded: Count of all cuts added across all iterations
  - xsol, nsol: Current master solution values

Convergence Condition:
  When no new cuts are added in an iteration, it means:
  - All n_vars[k] are "tight" (equal to their cut values)
  - All recourse costs are accurately represented in master
  - Master solution is optimal for the stochastic program
"""

# Initialize tracking variables
TotalCutsAdded = 0      # Total Benders cuts added across all iterations
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
    nsol = {k: n_vars[k].X for k in scenarios}  # Current recourse approximations
    
    # Initialize upper bound calculation with first-stage cost
    # UB = first-stage cost + expected recourse cost
    UB = sum(theta[c]*xsol[c] for c in cities)
    
    # ========== ITERATION HEADER OUTPUT ==========
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration}")
    print(f"{'='*70}")
    print(f"Lower Bound (LB) from Master: {LB:.2f}")
    print(f"\nCurrent First-Stage Solution x*:")
    for c in cities:
        print(f"  x[{c}] = {xsol[c]:.4f}")
    
    # Reset cut counter for this iteration
    cuts_added_this_iter = 0
    
    # ========== PROCESS EACH SCENARIO ==========
    print(f"\n{'─'*70}")
    print(f"Evaluating Recourse Subproblems:")
    print(f"{'─'*70}")
    
    for k in scenarios:
        # Solve recourse subproblem for this scenario at current x*
        SPobj, gamma, pi = solve_subproblem(k, xsol)
        
        # Check for infeasibility
        if SPobj is None:
            continue
        
        # Add weighted recourse cost to upper bound
        UB += prob * SPobj
        
        # Print scenario evaluation results
        print(f"\nScenario {k}:")
        print(f"  Recourse Cost Q_{k}(x*) = {SPobj:.2f}")
        print(f"  Dual for Capacity (γ) = {gamma:.4f}")
        for c in cities:
            print(f"    π[{c}] = {pi[c]:.4f}", end="  ")
            if (cities.index(c) + 1) % 2 == 0:
                print()
        
        # ========== BENDERS CUT GENERATION ==========
        # Check if recourse cost exceeds the approximation (cut is violated)
        # Cut violated if: n_vars[k] < SPobj - tolerance
        # This means master underestimated the recourse cost
        
        if nsol[k] < SPobj - CutViolationTolerance:
            # Cut IS violated - need to add it to tighten master
            
            # Construct Benders cut using dual solution
            # Cut form: n[k] - sum_c[(γ + π[c]) * x[c]] >= RHS
            
            lhs = LinExpr(n_vars[k])  # Start with n[k] on LHS
            cut_coeff = {}
            
            # Subtract dual-weighted first-stage variables
            for c in cities:
                coeff = pi[c] + gamma
                lhs.addTerms(-coeff, x[c])  # Add -(coeff)*x[c] to LHS
                cut_coeff[c] = coeff
            
            # Compute RHS of cut
            # RHS = SPobj - sum_c[(γ + π[c]) * xsol[c]]
            rhs = SPobj - sum(cut_coeff[c]*xsol[c] for c in cities)
            
            # Add constraint to master problem
            # n[k] >= RHS + sum_c[(γ + π[c]) * x[c]]
            MP.addConstr(lhs >= rhs, name=f"BendersCut_{k}_iter{iteration}")
            
            # Log the cut addition
            print(f"   CUT ADDED for scenario {k}")
            print(f"    Constraint: n[{k}] - Σ((γ+π[c])*x[c]) ≥ {rhs:.4f}")
            print(f"    Current violation: n[{k}] = {nsol[k]:.4f} < Q_{k}(x*) = {SPobj:.4f}")
            
            TotalCutsAdded += 1
            cuts_added_this_iter += 1
        else:
            # Cut NOT violated - no need to add cut for this scenario
            print(f"   No cut needed (n[{k}] = {nsol[k]:.4f} ≥ Q_{k}(x*) = {SPobj:.4f})")
    
    # ========== ITERATION SUMMARY ==========
    print(f"\n{'─'*70}")
    print(f"Iteration {iteration} Summary:")
    print(f"  Master LB = {LB:.2f}")
    print(f"  Current UB = {UB:.2f}")
    print(f"  Optimality Gap = {abs(UB - LB) / (abs(LB) + 1e-9) * 100:.2f}%")
    print(f"  Cuts added this iteration: {cuts_added_this_iter}")
    print(f"  Total cuts added so far: {TotalCutsAdded}")
    
    # Update best upper bound
    if UB < BestUB:
        BestUB = UB
    
    # ========== CONVERGENCE CHECK ==========
    if cuts_added_this_iter == 0:
        print(f"\n[CONVERGENCE] No new cuts added - solution is optimal!")
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
  - Total Cuts: Number of Benders constraints added
  - Runtime: Total computation time

For Optimal Solution:
  When cuts_added = 0 (convergence), the final x* is optimal for the full
  stochastic program, and LB ≈ UB (within numerical tolerance).
"""

print(f"\n{'='*70}")
print(f"FINAL RESULTS - MULTI-CUT BENDERS DECOMPOSITION")
print(f"{'='*70}")
print(f"Total Iterations Completed:     {iteration}")
print(f"Total Benders Cuts Added:       {TotalCutsAdded}")
print(f"Total Computation Time:         {total_time:.3f} seconds")
print(f"\nFinal Lower Bound (LB):         {LB:.4f}")
print(f"Final Upper Bound (UB):         {BestUB:.4f}")
print(f"Optimality Gap:                 {abs(BestUB - LB) / (abs(LB) + 1e-9) * 100:.4f}%")
print(f"{'='*70}")

# Print final optimal solution if converged
if cuts_added_this_iter == 0:
    print(f"\n[OPTIMAL SOLUTION FOUND]")
    print(f"\nOptimal First-Stage Allocations:")
    for c in cities:
        if xsol[c] > 1e-6:  # Only print non-zero allocations
            print(f"  x[{c}] = {xsol[c]:.4f} units")
    print(f"\nExpected Total Cost: {LB:.2f}")
else:
    print(f"\n[PARTIAL RESULTS (Not Fully Converged)]")
    print(f"Algorithm stopped after {iteration} iterations (max_iters limit or other condition)")
    print(f"Last First-Stage Allocations:")
    for c in cities:
        if xsol[c] > 1e-6:
            print(f"  x[{c}] = {xsol[c]:.4f} units")

print(f"{'='*70}\n")