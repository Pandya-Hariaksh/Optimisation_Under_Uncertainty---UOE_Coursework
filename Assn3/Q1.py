"""
================================================================================
Assignment 3 Q1 - Benders Decomposition for Two-Stage Stochastic Program
================================================================================

ALGORITHM OVERVIEW:
-------------------
This script implements Benders decomposition (both multi-cut and single-cut capable)
for a two-stage stochastic inventory distribution problem. The algorithm decomposes
the full problem into:
  1. Master (first-stage) problem: Decides initial allocations x[c]
  2. Multiple subproblems (one per scenario k): Computes recourse costs Q_k(x)

BENDERS DECOMPOSITION CONCEPT:
------------------------------
Instead of solving the full stochastic program at once, Benders iteratively:
  - Solves the master problem to get lower bound (LB) and candidate x*
  - Solves each scenario's subproblem to evaluate actual recourse costs
  - If recourse cost exceeds its approximation, generates a Benders cut
  - Adds cuts back to master to refine approximation
  - Repeats until convergence (no new cuts added)

MATHEMATICAL FORMULATION:
-------------------------
Two-Stage Stochastic Program:

  Stage 1 (First-Stage/Here-and-Now):
    Minimize: sum_c [theta[c] * x[c]]
    Subject to:
      - sum_c [x[c]] <= I                (budget constraint)
      - x[c] >= 0 for all c

  Stage 2 (Second-Stage/Wait-and-See, per scenario k):
    Minimize: sum_c [theta_s[c]*(u[c] + v[c]) + h*z[c] + g*s[c]]
    Subject to:
      - I + sum_c(u[c]) >= sum_c(v[c]) + sum_c(x[c])  (capacity)
      - For each city c:
        Yn[c] + x[c] + v[c] + s[c] = demand[(c,k)] + z[c] + u[c]  (balance)
      - All variables >= 0

  Full Problem:
    Minimize: sum_c [theta[c]*x[c]] + sum_k [prob(k)*Q_k(x)]
    
    where Q_k(x) is the optimal value of Stage 2 problem for scenario k with x fixed

MASTER PROBLEM APPROXIMATION:
-----------------------------
The master problem approximates recourse costs using epigraph variables:

  Minimize: sum_c [theta[c] * x[c]]         (first-stage cost)
          + sum_k [prob(k) * n[k]]           (approximated recourse)
  
  Subject to:
    - sum_c [x[c]] <= I
    - Benders cuts (added dynamically):
      n[k] >= dual-based lower bound on Q_k(x)  [one per scenario per iteration]
    - x[c] >= 0, n[k] >= 0

SUBPROBLEM FORMULATION:
-----------------------
For each scenario k with fixed first-stage x, the subproblem minimizes recourse cost.

Variables:
  u[c]: Additional supply adjustment from center (can be positive or negative)
  v[c]: Additional demand variable (auxiliary for constraint formulation)
  z[c]: Unused/excess inventory at city c (slack variable, >= 0)
  s[c]: Shortage at city c (slack variable, >= 0)

Objective (per scenario k):
  Q_k(x) = Minimize: sum_c [theta_s[c]*(u[c] + v[c])]   (adjustment costs)
                   + h * sum_c [z[c]]                     (excess inventory penalty)
                   + g * sum_c [s[c]]                     (shortage penalty)

Constraints (Added each iteration with updated x):
  (A) Capacity: I + sum_c(u[c]) >= sum_c(v[c]) + sum_c(x[c])
      Interpretation: Available capacity (initial + adjustments) 
                      must cover recourse supply + first-stage allocation
  
  (B) Demand Balance (per city c):
      Yn[c] + x[c] + v[c] + s[c] = demand[(c,k)] + z[c] + u[c]
      Interpretation: Material balance at city c
      - LHS (available): Initial inventory + first-stage + recourse + shortage
      - RHS (needed): Demand + excess inventory + adjustment

DUAL VARIABLES (FOR BENDERS CUTS):
----------------------------------
The subproblem dual problem provides shadow prices (dual values) for constraints:

  gamma: Shadow price of capacity constraint (Constraint A)
         - How much Q_k increases per unit of additional depot capacity
         - Positive if capacity is binding, zero if slack

  pi[c]: Shadow price of city c demand constraint (Constraint B)
         - How much Q_k increases per unit of additional inventory at city c
         - Can be positive (inventory shortage) or negative (excess)

BENDERS CUT DERIVATION:
-----------------------
By LP strong duality theory:
  Q_k(x) = min_variables [obj] = max_duals [dual_obj]

For this problem, the dual relationship gives:
  Q_k(x) >= Q_k(x*) + sum_c [(pi[c] + gamma) * (x[c] - x*[c])]
  
where x* is the current first-stage solution and (pi[c] + gamma) are dual coefficients.

Rearranging:
  Q_k(x) >= Q_k(x*) - sum_c [(pi[c] + gamma) * x*[c]] + sum_c [(pi[c] + gamma) * x[c]]

This gives the Benders cut:
  n[k] >= RHS + sum_c [(pi[c] + gamma) * x[c]]
  
or equivalently:
  n[k] - sum_c [(pi[c] + gamma) * x[c]] >= RHS
  
where RHS = Q_k(x*) - sum_c [(pi[c] + gamma) * x*[c]]

This cut is a valid lower bound on Q_k(x) and is added to the master problem.

================================================================================
"""

import gurobipy as gp
from gurobipy import GRB, LinExpr, quicksum
import ReadData as Data

# ============================================================================
# SECTION 1: DATA LOADING AND PROBLEM PARAMETERS
# ============================================================================
"""
Load problem data from external module and initialize parameters.

Data Components:
  - cities: List of city nodes (e.g., ['C0', 'C1', ..., 'C9'])
  - scenarios: List of demand scenarios (e.g., ['S0', 'S1', ...])
  - theta[c]: First-stage cost per unit shipped to city c
  - theta_s[c]: Second-stage adjustment cost per unit
  - h: Cost per unit of unused/excess inventory
  - g: Cost per unit of shortage (unmet demand)
  - I: Total inventory available at central depot
  - Yn[c]: Initial inventory already located at city c
  - demand[(c,k)]: Demand at city c in scenario k
  - prob: Scenario probability (typically 1/|scenarios|)

Algorithm Parameters:
  - CutViolationTolerance: Threshold for detecting when cuts should be added
                           If n[k] < Q_k(x) - tolerance, cut is violated
  - max_iters: Safety limit on number of iterations (default 200)

Two-Stage Structure:
  - Stage 1: Make first-stage decisions x[c] before demand is known
  - Stage 2: After demand revealed, solve recourse problem for each scenario
"""

# Load all data from ReadData module
cities    = Data.cities          # List of city identifiers
scenarios = Data.scenarios       # List of scenario identifiers
prob      = Data.prob            # Scenario probability (e.g., 1/len(scenarios))
theta     = Data.theta           # First-stage cost per unit (dict indexed by city)
theta_s   = Data.theta_s         # Second-stage adjustment cost (dict)
h         = Data.h               # Excess inventory penalty cost
g         = Data.g               # Shortage penalty cost
I         = Data.I               # Total depot inventory available
Yn        = Data.Yn              # Initial inventory per city (dict)
demand    = Data.demand          # Demand dict: demand[(city, scenario)]

# Algorithm tolerance for cut violation detection
# If n[k] < Q_k(x) - CutViolationTolerance, the cut is considered violated
# and a new Benders cut will be added to tighten the approximation
CutViolationTolerance = 1e-4

# ============================================================================
# SECTION 2: MASTER PROBLEM FORMULATION
# ============================================================================
"""
Master Problem (First-Stage Problem):
--------------------------------------
The master problem represents the first-stage decisions and approximates
the expected recourse cost using epigraph variables and Benders cuts.

Problem Structure:
  Minimize: sum_c [theta[c] * x[c]]      (first-stage cost)
          + sum_k [prob(k) * n[k]]       (expected recourse cost - approximated)
  
  Subject to:
    - sum_c [x[c]] <= I                  (budget constraint)
    - Benders cuts (added dynamically):
      n[k] >= dual-based lower bound    (one cut per scenario per iteration)
    - x[c] >= 0, n[k] >= 0

Variables:
  x[c]: First-stage shipment quantity to city c (DECISION)
        - Represents units shipped from depot to city c before demand known
        - Cost = theta[c] per unit
  
  n[k]: Epigraph variable for scenario k (APPROXIMATION)
        - Lower bound approximation on true recourse cost Q_k(x)
        - Objective coefficient = prob(k) (expected value weighting)
        - Tightened by Benders cuts each iteration

Objective Interpretation:
  Total cost = first-stage shipping + expected recourse cost approximation
  
  The recourse approximation (sum of n[k]) is refined by cuts, moving towards
  the true expected recourse cost as iterations progress.

Solver Configuration:
  - OutputFlag=0: Suppress Gurobi solver output
  - Method=1: Use primal simplex (aids dual stability)
  - modelSense=MINIMIZE: Minimization problem
"""

# Create master problem
MP = gp.Model("BendersMaster")
MP.Params.OutputFlag = 0  # Suppress Gurobi solver output
MP.Params.method = 1      # Use primal simplex for dual stability
MP.modelSense = GRB.MINIMIZE

# Create first-stage decision variables x[c]
# Each x[c] has objective coefficient theta[c] (first-stage shipping cost)
x = {c: MP.addVar(lb=0, obj=theta[c], name=f"x_{c}") for c in cities}

# Budget constraint: total shipment cannot exceed depot inventory
# Interpretation: sum of all x[c] must be <= total available inventory I
MP.addConstr(quicksum(x[c] for c in cities) <= I, name="CenterInventory")

# Create epigraph variables n[k] for recourse cost approximation
# Each n[k] approximates Q_k(x) for scenario k with objective coefficient prob(k)
n_vars = {k: MP.addVar(lb=0, obj=prob, name=f"n_{k}") for k in scenarios}

# Finalize master problem structure
MP.update()

print(f"[MASTER] Master problem created with {len(cities)} x-variables "
      f"and {len(scenarios)} epigraph variables")

# ============================================================================
# SECTION 3: SUBPROBLEM TEMPLATE AND CONSTRUCTION
# ============================================================================
"""
Subproblem Templates (Second-Stage Recourse Problems):
-------------------------------------------------------
For each scenario k, a subproblem solves the recourse problem with x fixed.

The subproblem is created as a template with variables but no constraints.
Constraints are added fresh each iteration with updated x values and then
removed after solution (since constraints change each iteration).

Subproblem Variables:
  u[c]: Additional supply adjustment from center to city c
        - Non-negative; represents supply addition
  v[c]: Demand auxiliary variable
        - Non-negative; helps balance constraints
  z[c]: Unused/excess inventory at city c
        - Non-negative slack variable; cost = h per unit
  s[c]: Shortage at city c (unmet demand)
        - Non-negative slack variable; cost = g per unit

Subproblem Objective (same for all scenarios):
  Q_k(x) = Minimize: sum_c [theta_s[c]*(u[c] + v[c])]   (adjustment costs)
                   + h * sum_c [z[c]]                     (excess penalty)
                   + g * sum_c [s[c]]                     (shortage penalty)
  
  This objective is scenario-independent; only constraints change per scenario.

Why Template Approach:
  - Constraints change each iteration (they depend on current x and scenario demand)
  - Faster to remove and re-add constraints than rebuild subproblem
  - Preserves variable references for efficient solving
"""

def build_subproblem(model_name):
    """
    Factory function to create an empty subproblem model template.
    
    The subproblem is created with decision variables but WITHOUT constraints.
    Constraints are added in solve_subproblem() before each solve, and removed
    after solution to prepare for next iteration.
    
    Arguments:
      model_name (str): Name for the Gurobi model (e.g., "SP_S0")
    
    Returns:
      sp (gp.Model): Empty subproblem model with variables but no constraints
      (u, v, z, s) tuple: Variable dictionaries for the four variable groups
    
    Variable Groups Created:
      u: dict of u[c] variables (additional supply adjustment)
      v: dict of v[c] variables (demand adjustment)
      z: dict of z[c] variables (unused inventory)
      s: dict of s[c] variables (shortage)
    
    Objective Set (Fixed):
      Minimizes total recourse cost = adjustment + excess + shortage
      This objective is the same for all scenarios
    
    Constraints:
      NOT added here (added per iteration in solve_subproblem)
      Reason: Constraints depend on x solution and scenario demand
    """
    sp = gp.Model(model_name)
    sp.Params.OutputFlag = 0  # Suppress output
    
    # Create four groups of decision variables (one per city)
    u = {c: sp.addVar(lb=0, name=f"u_{c}") for c in cities}
    v = {c: sp.addVar(lb=0, name=f"v_{c}") for c in cities}
    z = {c: sp.addVar(lb=0, name=f"z_{c}") for c in cities}
    s = {c: sp.addVar(lb=0, name=f"s_{c}") for c in cities}
    
    # Set objective function (same for all scenarios, scenario-independent)
    sp.setObjective(
        quicksum(theta_s[c]*(u[c] + v[c]) for c in cities) +  # Adjustment costs
        h * quicksum(z[c] for c in cities) +                   # Excess penalty
        g * quicksum(s[c] for c in cities),                    # Shortage penalty
        GRB.MINIMIZE
    )
    sp.update()
    return sp, u, v, z, s

# Create one subproblem instance per scenario
# These are reused across iterations with constraints updated each time
SP = {}        # Dictionary: SP[k] = subproblem model for scenario k
sub_vars = {}  # Dictionary: sub_vars[k] = (u, v, z, s) variable tuples for scenario k

for k in scenarios:
    sp, u, v, z, s = build_subproblem(f"SP_{k}")
    SP[k] = sp
    sub_vars[k] = (u, v, z, s)

print(f"[SUBPROBLEMS] Created {len(scenarios)} subproblem templates\n")

# ============================================================================
# SECTION 4: SUBPROBLEM SOLUTION FUNCTION
# ============================================================================
"""
Function: solve_subproblem(k, xsol)
-----------------------------------
Solves the recourse subproblem for scenario k with fixed first-stage solution xsol.

This function is called for each scenario in each Benders iteration.

Process:
  1. Remove constraints from previous iteration
  2. Add scenario-specific constraints using current xsol
  3. Solve the recourse problem
  4. Extract optimal value (recourse cost Q_k)
  5. Extract dual values (shadow prices) for cut generation
  6. Return results

Arguments:
  k (str): Scenario identifier (e.g., "S0")
  xsol (dict): Current first-stage solution {city: quantity}

Returns:
  SPobj (float): Optimal objective value = Q_k(xsol) [recourse cost]
  gamma (float): Dual value of capacity constraint
  pi (dict): Dual values of demand constraints, indexed by city

Constraints Added (Scenario k, First-Stage x = xsol):
  
  (A) Capacity Constraint:
      I + sum_c(u[c]) >= sum_c(v[c]) + sum_c(xsol[c])
      
      Interpretation:
        - Available capacity: I (initial) + u (adjustments)
        - Required: v (recourse) + x (first-stage sent to cities)
      
      Dual gamma: How much Q_k increases per unit of additional depot capacity
  
  (B) Demand Balance Constraints (per city c):
      Yn[c] + xsol[c] + v[c] + s[c] = demand[(c,k)] + z[c] + u[c]
      
      Rearranging:
      - LHS: Initial Yn[c] + first-stage xsol[c] + recourse v[c] + shortage s[c]
      - RHS: Demand + excess z[c] + adjustment u[c]
      
      Interpretation: Material balance at city c
        - What arrives: initial + first-stage + recourse + shortage
        - What's needed: demand + excess (waste) + adjustments
      
      Dual pi[c]: How much Q_k increases per unit of additional inventory at city c

Dual Variable Extraction:
  After solving, extract dual values (shadow prices) from constraints.
  These duals are used to generate Benders cuts that tighten the master.

Cut Generation (Simplified Explanation):
  The duals gamma and pi[c] satisfy the LP dual optimality conditions.
  They enable us to construct a lower bound on Q_k(x) that is tight at xsol:
  
  Q_k(x) >= Q_k(xsol) + sum_c [(gamma + pi[c]) * (x[c] - xsol[c])]
  
  This linear lower bound is added as a Benders cut to the master.
"""

def solve_subproblem(k, xsol):
    """
    Solve the recourse subproblem for scenario k with fixed x = xsol.
    
    Arguments:
      k (str): Scenario identifier
      xsol (dict): Current first-stage solution
    
    Returns:
      (SPobj, gamma, pi) if optimal
      (None, None, None) if infeasible or not optimal
    
    Steps:
      1. Get subproblem model and variables
      2. Remove all constraints from previous iteration
      3. Add capacity constraint (using xsol)
      4. Add demand balance constraints for each city (using xsol and scenario k)
      5. Optimize subproblem
      6. Extract and return objective and duals
    """
    sp = SP[k]
    u, v, z, s = sub_vars[k]
    
    # STEP 1: Clear previous constraints
    # Constraints must change each iteration because x changes and scenario is fixed
    sp.remove(sp.getConstrs())
    
    # STEP 2: Add capacity constraint for this scenario and xsol
    # Capacity: I + sum_c(u[c]) >= sum_c(v[c]) + sum_c(xsol[c])
    cap_constr = sp.addConstr(
        I + quicksum(u[c] for c in cities) 
        >= quicksum(v[c] for c in cities) + sum(xsol[c] for c in cities),
        name="Capacity"
    )
    
    # STEP 3: Add demand balance constraint for each city c
    # For city c: Yn[c] + xsol[c] + v[c] + s[c] = demand[(c,k)] + z[c] + u[c]
    DemandConstr = {}
    for c in cities:
        DemandConstr[c] = sp.addConstr(
            Yn[c] + xsol[c] + v[c] + s[c] 
            == demand[(c, k)] + z[c] + u[c],
            name=f"Demand_{c}"
        )
    
    sp.update()
    
    # STEP 4: Optimize subproblem
    sp.optimize()
    
    # Check if optimization was successful
    if sp.status != GRB.OPTIMAL:
        print(f"[ERROR] Subproblem for scenario {k} not optimal (status={sp.status})")
        return None, None, None
    
    # STEP 5: Extract objective value (recourse cost)
    SPobj = sp.objVal
    
    # STEP 6: Extract dual values (shadow prices) from constraints
    # gamma: dual of capacity constraint (affects x and I)
    gamma = cap_constr.Pi
    
    # pi[c]: dual of city c demand constraint (affects Yn[c] and x[c])
    pi = {c: DemandConstr[c].Pi for c in cities}
    
    return SPobj, gamma, pi

# ============================================================================
# SECTION 5: MAIN BENDERS DECOMPOSITION LOOP
# ============================================================================
"""
Main Benders Decomposition Algorithm:
--------------------------------------
Iteratively solves master problem and subproblems until convergence.

Algorithm Flow (Per Iteration):
  1. SOLVE MASTER PROBLEM
     - Get lower bound (LB) = master objective value
     - Extract candidate solution x*
     - Extract current epigraph variable values n*[k]
  
  2. EVALUATE ALL SUBPROBLEMS
     - For each scenario k:
       * Solve recourse problem Q_k(x*) with x = x*
       * Add weighted cost to upper bound: UB += prob(k) * Q_k(x*)
       * Extract dual values for cut generation
  
  3. GENERATE AND ADD BENDERS CUTS
     - For each scenario k where Q_k(x*) > n*[k] (approximation violated):
       * Use duals gamma and pi[c] to construct Benders cut
       * Add cut to master for next iteration
       * Mark that a cut was found
  
  4. CHECK CONVERGENCE
     - If no cuts were added in this iteration: CONVERGED
     - All n*[k] are now tight (equal to true Q_k(x*) values)
     - x* is optimal for the stochastic program
     - Stop iterations

Key Variables:
  - LB (Lower Bound): Master problem objective value
    * Lower bound on optimal solution value
    * Converges from below
  
  - UB (Upper Bound): First-stage cost + sum(prob*Q_k(x*))
    * Upper bound on optimal solution value
    * Feasible solution cost
    * Converges from above
  
  - Gap: (UB - LB) / LB
    * Optimality gap (0% when converged)
    * Decreases as algorithm progresses
  
  - CutFound: Boolean flag
    * True if any cuts were added in current iteration
    * False means convergence achieved

Benders Cut Formula:
  For scenario k with current solution x*:
    Cut: n[k] - sum_c[(gamma + pi[c])*x[c]] >= SPobj - sum_c[(gamma + pi[c])*x*[c]]
    
    Where:
      - gamma: dual of capacity constraint
      - pi[c]: dual of city c demand constraint
      - SPobj: Q_k(x*) = optimal recourse cost at x*
    
    This cut is a valid lower bound on Q_k(x) for any x and is tight at x*.

Convergence Criteria:
  - When no new cuts are added in an iteration
  - All epigraph variables n[k] are at their lower bound values (cuts)
  - The approximation equals the true recourse costs
  - LB = final master objective value
  - UB = best feasible solution value
"""

# Initialize tracking variables
TotalCutsAdded = 0   # Count total cuts added across all iterations
NoIters = 0          # Iteration counter
BestUB = float("inf")  # Best (lowest) upper bound found
CutFound = True      # Flag: whether cuts were added in current iteration

print(f"{'='*70}")
print(f"STARTING BENDERS DECOMPOSITION")
print(f"{'='*70}\n")

# Main iteration loop
while CutFound and NoIters < 200:
    NoIters += 1
    CutFound = False  # Reset; set to True if any cut is added this iteration
    
    # ========== STEP 1: SOLVE MASTER PROBLEM ==========
    MP.update()
    MP.optimize()
    
    # Check master problem status
    if MP.status != GRB.OPTIMAL:
        print("[ERROR] Master problem infeasible!")
        break
    
    # Extract master problem solution
    MPobj = MP.objVal              # Lower bound from master
    xsol = {c: x[c].X for c in cities}  # Current x solution
    nsol = {k: n_vars[k].X for k in scenarios}  # Current n solution

    # Initialize upper bound calculation
    # UB = first-stage cost + sum(prob*Q_k(x*))
    UB = sum(theta[c] * xsol[c] for c in cities)

    # ========== ITERATION HEADER ==========
    print(f"{'─'*70}")
    print(f"ITERATION {NoIters}")
    print(f"{'─'*70}")
    print(f"Lower Bound (LB) = {MPobj:.2f}")
    print(f"Current x solution:")
    for c in cities:
        print(f"  x[{c}] = {xsol[c]:.4f}")
    print()

    # ========== STEP 2-3: EVALUATE SUBPROBLEMS AND ADD CUTS ==========
    for k in scenarios:
        # Solve recourse problem for this scenario
        SPobj, gamma, pi = solve_subproblem(k, xsol)
        
        if SPobj is None:
            continue
        
        # Add weighted recourse cost to upper bound
        UB += prob * SPobj

        # Print scenario solution and duals
        print(f"Scenario {k}: Q_k(x*) = {SPobj:.2f}")
        print(f"  Dual for Capacity (gamma) = {gamma:.3f}")
        for c in cities:
            print(f"  Dual for Demand[{c}] (pi) = {pi[c]:.3f}")

        # ========== CHECK IF CUT IS VIOLATED ==========
        """
        A Benders cut is violated if the current approximation n[k] is 
        less than the true recourse cost Q_k(x*) by more than tolerance.
        
        Violation: n[k] < Q_k(x*) - CutViolationTolerance
        
        If violated, we add a Benders cut to tighten the approximation.
        """
        
        if nsol[k] < SPobj - CutViolationTolerance:
            # Cut IS violated - add it to master
            
            CutFound = True
            TotalCutsAdded += 1
            
            # Construct Benders cut
            # Cut form: n[k] - sum_c[(gamma + pi[c])*x[c]] >= RHS
            # Where RHS = Q_k(x*) - sum_c[(gamma + pi[c])*x*[c]]
            
            lhs = LinExpr(n_vars[k])  # Start with n[k]
            
            # Add -(gamma + pi[c])*x[c] for each city
            lhs -= quicksum((pi[c] + gamma) * x[c] for c in cities)
            
            # Compute RHS
            rhs = SPobj - sum((pi[c] + gamma) * xsol[c] for c in cities)
            
            # Add cut to master problem
            MP.addConstr(lhs >= rhs, name=f"BendersCut_{k}_it{NoIters}")
            
            print(f"  ✓ BENDERS CUT ADDED for scenario {k}:")
            print(f"    n[{k}] - Σ((γ+π[c])*x[c]) >= {rhs:.2f}")
            print(f"    (violation: n[{k}]={nsol[k]:.2f} < Q_k={SPobj:.2f})")
        else:
            # No cut needed
            print(f"  ✗ No cut needed: n[{k}]={nsol[k]:.2f} >= {SPobj:.2f}")
        
        print()

    # ========== STEP 4: ITERATION SUMMARY ==========
    # Update best upper bound found
    if UB < BestUB:
        BestUB = UB
    
    # Calculate optimality gap
    gap_pct = 100.0 * (UB - MPobj) / abs(MPobj) if abs(MPobj) > 1e-9 else 0.0
    
    print(f"Iteration {NoIters} Summary:")
    print(f"  Master LB = {MPobj:.2f}")
    print(f"  Current UB = {UB:.2f} (Best UB = {BestUB:.2f})")
    print(f"  Optimality Gap = {gap_pct:.2f}%")
    print(f"  Cuts added this iteration = {1 if CutFound else 0}")
    print(f"  Total cuts so far = {TotalCutsAdded}\n")
    
    # ========== CONVERGENCE CHECK ==========
    if not CutFound:
        print("[CONVERGENCE] No new cuts added - solution is optimal!")
        break

# ============================================================================
# SECTION 6: FINAL RESULTS SUMMARY
# ============================================================================
"""
Final Summary After Convergence:
--------------------------------
Reports the optimal or best solution found by Benders decomposition.

Output Includes:
  - Total iterations executed
  - Final lower bound (LB)
  - Final upper bound (UB)
  - Total cuts added
  - Optimal first-stage solution x[c]
  - Achieved optimality gap

Interpretation:
  - If converged: LB = UB (within numerical tolerance), gap = 0%
  - x* is provably optimal for the two-stage stochastic program
  - If not converged: best feasible solution is reported (UB)
"""

print(f"\n{'='*70}")
print(f"FINAL SUMMARY - BENDERS DECOMPOSITION")
print(f"{'='*70}")
print(f"Total Iterations: {NoIters}")
if MP.status == GRB.OPTIMAL:
    finalLB = MPobj
    print(f"Final Lower Bound (LB): {finalLB:.2f}")
    print(f"Final Upper Bound (UB): {BestUB:.2f}")
    print(f"Optimality Gap: {100.0*(BestUB - finalLB)/(abs(finalLB)+1e-9):.4f}%")
    print(f"Total Cuts Added: {TotalCutsAdded}")
    print(f"\nOptimal First-Stage Shipments:")
    for c in cities:
        if x[c].X > 1e-6:  # Only print non-zero values
            print(f"  x[{c}] = {x[c].X:.2f} units")
    print(f"\nOptimal Total Cost: ${finalLB:.2f}")
else:
    print(f"Master problem status: {MP.status}")
    print("Could not find optimal solution.")

print(f"{'='*70}\n")