"""
================================================================================
Assignment 3 Q2 - Sample Average Approximation for Two-Stage Stochastic Program
================================================================================

PROBLEM SETUP:
--------------
Two-stage stochastic linear program with Poisson-distributed uncertainty.

Stage 1: Choose x in [0, 5]
  Cost: -0.75 * x

Stage 2: After uncertainty xi is revealed, solve:
  Minimize: -v1 + 3*v2 + v3 + v4
  Subject to:
    -v1 + v2 - v3 + v4 = xi + 0.5*x
    -v1 + v2 + v3 - v4 = 1 + xi + 0.25*x
    v1, v2, v3, v4 greater than or equal to 0

Total Objective: -0.75*x plus E[recourse cost]

SOLUTION APPROACH - SAA WITH VALIDATION:
-----------------------------------------
Phase 1: Generate M=10 independent SAA batches (each with N=30 scenarios)
         Record optimal x and objective from each batch
         Keep best candidate x*

Phase 2: Compute lower bound from M estimates
         Use t-distribution to form 95 percent confidence interval

Phase 3: Evaluate best x* on N_tilde=500 independent test scenarios
         Compute upper bound with 95 percent confidence interval

Phase 4: Estimate optimality gap between bounds

Confidence Intervals:
  Lower bound uses t-distribution (small sample M=10)
  Upper bound uses normal approximation (large sample N_tilde=500)

================================================================================
"""

import math
import numpy as np
from gurobipy import Model, GRB
import statistics
from scipy.stats import t

# ============================================================================
# FUNCTION 1: SOLVE ONE SAA BATCH (EXTENSIVE FORM)
# ============================================================================
"""
Function: solve_SAA_problem(xi_sample)

Builds and solves the extensive form combining all N scenarios and first-stage
decision x into one large optimization problem.

Formulation:
  Minimize: -0.75*x + (1/N) * sum over i of (-v1[i] + 3*v2[i] + v3[i] + v4[i])
  
  Subject to for each scenario i:
    -v1[i] + v2[i] - v3[i] + v4[i] = xi_sample[i] + 0.5*x
    -v1[i] + v2[i] + v3[i] - v4[i] = 1 + xi_sample[i] + 0.25*x
    v1[i], v2[i], v3[i], v4[i] greater than or equal to 0
    0 less than or equal to x less than or equal to 5

Variables:
  x: first-stage decision (bounded to [0, 5])
  v1[i], v2[i], v3[i], v4[i]: second-stage variables for scenario i

Returns:
  tuple (objective_value, x_optimal)
    objective_value: optimal cost for this batch
    x_optimal: optimal first-stage decision
"""

def solve_SAA_problem(xi_sample):
    """Solve extensive form for one SAA batch with N scenarios."""
    N = len(xi_sample)
    model = Model("SAA_ExtensiveForm")
    model.setParam("OutputFlag", 0)

    # First-stage variable
    x = model.addVar(lb=0, ub=5, name="x", vtype=GRB.CONTINUOUS)
    
    # Second-stage variables for each scenario
    v1 = model.addVars(N, lb=0, name="v1")
    v2 = model.addVars(N, lb=0, name="v2")
    v3 = model.addVars(N, lb=0, name="v3")
    v4 = model.addVars(N, lb=0, name="v4")

    # Objective: first-stage cost minus recourse cost (averaged over scenarios)
    obj_expr = -0.75 * x
    for i in range(N):
        obj_expr += (1.0 / N) * (-v1[i] + 3 * v2[i] + v3[i] + v4[i])
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Linking constraints: connect first-stage x to second-stage v variables
    for i in range(N):
        xi_val = xi_sample[i]
        model.addConstr(-v1[i] + v2[i] - v3[i] + v4[i] == xi_val + 0.5 * x)
        model.addConstr(-v1[i] + v2[i] + v3[i] - v4[i] == 1 + xi_val + 0.25 * x)

    model.optimize()
    return model.objVal, x.X

# ============================================================================
# FUNCTION 2: OUT-OF-SAMPLE EVALUATION
# ============================================================================
"""
Function: evaluate_candidate(x_candidate, xi_eval)

Takes a fixed first-stage decision x_candidate and evaluates its performance
on a large independent test set xi_eval.

Process:
  For each test scenario xi_val in xi_eval:
    Solve the second-stage subproblem with x set to x_candidate and xi to xi_val
    Record the recourse cost Q(x_candidate, xi_val)

After all scenarios:
  Compute total cost = -0.75*x_candidate + recourse_cost for each scenario
  Return mean and standard deviation of total costs

Why This is Upper Bound:
  x_candidate was optimized on training data (SAA batch)
  Evaluating on independent test data gives a feasible solution
  Any feasible solution is an upper bound on true optimum

Returns:
  tuple (mean_total_cost, stdev_total_cost)
    Used to form 95 percent confidence interval for upper bound
"""

def evaluate_candidate(x_candidate, xi_eval):
    """Evaluate fixed x on large out-of-sample test set."""
    recourse_values = []

    for xi_val in xi_eval:
        # Create subproblem for single scenario
        ssp = Model("Subproblem")
        ssp.setParam("OutputFlag", 0)
        
        # Second-stage variables
        v1 = ssp.addVar(lb=0, name="v1")
        v2 = ssp.addVar(lb=0, name="v2")
        v3 = ssp.addVar(lb=0, name="v3")
        v4 = ssp.addVar(lb=0, name="v4")
        
        # Minimize recourse cost for this scenario
        ssp.setObjective(-v1 + 3 * v2 + v3 + v4, GRB.MINIMIZE)

        # Constraints using fixed x_candidate and current xi_val
        ssp.addConstr(-v1 + v2 - v3 + v4 == xi_val + 0.5 * x_candidate)
        ssp.addConstr(-v1 + v2 + v3 - v4 == 1 + xi_val + 0.25 * x_candidate)

        ssp.optimize()
        recourse_values.append(ssp.objVal)
    
    # Compute total costs and return statistics
    total_costs = [-0.75 * x_candidate + rv for rv in recourse_values]
    return np.mean(total_costs), np.std(total_costs, ddof=1)

# ============================================================================
# MAIN ALGORITHM
# ============================================================================
"""
Main Algorithm: Multi-Batch SAA with Out-of-Sample Validation

Algorithm Parameters:
  M = 10:        Number of independent SAA batches
  N = 30:        Scenarios per batch
  N_tilde = 500: Out-of-sample test scenarios
  seed_base:     Base random seed for reproducibility

Phase 1 - GENERATE LOWER BOUND ESTIMATES:
  For each batch m from 1 to M:
    - Generate unique seed for reproducibility: seed_base plus m
    - Draw N scenarios from Poisson(0.5) distribution
    - Solve extensive form
    - Record optimal objective value
    - Track best x value found

  Result: M optimal values and best candidate x*

Phase 2 - LOWER BOUND WITH 95 PERCENT CI:
  Compute statistics from M SAA objective values:
    - Mean: LB_mean
    - Standard deviation: LB_stdev
    - Use t-distribution with df equals M minus 1
    - Confidence interval: [mean minus halfwidth, mean plus halfwidth]
    - halfwidth equals t_critical times stdev divided by sqrt(M)

  Why t-distribution?
    Small sample size (M = 10) requires t-distribution
    More conservative than normal distribution

  Interpretation:
    True optimal value is likely greater than or equal to lower bound

Phase 3 - UPPER BOUND WITH 95 PERCENT CI:
  Take best x* from Phase 1:
    - Generate N_tilde equals 500 independent test scenarios
    - Evaluate x* on each test scenario using evaluate_candidate function
    - Compute mean and std dev of total costs

  Use normal approximation for 95 percent CI:
    - z_critical equals 1.96 for large samples
    - halfwidth equals 1.96 times stdev divided by sqrt(N_tilde)

  Why normal approximation?
    Large sample size (N_tilde = 500) by Central Limit Theorem

  Interpretation:
    True optimal value is likely less than or equal to upper bound

Phase 4 - OPTIMALITY GAP:
  Worst-case gap equals upper_bound_high minus lower_bound_low
  Express as percentage relative to upper bound

  If gap is small (less than 5 percent): algorithm performed well
  If gap is large: consider increasing M or N
"""

def main():
    """Execute complete SAA algorithm with validation."""
    
    # Algorithm parameters
    M = 10              # Number of SAA batches
    N = 30              # Scenarios per batch
    N_tilde = 500       # Out-of-sample test scenarios
    seed_base = 160325  # Base seed for reproducibility

    print("=" * 80)
    print("SAMPLE AVERAGE APPROXIMATION (SAA) FOR TWO-STAGE STOCHASTIC PROGRAM")
    print("=" * 80)
    print(f"\nAlgorithm Parameters:")
    print(f"  M (SAA batches):              {M}")
    print(f"  N (scenarios per batch):      {N}")
    print(f"  N_tilde (test scenarios):     {N_tilde}")
    print(f"  Base seed:                    {seed_base}\n")

    # ========== PHASE 1: GENERATE M SAA ESTIMATES ==========
    print("=" * 80)
    print("PHASE 1: SOLVING M INDEPENDENT SAA PROBLEMS")
    print("=" * 80 + "\n")
    
    results_saa = []
    best_obj = float('inf')
    best_x = None
    
    for m in range(M):
        # Unique seed ensures different sample for each batch
        seed = seed_base + m
        np.random.seed(seed)
        
        # Draw N scenarios from Poisson(0.5)
        xi_sample = np.random.poisson(0.5, N)
        
        # Solve extensive form
        obj_val, x_val = solve_SAA_problem(xi_sample)
        results_saa.append(obj_val)
        
        # Track best solution
        if obj_val < best_obj:
            best_obj = obj_val
            best_x = x_val
        
        print(f"Batch {m + 1:2d}: Objective equals {obj_val:8.4f}, x* equals {x_val:6.4f}")
    
    print()

    # ========== PHASE 2: COMPUTE LOWER BOUND ==========
    print("=" * 80)
    print("PHASE 2: LOWER BOUND ESTIMATION (USING t-DISTRIBUTION)")
    print("=" * 80 + "\n")
    
    # Compute statistics
    LB_mean = statistics.mean(results_saa)
    LB_stdev = statistics.stdev(results_saa) if M > 1 else statistics.pstdev(results_saa)
    
    # Get t-critical value for 95 percent CI
    t_val = t.ppf(0.975, df=M - 1)
    
    # Compute confidence interval
    LB_halfwidth = t_val * LB_stdev / math.sqrt(M)
    LB_CI = (LB_mean - LB_halfwidth, LB_mean + LB_halfwidth)
    
    print(f"SAA Objectives from M equals {M} batches:")
    print(f"  {results_saa}\n")
    print(f"Statistics:")
    print(f"  Mean:                        {LB_mean:.4f}")
    print(f"  Standard deviation:          {LB_stdev:.4f}")
    print(f"  t-critical (df equals {M-1}):        {t_val:.4f}")
    print(f"  95 percent CI half-width:    {LB_halfwidth:.4f}\n")
    print(f"95 Percent Confidence Interval for Lower Bound:")
    print(f"  [{LB_CI[0]:.4f}, {LB_CI[1]:.4f}]\n")

    # ========== PHASE 3: OUT-OF-SAMPLE EVALUATION ==========
    print("=" * 80)
    print("PHASE 3: OUT-OF-SAMPLE EVALUATION (USING NORMAL APPROXIMATION)")
    print("=" * 80 + "\n")
    
    # Use best candidate from Phase 1
    print(f"Candidate solution: x* equals {best_x:.4f}")
    print(f"(From batch with lowest objective: {best_obj:.4f})\n")
    
    # Generate test scenarios with fixed seed
    np.random.seed(99999)
    xi_eval = np.random.poisson(0.5, N_tilde)
    
    # Evaluate on test set
    mean_eval, stdev_eval = evaluate_candidate(best_x, xi_eval)
    
    # Form 95 percent CI using normal approximation
    z_critical = 1.96
    UB_halfwidth = z_critical * (stdev_eval / math.sqrt(N_tilde))
    UB_CI = (mean_eval - UB_halfwidth, mean_eval + UB_halfwidth)
    
    print(f"Out-of-Sample Evaluation Statistics (N_tilde equals {N_tilde}):")
    print(f"  Mean total cost:             {mean_eval:.4f}")
    print(f"  Standard deviation:          {stdev_eval:.4f}")
    print(f"  z-critical (95 percent):     {z_critical:.4f}")
    print(f"  95 percent CI half-width:    {UB_halfwidth:.4f}\n")
    print(f"95 Percent Confidence Interval for Upper Bound:")
    print(f"  [{UB_CI[0]:.4f}, {UB_CI[1]:.4f}]\n")

    # ========== PHASE 4: OPTIMALITY GAP ==========
    print("=" * 80)
    print("PHASE 4: OPTIMALITY GAP ANALYSIS")
    print("=" * 80 + "\n")
    
    # Compute worst-case gap
    worst_gap_num = UB_CI[1] - LB_CI[0]
    worst_gap_den = max(abs(UB_CI[1]), 1e-6)
    worst_gap_pct = 100.0 * worst_gap_num / worst_gap_den
    
    print(f"Final Summary:")
    print(f"  Lower Bound 95 percent CI:   [{LB_CI[0]:.4f}, {LB_CI[1]:.4f}]")
    print(f"  Upper Bound 95 percent CI:   [{UB_CI[0]:.4f}, {UB_CI[1]:.4f}]\n")
    print(f"Optimal Value Estimate:")
    print(f"  True optimum in interval:    [{LB_CI[0]:.4f}, {UB_CI[1]:.4f}]")
    print(f"  Worst-case gap:              {worst_gap_pct:.2f} percent\n")
    
    # Assessment
    print(f"Algorithm Assessment:")
    if worst_gap_pct less than 5.0:
        print(f"  Tight bounds - excellent performance")
    elif worst_gap_pct less than 10.0:
        print(f"  Good bounds - acceptable performance")
    else:
        print(f"  Loose bounds - increase M or N for improvement")
    print()

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()