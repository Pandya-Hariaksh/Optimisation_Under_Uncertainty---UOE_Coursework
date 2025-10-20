import numpy as np
import gurobipy as gp
from gurobipy import GRB

###############################################################################
# Q3: PROBLEM-DRIVEN SCENARIO REDUCTION
#
# This script illustrates how to reduce an original scenario set (size N=100, from
# Poisson(0.5)) to a smaller set (size N'=10). We refer back to Q2's two-stage
# model:
#
#   min_{0 <= x <= 5} -0.75*x + E_xi[ Q(x, xi) ],
#
# where xi ~ Poisson(0.5), and
#
#   Q(x, xi) = min_{v1,v2,v3,v4 >= 0} [ -v1 + 3v2 + v3 + v4 ]
#             s.t. -v1 + v2 - v3 + v4 = xi + 0.5*x
#                  -v1 + v2 + v3 - v4 = 1 + xi + 0.25*x.
#
# The approach:
#   1) Generate N=100 scenarios {xi_full} from Poisson(0.5) distribution
#   2) For each scenario i, solve single-scenario subproblem => x_i^* (optimal x if only xi_i existed)
#   3) Construct NxN cost matrix V[i,j] = total cost if we "train on" xi_i but "face" xi_j
#   4) Solve scenario-clustering MIP that picks N'=10 cluster representatives
#      minimizing expected cost of mismatch between trained and actual scenarios
#   5) Compare solutions:
#      - x_full_sol: optimal first-stage decision using all 100 scenarios
#      - x_sub_sol: optimal first-stage decision using only 10 cluster representatives
#   6) Evaluate out-of-sample performance on Mtest=10,000 fresh Poisson(0.5) draws
#      to quantify cost of scenario reduction
###############################################################################

def solve_single_scenario_subproblem(xi_val):
    """
    For Q2: Solve the single-scenario version of the 2-stage model for demand xi_val.
    
    Formulation:
      min  -0.75*x + [ -v1 + 3*v2 + v3 + v4 ]
      s.t.  0 <= x <= 5
            -v1 + v2 - v3 + v4 = xi_val + 0.5*x          (demand constraint 1)
            -v1 + v2 + v3 - v4 = 1 + xi_val + 0.25*x     (demand constraint 2)
            v1, v2, v3, v4 >= 0
    
    Purpose: Find the best first-stage decision x if only scenario xi_val existed.
    This represents the "optimal preparation" for knowing the demand in advance.
    
    Returns:
      x_i^*: optimal first-stage value given scenario xi_val (scalar between 0 and 5)
    """
    m = gp.Model("single_scenario")
    m.setParam("OutputFlag", 0)

    # First-stage decision variable: quantity to order
    x = m.addVar(lb=0, ub=5, name="x", vtype=GRB.CONTINUOUS)

    # Second-stage recourse variables: adjustments after demand is revealed
    v1 = m.addVar(lb=0, name="v1")  # Recourse variable 1
    v2 = m.addVar(lb=0, name="v2")  # Recourse variable 2
    v3 = m.addVar(lb=0, name="v3")  # Recourse variable 3
    v4 = m.addVar(lb=0, name="v4")  # Recourse variable 4

    # Add scenario-based constraints for specific demand xi_val:
    # These represent feasibility requirements that x must satisfy for this scenario
    m.addConstr(-v1 + v2 - v3 + v4 == xi_val + 0.5*x,  name="constr1")
    m.addConstr(-v1 + v2 + v3 - v4 == 1 + xi_val + 0.25*x, name="constr2")

    # Full objective: first-stage cost + second-stage recourse cost
    #  = -0.75*x + (-v1 + 3*v2 + v3 + v4)
    # Negative coefficients represent costs/penalties for unmet demand
    obj_expr = -0.75*x + (-v1 + 3*v2 + v3 + v4)
    m.setObjective(obj_expr, GRB.MINIMIZE)

    m.optimize()
    # Return the optimal x value (first-stage decision for this scenario)
    return x.X

def evaluate_cost(x_val, xi_val):
    """
    For Q2: Evaluate the total cost if first-stage decision x=x_val is made
    but scenario xi_val actually occurs.
    
    Formula: cost(x_val, xi_val) = -0.75*x_val + Q(x_val, xi_val)
    where Q(x_val, xi_val) is the optimal recourse cost (second-stage problem).
    
    Purpose: Calculate the true cost of mismatch between x decision and actual scenario.
    This is used to build the cost matrix V[i,j] for scenario clustering.
    
    Parameters:
      x_val: first-stage decision value (scalar, typically between 0-5)
      xi_val: actual demand scenario that occurs (scalar, from Poisson distribution)
    
    Returns:
      total_cost: scalar = first-stage cost + optimal recourse cost
    """
    m = gp.Model("evaluate")
    m.setParam("OutputFlag", 0)

    # Second-stage recourse variables for this specific scenario
    v1 = m.addVar(lb=0, name="v1")
    v2 = m.addVar(lb=0, name="v2")
    v3 = m.addVar(lb=0, name="v3")
    v4 = m.addVar(lb=0, name="v4")

    # Constraints determined by fixed x_val and actual scenario xi_val
    # These constraints are now deterministic (not subject to optimization)
    m.addConstr(-v1 + v2 - v3 + v4 == xi_val + 0.5*x_val,  name="constr1")
    m.addConstr(-v1 + v2 + v3 - v4 == 1 + xi_val + 0.25*x_val, name="constr2")

    # Second-stage objective: minimize recourse cost given x_val and xi_val
    m.setObjective(-v1 + 3*v2 + v3 + v4, GRB.MINIMIZE)
    m.optimize()

    # Extract second-stage cost (recourse cost)
    recourse_cost = m.objVal
    # Total cost = first-stage cost + second-stage recourse cost
    total_cost = -0.75*x_val + recourse_cost
    return total_cost

def build_cost_matrix(xi_array):
    """
    Build the NxN cost matrix V used in scenario clustering.
    
    V[i,j] = cost(x_i^*, xi_j) represents:
    - x_i^*: optimal first-stage decision if we trained on scenario i
    - xi_j: actual scenario that occurs
    
    The cost matrix captures the mismatch between training scenario and actual scenario.
    Large V[i,j] values indicate scenario i is a poor preparation for scenario j.
    
    Algorithm:
      Step 1: For i in [0..N-1], solve single-scenario subproblem for xi_i => x_i^*
              (what is the best first-stage decision if only scenario i existed?)
      Step 2: For each pair (i,j), evaluate cost if we chose x_i^* but faced scenario j
              V[i,j] = cost(x_i^*, xi_j)
    
    Parameters:
      xi_array: numpy array of N scenario values (demands from Poisson distribution)
    
    Returns:
      V: NxN numpy array where V[i,j] = total cost for scenario mismatch (i,j)
    """
    N = len(xi_array)
    x_star = np.zeros(N)  # Store optimal x for each scenario i

    # Step 1: Solve single-scenario problem for each scenario i to get x_i^*
    print("  Computing optimal x_i^* for each of N=%d scenarios..." % N)
    for i in range(N):
        x_star[i] = solve_single_scenario_subproblem(xi_array[i])

    # Step 2: Evaluate cost matrix V[i,j] = cost if trained on i but faced with j
    print("  Building %d x %d cost matrix V[i,j]..." % (N, N))
    V = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            V[i, j] = evaluate_cost(x_star[i], xi_array[j])
    
    return V

def solve_clustering_MIP(V, K):
    """
    Solve the scenario-clustering Mixed Integer Program.
    
    This MIP selects K representative scenarios from N total scenarios and assigns
    each scenario to its closest representative, minimizing expected cost discrepancy.
    
    Mathematical Formulation (from reference paper Eqs. (24)-(29)):
      
      Decision Variables:
        u_j ∈ {0,1}: indicator if scenario j is chosen as cluster representative
        x_{i,j} ∈ {0,1}: indicator if scenario i is assigned to cluster j
        t_j ≥ 0: cost for cluster j (absolute deviation penalty)
      
      Objective: minimize (1/N) * sum_j t_j
        - Minimizes expected cost of scenarios being assigned to "wrong" clusters
        - Dividing by N normalizes across cluster sizes
      
      Key Constraints:
        1) Flow assignment: each scenario i assigned to exactly one cluster j
        2) Cluster consistency: can only assign to j if u_j=1 (j is representative)
        3) Self-assignment: representative j must be assigned to itself (x_{j,j}=u_j)
        4) Cardinality: exactly K representatives chosen
        5) Absolute value linearization: t_j ≥ ±sum_i x_{i,j}*(V[j,i] - V[j,j])
    
    Purpose: Trade off having fewer scenarios (computational efficiency) against
    having scenarios that well-represent the full distribution (solution quality).
    
    Parameters:
      V: NxN cost matrix from build_cost_matrix()
      K: number of cluster representatives to select (integer, K < N)
    
    Returns:
      rep_scenarios: list of K scenario indices chosen as representatives
      assignment: length-N array where assignment[i] = index of representative for scenario i
    """
    N = V.shape[0]
    model = gp.Model("ScenarioClustering")
    model.setParam("OutputFlag", 0)

    # u_j: binary indicator if scenario j is chosen as cluster representative
    # If u_j = 1, then scenario j represents a cluster; if u_j = 0, it doesn't
    u = model.addVars(N, vtype=GRB.BINARY, name="u")

    # x_{i,j}: binary indicator if scenario i is assigned to representative cluster j
    # If x_{i,j} = 1, then scenario i's data is grouped with cluster representative j
    xij = model.addVars(N, N, vtype=GRB.BINARY, name="xij")

    # t_j: non-negative cost variable for cluster j
    # Represents absolute deviation: |sum_i x_{i,j}*(V[j,i] - V[j,j])|
    # Large t_j means scenarios assigned to j have high cost mismatch with j
    t = model.addVars(N, lb=0, name="t")

    # Objective: minimize average cluster cost across all representatives
    # (1/N) * sum_j t_j = expected cost of the reduced scenario set
    obj = gp.quicksum(t[j] for j in range(N)) / N
    model.setObjective(obj, GRB.MINIMIZE)

    # Linearize absolute value in t_j for each cluster j:
    # t_j >= |sum_i x_{i,j}*(V[j,i] - V[j,j])|
    # Decomposed into two linear constraints:
    #   t_j >=  (positive direction)
    #   t_j >= -(negative direction)
    for j in range(N):
        # Compute sum_i x_{i,j}*(V[j,i] - V[j,j]) = cost difference
        lhs = gp.quicksum(xij[i, j] * V[j, i] for i in range(N)) - \
              gp.quicksum(xij[i, j] * V[j, j] for i in range(N))
        # t_j must be >= both positive and negative parts of lhs
        model.addConstr(t[j] >= lhs,   name=f"abs_pos_{j}")
        model.addConstr(t[j] >= -lhs,  name=f"abs_neg_{j}")

    # Flow balance: each scenario i must be assigned to exactly one representative j
    # Constraint: sum_j x_{i,j} = 1 for all i
    # Ensures no scenario is unassigned or multiply-assigned
    for i in range(N):
        model.addConstr(gp.quicksum(xij[i, j] for j in range(N)) == 1,
                       name=f"assign_{i}")

    # Cluster consistency: can only assign scenario i to j if j is chosen as representative
    # Constraint: x_{i,j} <= u_j for all i, j
    # If u_j = 0 (j not a representative), then x_{i,j} must = 0 (can't assign to j)
    # Also, representative j must assign itself to itself: x_{j,j} = u_j
    for j in range(N):
        for i in range(N):
            model.addConstr(xij[i, j] <= u[j], name=f"cluster_{i}_{j}")
        # Special case: representative must be assigned to itself
        model.addConstr(xij[j, j] == u[j], name=f"self_assign_{j}")

    # Cardinality constraint: choose exactly K representatives
    # Constraint: sum_j u_j = K
    # Ensures we get exactly the requested number of cluster representatives
    model.addConstr(gp.quicksum(u[j] for j in range(N)) == K,
                   name="num_clusters")

    # Solve the MIP
    model.optimize()

    # Extract solution: which scenarios are representatives and which cluster each scenario belongs to
    rep_scenarios = []
    assignment = np.zeros(N, dtype=int)
    
    # Find which scenarios u_j were chosen as representatives (u_j = 1)
    for j in range(N):
        if u[j].X > 0.5:
            rep_scenarios.append(j)

    # For each scenario i, find which representative j it is assigned to (x_{i,j} = 1)
    for i in range(N):
        for j in range(N):
            if xij[i, j].X > 0.5:
                assignment[i] = j
                break

    return rep_scenarios, assignment

def solve_stochastic_program(xi_array, prob_array):
    """
    Solve the Q2 two-stage stochastic program in extensive form (all scenarios explicit).
    
    Formulation:
      min   -0.75*x + sum_m [ prob_array[m] * Q_m(x) ]
      s.t.  0 <= x <= 5
            second-stage constraints for each scenario m
    
    where Q_m(x) is the recourse cost for scenario m given first-stage decision x,
    and prob_array[m] is the probability of scenario m.
    
    Approach:
      - Unify first-stage variable x (same x for all scenarios)
      - Replicate second-stage variables and constraints for each scenario
      - Weight recourse costs by scenario probabilities
    
    This is the "extensive form" formulation: combines all scenarios into one large LP.
    
    Parameters:
      xi_array: array of M scenario values (demands)
      prob_array: array of M probabilities (must sum to 1)
    
    Returns:
      x_stoch: scalar, optimal first-stage decision x* that minimizes expected cost
    """
    M = len(xi_array)
    bigm = gp.Model("StochEF")
    bigm.setParam("OutputFlag", 0)

    # First-stage variable x (shared across all scenarios)
    # This is the decision we make "here-and-now" before uncertainty is revealed
    x = bigm.addVar(lb=0, ub=5, name="x")

    # Second-stage variables (scenario-specific)
    # For each scenario m, we have recourse variables v1_m, v2_m, v3_m, v4_m
    v1 = {}
    v2 = {}
    v3 = {}
    v4 = {}
    for m in range(M):
        v1[m] = bigm.addVar(lb=0, name=f"v1_{m}")  # Scenario m recourse var 1
        v2[m] = bigm.addVar(lb=0, name=f"v2_{m}")  # Scenario m recourse var 2
        v3[m] = bigm.addVar(lb=0, name=f"v3_{m}")  # Scenario m recourse var 3
        v4[m] = bigm.addVar(lb=0, name=f"v4_{m}")  # Scenario m recourse var 4

    # Add constraints for each scenario m
    for m in range(M):
        xi_val = xi_array[m]
        # Scenario m constraint 1: -v1_m + v2_m - v3_m + v4_m = xi_m + 0.5*x
        bigm.addConstr(-v1[m] + v2[m] - v3[m] + v4[m] == xi_val + 0.5*x,
                      name=f"constr1_{m}")
        # Scenario m constraint 2: -v1_m + v2_m + v3_m - v4_m = 1 + xi_m + 0.25*x
        bigm.addConstr(-v1[m] + v2[m] + v3[m] - v4[m] == 1 + xi_val + 0.25*x,
                      name=f"constr2_{m}")

    # Build objective: expected cost over all scenarios
    # First-stage cost (same for all scenarios): -0.75*x
    # Second-stage costs (weighted by probability): sum_m prob_m * (-v1_m + 3*v2_m + v3_m + v4_m)
    obj_expr = -0.75*x
    for m in range(M):
        # Each scenario m contributes prob_array[m] * (second-stage cost)
        obj_expr += prob_array[m] * (-v1[m] + 3*v2[m] + v3[m] + v4[m])
    bigm.setObjective(obj_expr, GRB.MINIMIZE)

    # Solve the stochastic program
    bigm.optimize()
    
    # Return optimal first-stage decision x*
    return x.X

def main():
    """
    Main script: execute scenario reduction workflow and compare solutions.
    
    Workflow:
      A) Generate N=100 Poisson(0.5) demand scenarios (full scenario set)
      B) Build NxN cost matrix from single-scenario solutions
      C) Solve clustering MIP to select K=10 representative scenarios
      D) Solve stochastic program with full 100 scenarios => x_full_sol
      E) Solve stochastic program with reduced 10 scenarios => x_sub_sol
      F) Evaluate out-of-sample performance on Mtest=10,000 independent test samples
      G) Compare costs to quantify value of scenario reduction
    """
    
    # ========== Step A: Generate full scenario set ==========
    print("="*70)
    print("SCENARIO REDUCTION: Q2 TWO-STAGE STOCHASTIC PROGRAM")
    print("="*70)
    
    np.random.seed(160325)  # Reproducibility: fixed random seed
    N = 100
    xi_full = np.random.poisson(lam=0.5, size=N)  # Generate N=100 scenarios from Poisson(0.5)
    p_full = np.ones(N)/N  # Equal probability for each scenario: 1/100

    print(f"\nStep A: Generated N={N} scenarios from Poisson(0.5)")
    print(f"  Scenario range: [{np.min(xi_full)}, {np.max(xi_full)}]")
    print(f"  Mean: {np.mean(xi_full):.3f}, Std: {np.std(xi_full):.3f}")

    # ========== Step B: Build cost matrix ==========
    print(f"\nStep B: Building NxN cost matrix V...")
    V = build_cost_matrix(xi_full)
    print(f"  Cost matrix V is {V.shape[0]} x {V.shape[1]}")
    print(f"  V range: [{np.min(V):.4f}, {np.max(V):.4f}]")

    # ========== Step C: Solve clustering MIP ==========
    K = 10
    print(f"\nStep C: Solving scenario-clustering MIP (K={K})...")
    rep_scenarios, assignment = solve_clustering_MIP(V, K)
    print(f"  Chosen representative scenario indices: {sorted(rep_scenarios)}")

    # ========== Step D: Solve with full scenario set ==========
    print(f"\nStep D: Solving Q2 stochastic program with full N={N} scenarios...")
    x_full_sol = solve_stochastic_program(xi_full, p_full)
    print(f"  Optimal first-stage decision: x* = {x_full_sol:.6f}")

    # ========== Step E: Build reduced scenario set and solve ==========
    print(f"\nStep E: Building reduced scenario set from {K} cluster representatives...")
    xi_sub = []
    prob_sub = []
    for j in rep_scenarios:
        # Count scenarios assigned to cluster representative j
        cluster_size = sum(1 for i in range(N) if assignment[i] == j)
        # Probability of cluster j = fraction of scenarios in cluster
        p_j = float(cluster_size) / N
        xi_sub.append(xi_full[j])
        prob_sub.append(p_j)
    xi_sub = np.array(xi_sub)
    prob_sub = np.array(prob_sub)

    print(f"  Reduced scenario set has K={len(xi_sub)} scenarios")
    print(f"  Scenario values: {xi_sub}")
    print(f"  Scenario probabilities: {prob_sub}")

    print(f"\nStep E (continued): Solving Q2 stochastic program with reduced set...")
    x_sub_sol = solve_stochastic_program(xi_sub, prob_sub)
    print(f"  Optimal first-stage decision: x* = {x_sub_sol:.6f}")

    # ========== Step F: Out-of-sample evaluation ==========
    print(f"\nStep F: Out-of-sample evaluation on Mtest={10_000:,} fresh samples...")
    Mtest = 10_000
    np.random.seed(99999)  # Different seed for test set
    xi_test = np.random.poisson(0.5, size=Mtest)

    # Evaluate x_full_sol (from full scenario set) on test set
    print(f"  Evaluating full-scenario solution x_full = {x_full_sol:.6f}...")
    total_full = 0.0
    for s in range(Mtest):
        total_full += evaluate_cost(x_full_sol, xi_test[s])
    total_full /= Mtest

    # Evaluate x_sub_sol (from reduced scenario set) on test set
    print(f"  Evaluating reduced-scenario solution x_sub = {x_sub_sol:.6f}...")
    total_sub = 0.0
    for s in range(Mtest):
        total_sub += evaluate_cost(x_sub_sol, xi_test[s])
    total_sub /= Mtest

    # ========== Step G: Compare results ==========
    print("\n" + "="*70)
    print("OUT-OF-SAMPLE PERFORMANCE COMPARISON")
    print("="*70)
    print(f"Full-scenario solution (N={N}):")
    print(f"  First-stage decision: x = {x_full_sol:.6f}")
    print(f"  Average cost over {Mtest:,} test samples: {total_full:.6f}")
    print()
    print(f"Reduced-scenario solution (K={K}):")
    print(f"  First-stage decision: x = {x_sub_sol:.6f}")
    print(f"  Average cost over {Mtest:,} test samples: {total_sub:.6f}")
    print()
    gap_est = total_sub - total_full
    gap_pct = 100 * (gap_est / abs(total_full)) if total_full != 0 else 0
    print(f"Cost difference (gap): {gap_est:+.6f} ({gap_pct:+.2f}%)")
    if gap_est > 0:
        print(f"  => Reduced scenario set costs {abs(gap_est):.6f} more than full set")
    else:
        print(f"  => Reduced scenario set is {abs(gap_est):.6f} cheaper than full set (lucky!)")
    print("="*70)

if __name__ == "__main__":
    main()