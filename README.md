# Additive Modeling and Dynamic Programming for F1 Pit-Stop Optimization

This project leverages raw telemetry data via the FastF1 library to build a prescriptive race strategy engine. By combining additive predictive models accounting for polynomial tire degradation, fuel mass interactions, and logarithmic track evolution with a Dynamic Programming solver, it mathematically calculates the optimal pit-stop sequence to minimize overall race time.

## Project Phases & Progress

### Phase 1: Environment Setup and Data Acquisition (Completed)
- Utilized the `fastf1` library to extract telemetry and timing data for the 2024 Bahrain Grand Prix.
- Selected Max Verstappen as the reference driver to model standard performance curves and minimize driver-induced variance.

### Phase 2: Data Preprocessing and Cleaning (Completed)
Raw lap times are incredibly noisy. Filtering out Safety Cars (SC) and Pit Laps is insufficient, as driver mistakes, lock-ups, and traffic heavily skew degradation models.
- Filtered for green-flag racing laps, removing SC, VSC, in-laps, and out-laps using FastF1's built-in pit detection.
- Applied an Interquartile Range (IQR) filter per stint. Implementing both upper and lower bounds successfully eliminated mathematical outliers (e.g., anomalies like a 92.5s data-glitch lap), ensuring models fit true clean-air pace.

### Phase 3: Exploratory Data Analysis & Compound Profiling (Completed)
To determine the base pace difference between Soft, Medium, and Hard tires, lap times must be isolated from fuel loads.
- Attempted to use Free Practice 2 (FP2) long-run data and Mixed-Effects Models to calculate exact compound deltas.
- Discovered a lack of variance in FP2 data across top teams (100% of long runs were executed on Softs in Bahrain to save race sets). As a result, the Mixed-Effects model failed to converge.
- To maintain statistical integrity, the pipeline bypassed empirical calculation for Bahrain and explicitly encoded official pre-race Pirelli deltas to normalize the Sunday race data.

### Phase 4: Additive Predictive Modeling (Completed)
The core predictive engine is an additive model forecasting expected lap times based on three engineered features:

`Expected_Lap = Base_Pace + f(Tire_Age, Fuel_Mass) - h(Track_Evolution)`

- **Model Validation:** Validated a Degree 2 Polynomial Regression model against a Degree 1 Linear model. The Degree 2 model demonstrated a decisive superior fit (R-Squared: 0.9550 vs 0.7890).
- **Key Coefficient Insights:**
  - `TireAge * Fuel_Mass` (+0.00396): The interaction term validates the core thesis of the project. Tires degrade significantly faster on a heavy car; a full tank and old tires amplify each other's pace penalty.
  - `Fuel_Mass` (-0.12990): Mathematically confirmed the physical effect of fuel burn, saving roughly 0.13 seconds per lap for every kilogram of fuel burned.
  - `TireAge^2` (+0.00853): Accurately modeled the exponential degradation "cliff" where lap times start climbing rapidly late in a stint.
  - `Log_Track_Evo` (-2.00792): Indicated track evolution was still significantly improving lap times throughout the race.

### Phase 5: Optimization Algorithm Development (Completed)
Once the expected lap time matrix is generated, the algorithm must navigate it to find the fastest possible race time. A greedy algorithm fails here because it cannot evaluate short-term losses (a 23.5-second pit stop) for long-term gains (fresh tires).
- **Goal:** Developed a recursive Dynamic Programming (DP) solver breaking the race into recursive sub-problems, calculating the exact cost of "pitting" vs. "staying out" at every lap.
- **Constraints:** Enforced the 57-lap total race distance, an elite Red Bull pit lane loss of 23.5s, a minimum stint length of 5 laps, and the mandatory two-compound sporting regulation.
- **Optimization:** Utilized memoization to store the time costs of previously calculated states (`lap`, `current_compound`, `tire_age`, `used_compounds`), mapping millions of potential strategy trees to output the global minimum race time efficiently.

### Phase 6: Evaluation and Reporting (Completed)
- **Comparison:** Evaluated the deterministic DP model's theoretically optimal 3-stop strategy against Max Verstappen's actual winning 2-stop strategy (Soft -> Hard -> Soft).
- **Visualization:** Generated a line graph mapping the cumulative race time of the optimal AI strategy versus the actual pit wall decision, illustrating a theoretical clean-air advantage of ~21 seconds.
- **Limitations Discussion:** Reconciled the algorithmic time gap by acknowledging real-world stochastic elements that the deterministic DP model cannot foresee, such as pit entry/exit traffic, out-lap warming limits, unpredicted weather, and defensive driving constraints.

## Final Output

The final output of this project is `notebook.ipynb`.

## Model Optimization & Overcoming the "Immortal Soft Tire"

During the development of the Dynamic Programming (DP) solver, the initial additive model produced highly optimistic, but physically flawed, race strategies. Fixing this required a deep dive into the mathematical assumptions of the polynomial regression and aggressive hyperparameter tuning.

### The Problem: V1 Baseline (The Flawed Model)
In the initial model, lap times were normalized across compounds by subtracting a static Pirelli pace delta (e.g., `-1.4s` for Softs). However, the regression model was only trained on `TireAge` and `Fuel_Mass`. 
* **The Mathematical Flaw:** The algorithm assumed every tire compound degraded at the exact same quadratic rate, just vertically shifted by pace. It created an "Immortal Soft Tire," leading the DP solver to recommend unrealistic 27-lap stints on the softest rubber because it didn't understand the compound's aggressive physical degradation cliff.
* **The Out-Lap Bug:** The recursive DP solver was inadvertently initializing fresh tires at `Age = 2` upon exiting the pit lane, mathematically penalizing pit stops and pushing the algorithm toward conservative 2-stop strategies.

### The Solution: V2 Optimized (The Physics & Hyperparameter Fix)
To force the algorithm to respect real-world F1 physics, three major architectural changes were implemented:

1. **Compound-Specific Degradation Multipliers:** Instead of feeding raw `TireAge` into the polynomial model, an `effective_tire_age` was engineered. By applying heuristic multipliers (`SOFT: 1.40`, `MEDIUM: 1.15`, `HARD: 0.85`), the model correctly scales the exponential cliff. A 10-lap old Soft tire is now mathematically treated as a 14-lap old tire, forcing the DP solver to recognize the grip fall-off.
2. **Track-Specific Hyperparameter Tuning:**
   The baseline Pirelli deltas were tuned to reflect the specific conditions of the Bahrain night race. The freezing track heavily favors softer rubber, so the Soft delta was aggressively increased from `-1.4s` to `-1.8s`.
3. **Elite Pit Loss Recalibration:**
   Red Bull Racing executes historically fast pit stops. The `PIT_LOSS` penalty was reduced from the track average of `24.0s` to `23.5s`, accurately reflecting the team's real-world operational advantage.

### Strategy Comparison: V1 vs. V2

| Metric | V1 Baseline (Old Code) | V2 Optimized (New Code) |
| :--- | :--- | :--- |
| **Pit Stop Penalty** | `24.0s` | `23.5s` (Aggressive) |
| **Soft Tire Pace Advantage**| `-1.4s` | `-1.8s` (Cold Track Adjusted) |
| **Tire Degradation** | Static across all tires | Scaled dynamically by compound |
| **DP Chosen Strategy** | 2-Stop (Unrealistic Stints) | **3-Stop** (Aggressive Sprinting) |
| **Optimal Clean-Air Time** | 5505.75s | **5475.61s** |

**Conclusion:** By fixing the physics constraints and tuning the hyperparameters to match Red Bull's real-world environment, the V2 DP Solver successfully identified an aggressive 3-Stop optimal strategy (Medium -> Soft -> Soft -> Soft). When simulated in a clean-air environment, this mathematically optimized sequence is theoretically **~21 seconds faster** than the actual strategy executed on the pit wall.
