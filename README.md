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

### Phase 5: Optimization Algorithm Development (Pending)
Once the expected lap time matrix is generated, the algorithm must navigate it to find the fastest possible race time. A greedy algorithm fails here because it cannot evaluate short-term losses (a 24-second pit stop) for long-term gains (fresh tires).
- Goal: Develop a Dynamic Programming (DP) solver breaking the race into recursive sub-problems, calculating the cost of "pitting" vs. "staying out" at every lap.
- Constraints: Total race distance, fixed pit lane time loss, and the mandatory two-compound sporting regulation.
- Optimization: Utilize Memoization to store the time costs of previously calculated states (Lap, Compound, Age), mapping millions of potential strategy trees to output the global minimum race time efficiently.

### Phase 6: Evaluation and Reporting (Pending)
- Compare the deterministic DP model's optimal strategy against the actual winning strategy to calculate theoretical time gains.
- Generate a line graph showing the cumulative race time of the optimal strategy versus conventional 1-stop or 3-stop alternatives.
- Discuss limitations, specifically stochastic elements such as traffic, unpredicted weather, and inconsistent driver pace that the deterministic DP model cannot foresee.
