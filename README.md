# Additive Modeling and Dynamic Programming for F1 Pit-Stop Optimization

This project leverages raw telemetry data via the `FastF1` library to build a prescriptive race strategy engine. By combining additive predictive models accounting for polynomial tire degradation, fuel mass interactions, and logarithmic track evolution with a Dynamic Programming solver, it mathematically calculates the possible optimal pit-stop sequence to minimize overall race time.

## Core Concepts

To accurately simulate the non-linear environment of a Formula 1 race, this project relies on a pipeline of advanced statistical methods before any optimization occurs.

### 1. Robust Anomaly Detection (IQR & Z-Scores)
Raw lap times are incredibly noisy. Filtering out Safety Cars (SC) and Pit Laps is insufficient, as driver mistakes, lock-ups, and traffic (DRS trains) heavily skew degradation models. We apply Interquartile Range (IQR) and Z-Score ($\sigma > 2$) filtering to each driver's continuous stint.
* **Purpose:** This mathematically identifies and drops hidden outliers, ensuring our regression models fit the driver's *true* clean-air pace rather than their traffic-affected pace.

### 2. Compound Profiling (ANOVA & Mixed-Effects)
To determine the base pace difference between Soft, Medium, and Hard tires, we cannot rely on raw averages, as Free Practice lap times are contaminated by varying fuel loads. We use Free Practice 2 (FP2) long-run data and apply **Mixed-Effects Models** (or ANOVA).
* **Purpose:** This mathematically isolates the "Compound Delta" from the fuel load, proving statistically significant time differences between tires in a controlled, independent environment.

### 3. The Additive Pace Equation
The core predictive engine of this project is an additive model that forecasts expected lap times based on three heavily engineered features:

Expected_Lap = Base_Pace + f(Tire_Age, Fuel_Mass) - h(Track_Evolution)

#### A. Dynamic Tire Degradation & Fuel Interaction $f(x, y)$
Tires do not degrade linearly; they fall off a "cliff." Furthermore, a heavy car (Lap 1) destroys tires exponentially faster than a light car (Lap 50). We use **Ordinary Least Squares (OLS)** wrapped in **Polynomial Regression** (At Degree 2 to prevent mathematical overfitting and hallucinated lap times).
* **Interaction Term:** We multiply Tire Age by Fuel Mass. This dynamically steepens the degradation curve based on the physical weight of the car at that exact point in the race.

#### B. Logarithmic Track Evolution $h(x)$
As cars lay down rubber, the track gets faster, but this improvement experiences diminishing returns. We fit a logarithmic curve $y = a \ln(x) + b$ to map the track improvement.
* **Implementation Note:** For dry Sunday race data, the track is usually fully "rubbered in," meaning coefficient $a$ approaches zero. This variable remains in the pipeline to ensure the algorithm is robust enough to handle "green" tracks (e.g., following morning rain).

### 4. Strategy Optimization (Dynamic Programming)
Once the expected lap time matrix is generated, we must navigate it to find the fastest possible race time. Some may say, "why not use a greedy algorithm?" A greedy algorithm would fail here, as taking a short-term loss (a 24-second pit stop) is required for a long-term gain (fresh tires).
* **Implementation:** The algorithm breaks the race down into recursive sub-problems, calculating the cost of "pitting" vs. "staying out" at every single lap. It utilizes **Memoization** to store the time costs of previously calculated states (Current Lap, Current Compound, Tire Age), efficiently mapping millions of potential strategy trees to output the absolute global minimum race time.
