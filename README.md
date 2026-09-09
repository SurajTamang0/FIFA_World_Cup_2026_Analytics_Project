# FIFA World Cup 2026 Analytics Project

## Project Overview

This project applies statistical analysis techniques to FIFA World Cup 2026 player data obtained from FBref. The objective is to investigate several questions relating to player usage, attacking performance, defensive activity, disciplinary outcomes, and playing-time impact.

The project uses Python for data processing, sampling, statistical analysis, confidence interval estimation, hypothesis testing, and visualisation.

Four analytical questions are investigated:

1. Do substitute-dominant outfield players produce a different rate of attacking contribution per 90 minutes than starter-dominant players?
2. Among attacking players with meaningful tournament exposure, do high-volume shooters have a different scoring efficiency than lower-volume shooters?
3. Among defensively involved outfield players, do players with higher defensive-action rates accumulate a different disciplinary-card rate than players with lower defensive-action rates?
4. Among outfield players with meaningful tournament exposure, do players with a higher share of available team minutes have a different on/off goal-difference impact than players with a lower share of available team minutes?

---

## Data Source

The datasets used in this project were obtained from **FBref**, a football statistics database provided by Sports Reference.

**Source:** [FBref - 2026 FIFA World Cup Statistics](https://fbref.com/en/comps/1/2026/2026-World-Cup-Stats)

The player-level statistical tables used for the analysis were extracted from the 2026 FIFA World Cup section of FBref. Five categories of player statistics were used:

- Standard player statistics
- Goalkeeping statistics
- Shooting statistics
- Playing-time statistics
- Miscellaneous statistics

The extracted datasets were saved locally as CSV files in:

`data/raw/`

The original raw datasets were retained separately from the processed analytical samples to preserve the source data and support reproducibility.

**Website:** [FBref](https://fbref.com/en/)

## Project Structure

```text
FIFA_World_Cup_2026_Analytics_Project/
|
|-- data/
|   |-- raw/
|   |   |-- fbref_2026_world_cup_player_standard.csv
|   |   |-- fbref_2026_world_cup_player_shooting.csv
|   |   |-- fbref_2026_world_cup_player_playing_time.csv
|   |   |-- fbref_2026_world_cup_player_miscellaneous.csv
|   |   `-- fbref_2026_world_cup_player_goalkeeping.csv
|   |
|   `-- processed/
|       |-- task1_super_sub_sample.csv
|       |-- task2_shooting_efficiency_sample.csv
|       |-- task3_defensive_pressure_sample.csv
|       `-- task4_onoff_impact_sample.csv
|
|-- figures/
|   |-- task1_super_sub_effect.png
|   |-- task2_shooting_efficiency.png
|   |-- task3_defensive_pressure.png
|   `-- task4_onoff_impact.png
|
|-- src/
|   |-- tasks/
|   |   |-- task1_super_sub_effect.py
|   |   |-- task2_shooting_efficiency.py
|   |   |-- task3_defensive_pressure.py
|   |   `-- task4_onoff_impact.py
|   |
|   |-- data_acquisition.py
|   |-- data_validation.py
|   |-- question_feasibility.py
|   `-- statistical_utils.py
|
|-- project.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

---

## Analytical Method

### Data Preparation

The raw FBref datasets are loaded and merged using player-level identifiers. Appropriate eligibility criteria are then applied for each research question.

Goalkeepers are excluded where the analysis concerns outfield players. Minimum playing-time requirements are also applied to reduce the influence of players with very limited tournament exposure.

Missing values in variables required for each statistical comparison are checked before sampling and hypothesis testing.

### Group Construction

Each research question compares two player groups.

Task 1 compares:

- Starter-dominant players
- Substitute-dominant players

Task 2 compares:

- Higher-volume shooters
- Lower-volume shooters

Task 3 compares:

- Higher defensive-activity players
- Lower defensive-activity players

Task 4 compares:

- Higher-minute-share players
- Lower-minute-share players

Median-based splits are used where continuous variables need to be converted into transparent comparison groups.

---

## Sampling Strategy

A stratified simple random sampling approach is used.

For each analytical task:

- 64 observations are sampled from each comparison group.
- The total analytical sample size is 128 players.
- A fixed random seed of `2026` is used.

Using a fixed random seed makes the sampling process reproducible.

The resulting analytical samples are saved in:

```text
data/processed/
```

---

## Statistical Analysis

The following procedures are performed for each research question:

- Descriptive statistics
- Group means and medians
- Standard deviations
- 95% confidence intervals
- Difference in group means
- 95% confidence interval for the difference
- Welch two-sample t-test
- Outlier diagnostics using the interquartile range
- Skewness assessment
- Statistical interpretation
- Visualisation

The significance level used for hypothesis testing is:

```text
alpha = 0.05
```

Welch's independent two-sample t-test is used because it does not require the two population variances to be equal.

---

## Task 1 - Super-Sub Effect

### Research Question

Do substitute-dominant outfield players produce a different rate of attacking contribution per 90 minutes than starter-dominant players?

The response variable is attacking contribution per 90 minutes, calculated using goals and assists.

### Sample Results

Starter-dominant players:

```text
Mean G+A per 90 = 0.262
95% CI = [0.174, 0.350]
```

Substitute-dominant players:

```text
Mean G+A per 90 = 0.337
95% CI = [0.192, 0.481]
```

Difference:

```text
Substitute - Starter = 0.074
95% CI = [-0.093, 0.242]
```

Welch t-test:

```text
t = 0.880
p = 0.3809
```

At the 5% significance level, the null hypothesis is not rejected. The sample does not provide sufficient evidence that mean attacking contribution per 90 differs between substitute-dominant and starter-dominant players.

---

## Task 2 - Shooting Volume and Scoring Efficiency

### Research Question

Among attacking players with meaningful tournament exposure, do high-volume shooters have a different scoring efficiency than lower-volume shooters?

Scoring efficiency is measured using goals per shot.

### Sample Results

Higher-volume shooters:

```text
Mean goals per shot = 0.074
95% CI = [0.042, 0.106]
```

Lower-volume shooters:

```text
Mean goals per shot = 0.118
95% CI = [0.057, 0.179]
```

Difference:

```text
Higher-volume - Lower-volume = -0.043
95% CI = [-0.112, 0.025]
```

Welch t-test:

```text
t = -1.257
p = 0.2120
```

The null hypothesis is not rejected at the 5% significance level. The sample therefore does not provide sufficient evidence of a statistically significant difference in mean scoring efficiency between higher-volume and lower-volume shooters.

---

## Task 3 - Defensive Pressure and Discipline

### Research Question

Among defensively involved outfield players, do players with higher defensive-action rates accumulate a different disciplinary-card rate than players with lower defensive-action rates?

Defensive activity is represented using tackles won and interceptions per 90 minutes. Discipline is represented using card rate per 90 minutes.

### Sample Results

Higher defensive-activity players:

```text
Mean card rate per 90 = 0.138
95% CI = [0.087, 0.190]
```

Lower defensive-activity players:

```text
Mean card rate per 90 = 0.132
95% CI = [0.069, 0.194]
```

Difference:

```text
Higher activity - Lower activity = 0.007
95% CI = [-0.073, 0.087]
```

Welch t-test:

```text
t = 0.164
p = 0.8703
```

The null hypothesis is not rejected. There is insufficient statistical evidence in the analytical sample to conclude that mean disciplinary-card rates differ between players with higher and lower defensive activity.

---

## Task 4 - Playing Time and On/Off Impact

### Research Question

Among outfield players with meaningful tournament exposure, do players with a higher share of available team minutes have a different on/off goal-difference impact than players with a lower share of available team minutes?

### Sample Results

Higher-minute-share players:

```text
Mean On-Off = 0.817
95% CI = [0.024, 1.610]
```

Lower-minute-share players:

```text
Mean On-Off = 0.116
95% CI = [-0.320, 0.551]
```

Difference:

```text
Higher minute share - Lower minute share = 0.701
95% CI = [-0.197, 1.600]
```

Welch t-test:

```text
t = 1.548
p = 0.1248
```

The null hypothesis is not rejected at the 5% significance level. Although the sampled higher-minute-share players have a larger mean On-Off value, the evidence is insufficient to establish a statistically significant difference between the two playing-time groups.

---

## Overall Findings

Across all four analyses, none of the Welch two-sample t-tests produced a p-value below the selected significance level of 0.05.

The results therefore do not provide sufficient evidence to reject the null hypothesis for any of the four research questions.

This does not demonstrate that the groups are identical. Instead, it indicates that the analytical samples do not provide sufficiently strong statistical evidence to establish differences in the population means under the methods used in this project.

The confidence intervals and descriptive statistics provide additional information about the magnitude and uncertainty of the observed differences.

---

## Limitations

Several limitations should be considered when interpreting the results.

Player observations are associated with national teams and are therefore not necessarily completely independent of team context.

Tournament data also represents a relatively short observation period. Player statistics can consequently be affected by limited minutes, opponent strength, tactical roles, match state, substitution timing, and other contextual factors.

Median splits provide a transparent method for constructing comparison groups but simplify continuous player characteristics into two categories.

The analyses are observational. Statistical associations should therefore not be interpreted as evidence of causal relationships.

Some response variables are also right-skewed and contain potential outliers. Welch's t-test provides some robustness to unequal variances, but distributional characteristics should still be considered when interpreting the results.

---

## Reproducibility

The project uses a fixed random seed:

```text
2026
```

This allows the same analytical samples and statistical results to be reproduced when the analysis is rerun using the same source datasets and software environment.

---

## Running the Project

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the complete analysis

From the project root directory:

```bash
python project.py
```

This executes all four analytical tasks sequentially.

Individual tasks can also be run separately:

```bash
python src/tasks/task1_super_sub_effect.py
python src/tasks/task2_shooting_efficiency.py
python src/tasks/task3_defensive_pressure.py
python src/tasks/task4_onoff_impact.py
```

---

## Generated Outputs

Running the complete project generates four analytical sample datasets:

```text
data/processed/task1_super_sub_sample.csv
data/processed/task2_shooting_efficiency_sample.csv
data/processed/task3_defensive_pressure_sample.csv
data/processed/task4_onoff_impact_sample.csv
```

It also generates four figures:

```text
figures/task1_super_sub_effect.png
figures/task2_shooting_efficiency.png
figures/task3_defensive_pressure.png
figures/task4_onoff_impact.png
```

---

## Technologies Used

- Python
- pandas
- NumPy
- SciPy
- Matplotlib
- Git
- GitHub
- Visual Studio Code

---

## Conclusion

This project demonstrates a reproducible statistical workflow for analysing FIFA World Cup 2026 player data.

The analysis integrates data preparation, eligibility filtering, stratified random sampling, descriptive statistics, confidence interval estimation, Welch two-sample hypothesis testing, diagnostic assessment, and data visualisation.

While none of the four comparisons produced statistically significant evidence at the 5% level, the results illustrate how statistical inference can be used to evaluate football performance questions while accounting for sampling uncertainty and acknowledging the limitations of observational tournament data.