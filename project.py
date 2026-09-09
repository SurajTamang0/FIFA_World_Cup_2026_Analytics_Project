"""
Main entry point for the FIFA World Cup 2026 Analytics Project.

This script runs the four completed analytical tasks in sequence:

1. Super-Sub Effect
2. Shooting Volume vs Efficiency
3. Defensive Pressure vs Discipline
4. Playing Time vs On/Off Impact

Each task performs its own:
- data preparation
- sampling
- descriptive statistics
- confidence intervals
- Welch two-sample t-test
- assumption diagnostics
- visualisation
- output generation
"""

from src.tasks.task1_super_sub_effect import main as run_task1
from src.tasks.task2_shooting_efficiency import main as run_task2
from src.tasks.task3_defensive_pressure import main as run_task3
from src.tasks.task4_onoff_impact import main as run_task4


def main():
    """
    Run all four analytical tasks sequentially.
    """

    print("=" * 75)
    print("FIFA WORLD CUP 2026 ANALYTICS PROJECT")
    print("Running all four analytical tasks")
    print("=" * 75)

    print("\nStarting Task 1...\n")
    run_task1()

    print("\nStarting Task 2...\n")
    run_task2()

    print("\nStarting Task 3...\n")
    run_task3()

    print("\nStarting Task 4...\n")
    run_task4()

    print("\n" + "=" * 75)
    print("ALL FOUR ANALYTICAL TASKS COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print("\nGenerated analytical samples:")
    print("  data/processed/task1_super_sub_sample.csv")
    print("  data/processed/task2_shooting_efficiency_sample.csv")
    print("  data/processed/task3_defensive_pressure_sample.csv")
    print("  data/processed/task4_onoff_impact_sample.csv")

    print("\nGenerated figures:")
    print("  figures/task1_super_sub_effect.png")
    print("  figures/task2_shooting_efficiency.png")
    print("  figures/task3_defensive_pressure.png")
    print("  figures/task4_onoff_impact.png")


if __name__ == "__main__":
    main()