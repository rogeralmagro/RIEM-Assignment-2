"""
Main script for RIEM Assignment 2.

This file runs the main workflow for the assignment.
Step 1 and Step 2 are kept in separate modules to keep the code simple.
"""

from src.step1 import run_step1
from src.step2 import run_step2


def main():
    """Run the assignment workflow."""

    print("RIEM Assignment 2")
    print("-----------------")

    # Step 1: day-ahead and balancing market participation
    # Uncomment when Step 1 is implemented.
    # run_step1()

    # Step 2: ancillary service market participation
    run_step2()


if __name__ == "__main__":
    main()