"""``python -m growth_review`` -- print the dataset and theory-model registries."""
from .datasets import summary_table
from .theory import summary_table as theory_summary_table

if __name__ == "__main__":
    print(summary_table())
    print("\n" + "=" * 88 + "\ntheory models (growth_review.theory)\n" + "=" * 88)
    print(theory_summary_table())
