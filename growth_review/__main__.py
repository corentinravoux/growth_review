"""``python -m growth_review`` -- print the dataset registry."""
from .datasets import summary_table

if __name__ == "__main__":
    print(summary_table())
