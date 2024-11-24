# Rule Based Script Executor

## Description

This script allows you to execute either the `forward_chaining.py` or `backward_chaining.py` script based on user input. The script uses `argparse` for command-line argument parsing and `subprocess` for executing the chosen script.

## Usage

To use the script, you need to provide either the `--forward` or `--backward` argument when running `rule_based.py`.

### Example Commands

- To execute `forward_chaining.py`: python rule_based.py --forward
- To execute `backward_chaining.py`: python rule_based.py --backward


## Requirements

- Python 3.x
- `forward_chaining.py` and `backward_chaining.py` scripts should be present in the same directory as `rule_based.py`.
- following libraries:
	- pandas
	- PyYAML
	- scikit-learn
	- matplotlib
	- seaborn
	- openpyxl
pip install ..."# Artificial-Intelligence-24-25" 
