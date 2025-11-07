# Examples

This directory contains example scripts demonstrating how to use various functions in the jdb_to_nwb package.

## plot_probability_matching_example.py

Demonstrates how to use the `plot_probability_matching` function to visualize within-session probability matching behavior.

### Usage

```bash
cd /path/to/jdb_to_nwb
python examples/plot_probability_matching_example.py
```

### What it does

The script:
1. Loads trial and block data from arduino text files
2. Parses the data using `parse_arduino_text`
3. Creates a probability matching plot showing:
   - Rolling average of port choice frequencies
   - Reward delivery events at each port
   - Block transitions and reward probabilities

### Output

The plot is saved to `./output/probability_matching.png` and shows:
- **Top 3 subplots**: Bars indicating when rewards were delivered at ports A (blue), B (orange), and C (green)
- **Main plot**: Rolling averages of port visit frequencies over trials, with vertical lines indicating block changes
- **Text labels**: Reward probabilities (pA:pB:pC) for each block
