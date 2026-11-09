# Figure 2

Run `python Code/06_make_figures.py` from the repository root to make all three
figures from the completed analysis.

To render only this figure:

```bash
python Code/Figure2/plot_figure2.py --resources work/november191/resources.json --products work/november191/figure_products --output-dir work/november191/figures
```

The script reads completed products and writes PNG/PDF. It does not refit the
observations. See [the main workflow](../README.md).
