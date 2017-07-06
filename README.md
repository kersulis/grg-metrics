# grg-metrics
System-wide metrics for [GRG-format][2] networks.

## Computing metrics
To compute metrics for every GRG data file in a directory:

```python
metrics = grg_metrics.compute_metrics(dir_path)
```

Output `metrics` is a Pandas DataFrame containing metric data.

The research behind this code is based on the [NESTA archive][3] networks. We chose a representative subset of networks to keep redundancy to a minimum (i.e. including only a sample of the many Polish grid network files). Provided you have a translation of NESTA into GRG format, you can compute metrics on this subset with:

```python
v11_rep_names = grg_metrics.nesta_v11_representative()
v11_rep_files = ['../NESTA_GRGv1.1/%s.json' % s for s in v11_rep_names]
metrics = grg_metrics.compute_metrics(v11_rep_files)
```

The above code assumes the NESTA directory is in the directory above `grg_metrics`. It also assumes (for now, at least) that a 403-bus RTE network is included in the NESTA directory. If you do not have this network, you must remove `case_403_rte` from the list returned by `grg_metrics.nesta_v11_representative()` after the first line above.

## Exploring metrics
For high-quality output display, we recommend exploring metrics in the [Jupyter notebook][1] environment. Here are a few examples of navigating a `metrics` DataFrame:

Sort networks by number of edges, in descending order:

```python
metrics.edges.sort_values(ascending=False)
```

Compare degree assortativity coefficients for IEEE 118- and 300-bus cases (available from the NESTA archive):

```python
metrics.degree_assortativity.loc[['nesta_case118_ieee', 'nesta_case300_ieee']]
```

Describe distribution of mean degree values for large and medium networks:

```python
metrics.query("size == 'large' | size == 'medium'").mean_degree.describe()
```

Note:
- "large" networks have >5k nodes
- "medium" is 1k - 5k
- "small" is 20 - 1k
- "tiny" is anything smaller

## Analyzing metrics
Part of the research project that led to this package was development of sensible warning and error thresholds for our metrics. A power grid bus with twenty connections, for example, should be flagged as unrealistic. To run our checks on a metrics DataFrame:

```python
msg = grg_metrics.analyze_metrics(metrics)
```

The input `metrics` is the output of `grg_metrics.compute_metrics`, and `msg` is a DataFrame table of warnings and errors. To replace descriptive warning and error messages with "Warning" and "Error" respectively, add `describe=False` when calling `analyze_metrics`.

## Testing
Run `pytest test.py` in the `test` subdirectory.

[1]: http://jupyter.org/
[2]: https://gdg.engin.umich.edu/release-v1-0/
[3]: https://arxiv.org/abs/1411.0359
