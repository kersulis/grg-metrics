# grg-metrics
System-wide metrics for [GRG-format][2] networks.

## Installation
1. Download this package and add its location to the Python path:
```
import sys
sys.path.append('/path/to/grg-metrics')
```
2. Install the packages listed in `requirements.txt` using pip or conda

## Computing metrics
To compute metrics for every GRG data file in a directory:

```python
import grg_metrics
metrics = grg_metrics.compute_metrics(dir_path)
```

The output, `metrics`, is a Pandas DataFrame containing metric data.

The [research behind this code][pscc] is based on the [NESTA archive][3] networks. We chose a representative subset of networks to keep redundancy to a minimum (i.e. including only a sample of the many Polish grid network files). Once you have NESTA in GRG bus-branch format, you can compute metrics on this subset with:

```python
v11_rep_names = grg_metrics.nesta_v11_representative()
v11_rep_files = ['../NESTA_GRGv1.1/%s.json' % s for s in v11_rep_names]
metrics = grg_metrics.compute_metrics(v11_rep_files)
```

The above code assumes the NESTA directory is in the directory above `grg_metrics`. It also assumes (for now, at least) that a 403-bus RTE network is included in the NESTA directory. If you do not have this network, you must remove `case_403_rte` from the list returned by `grg_metrics.nesta_v11_representative()` after the first line above.

## Exploring metrics
We recommend exploring metrics in the [Jupyter notebook][1] environment. Here are a few examples of navigating a `metrics` DataFrame:

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
The research that led to this package involved the development of sensible warning and error thresholds for our metrics. A substation with twenty connections, for example, should be flagged as unrealistic. To run our checks on a metrics DataFrame:

```python
msg = grg_metrics.analyze_metrics(metrics)
```

The input `metrics` is the DataFrame that comes from running `grg_metrics.compute_metrics`, and `msg` is a corresponding DataFrame table of warnings and errors. To replace descriptive warning and error messages with "Warning" and "Error" respectively, add `describe=False` when calling `analyze_metrics`.

## Optional metrics
The following metrics may be computed by passing the indicated keyword argument. They are not computed by default because there are no corresponding thesholds, but they do contain interesting information.
* [Average shortest path length][shortest]: `compute_average_shortest_path_length=True`
* [Fiedler value][fiedler]: `compute_fiedler_value=True`
* [Spectral radius][spectral] of adjacency matrix: `compute_adj_spectral_radius=True`
* [Maximal cliques][mc]: `compute_maximal_cliques=True`

## Extended branch
We considered many more metrics than ultimately made it into the final set. These tend to be more computationally demanding and difficult to interpret intuitively. The code for computing this metrics is available in this package, but you need to check out the `extended` branch. With this branch checked out, see `sdp.py` and `weighted.py`.

Note: the extended branch requires scipy, scikit-learn, and scikit-sparse (for sparse Cholesky factorization). Scikit-sparse tends to be difficult to install, especially on Windows, so this functionality was omitted from the master branch.

## Testing
Run `pytest test.py` in the `test` subdirectory.

[1]: http://jupyter.org/
[2]: https://gdg.engin.umich.edu/release-v1-0/
[3]: https://arxiv.org/abs/1411.0359
[pscc]: https://ieeexplore.ieee.org/document/8442682/
[shortest]: https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.shortest_paths.generic.average_shortest_path_length.html
[fiedler]: https://en.wikipedia.org/wiki/Algebraic_connectivity
[spectral]: https://en.wikipedia.org/wiki/Spectral_radius
[mc]: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.clique.find_cliques.html#networkx.algorithms.clique.find_cliques
