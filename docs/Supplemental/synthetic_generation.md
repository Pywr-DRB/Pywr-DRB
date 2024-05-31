# Synthetic streamflow generation

## Overview

This page describes the motivation and methods used to generate sets, or ensembles, of synthetic streamflow timeseries in the DRB.

### Ensemble generation goals

Exploratory, bottom-up policy assessments rely upon a sampling of a wide range of plausible scenarios (or streamflow conditions) and subsequent identification of consequential scenarios. Given that only a single historic streamflow timeseries is available, we generate a large set, or ensemble, of streamflow timeseries to explore via simulation.

#### Goals of synthetic generation:
1. Framework must be flexible enough to allow for drought not observed in the historic record, and include extreme events.
2. Must approximately maintain historic statistical properties.
3. Must provide parameters to adjust the severity of the scenario droughts.

***
## Kirsch-Nowak Synthetic Generator

The Kirsch-Nowak synthetic streamflow generation method has been shown to produce streamflow timeseries that maintain the historic streamflow statistics while capturing the large degree of internal variability inherent in the system.

The Kirsch-Nowak synthetic generation method consists of two components.

First, a bootstrapping technique combined with a Cholesky decomposition of the autocorrelation matrix are used to generate an ensemble of flows which preserves historic cross-site correlation and inter-site temporal correlation, as presented in {cite:p}`kirsch2013evaluating`. (This is the *Kirsch* part of the Kirsch-Nowak generator.)

The monthly flows are then disaggregated, converted from monthly flows to daily flow values using the method presented in {cite:p}`nowak2010nonparametric`.

### Kirsch monthly streamflow generator

Also known as the *modified Fractional Gaussian Noise* generator. This method bootstrap technique to sample from the historic record in a way that maintains cross-site correlations.  Additionally, a Cholesky decomposition of the historic autocorrelation matrix to impose inter-site temporal correlation.

This method is presented in detail in Kirsch et al. (2013).

Given historic streamflow at $m$ sites, $Q_H \in \mathcal{R}^{N_h \times T}$, where $N_H$ is the number of years and $T$ is the timesteps per year.

The goal is to generate $Q_S \in \mathcal{R}^{N_s \times T}$ for an arbitrary number $S$, while preserving spatial and temporal correlation.

The streamflow is standardized:

$$
Z_{H_{i,j}} = \frac{(Y_{H_{i,j}} - \hat{\mu}_j)}{\hat{\sigma}_j}
$$

A matrix of streamflow values, $C$ is sampled with replacement from $Z_H$, by using *indices* $M$ in which $M_{i,j}$ are sampled with replacement from $(1,2,...,N_H)$.

$M_{i,j}$ is the historical year that will be sampled to create the streamflow value in year $i$, week $j$ of the synthetic record.

Thus, $C$ is a matrix of *uncorrelated* streamflows:

$$
C_{i,j} = Z_{H_{(M_{i,j}),j}}
$$

The indices matrix $M$ is used to perform the bootstrap resampling for each site so that correlation of flows between sites can be approximately preserved.

The autocorrelation is then imposed at each site using site-specific historic autocorrelation matrix $P_H = \text{corr}(Z_H)$, in which $P_{H_{i,j}}$ is the historic correlation between week $i$ and week $j$.

Using Cholesky Decomposition, $P_H$ can be factored into triangular matrices: $P_H = U^TU$.

Then, the synthetic standard normal variables are:
$$
Z = CU
$$

which are transformed back into real-space flows following:

$$
Q_{S_{i,j}} = \text{exp}(\hat{\mu}_j + Z_{S_{i,j}}\hat{\sigma}_j)
$$

Then, inter-year correlations are preserved by repeating this process starting in week 27 and ending in week 26 of the following year, and constructing $Q_H^{'}$, $U^{'}$, $C^'$, and $Z^{'}$.

The final synthetic timeseries is then a combination of $Z[27:52]$ and $Z^{'}[1:26]$.

## Nowak disaggregation to daily flows

This method is presented in greater detail in the publication.

***
## References
