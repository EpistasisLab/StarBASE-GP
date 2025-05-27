# StarBASE-GP

<div align="center">
  <img src="./docs/starbase-gp-logo.png" alt="Logo" width="540" height="540">
</div>

StarBASE-GP : Star-Based Automated Single-locus and Epistasis analysis tool - Genetic Programming
==================================

## Abstract

> We present the Star-Based Automated Single-locus and Epistasis analysis tool – Genetic Programming (StarBASE-GP), an automated framework for discovering meaningful genetic variants associated with phenotypic variation in large-scale genomic datasets. 
StarBASE-GP uses a genetic programming–based multi-objective optimization strategy to evolve machine learning pipelines that simultaneously maximize explanatory power ($r^2$) and minimize pipeline complexity. 
Biological domain knowledge is integrated at multiple stages, including the use of nine inheritance encoding strategies to model deviations from additivity, a custom linkage disequilibrium pruning node that minimizes redundancy among features, and a dynamic variant recommendation system that prioritizes informative candidates for pipeline inclusion. 
We evaluate StarBASE-GP on a cohort of \textit{Rattus norvegicus} (brown rat) to identify variants associated with body mass index, benchmarking its performance against a random baseline and a biologically naïve version of the tool. 
StarBASE-GP consistently evolves Pareto fronts with superior performance, yielding higher accuracy in identifying both ground truth and novel quantitative trait loci, highlighting relevant targets for future validation. 
By incorporating evolutionary search and relevant biological theory into a flexible automated machine learning framework, StarBASE-GP demonstrates robust potential for advancing variant discovery in complex traits.
