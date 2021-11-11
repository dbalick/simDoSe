# simDoSe: `Sim`ulate `Do`minance and `Se`lection
*A fast Wright-Fisher simulator for arbitrary diploid selection evolving through realistic human demography.*

## Features
- Fast and flexible due to the absence of linkage (i.e., infinite recombination limit)
- Arbitrary dominance and selection coefficients, including under- and overdominant diploid selection.
- Properly handles high mutation rates by activating the recurrent mutation kernel.
- Choose from several literature-based demographies, as well as from equilibrium, linear growth, and exponential growth toy models.
- Models random sampling of a population to form a sequenced population sample. 
- Create many simulated 'genes' from a single simulation with a larger number of simulated sites.
- Create gene sets from an imported list of lengths (i.e., target size/mutation rate)
- Create 'russian doll' simulated genesets, each formed of genes with descending target size (Lgenes=L/10, Lgenes=L/100, ...)
- Can model 'biallelic' genes (e.g., LOF mutations with similar consequence in a single gene)
- Entirely command line-based, so the only needed software is Python 2.7 and the numpy, scipy, and pandas packages.
- Flexible output specification, including full population, population sample, and per-gene site frequency spectra and corresponding summary statistics

## Additional details, instructions for running, examples
Please see the simDoSe user manual for detailed information on the mathematical models and commannd line options.
