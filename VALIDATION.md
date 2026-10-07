# Release checks

The release was checked using Python 3.10 and the package versions in
`requirements.txt`.

- 14 numerical tests pass, including BGe marginal scores, Structure MCMC
  detailed balance, ratio-HT weights and budget accounting, DPVI monotonicity,
  OPAD support enlargement, NWSS discovery, and generalized R-hat.
- `python reproduce.py all --quick` completes the illustration, three KL jobs
  (four panels), four trade-off jobs, and three R-hat jobs, producing 30 PDFs
  plus PNGs and saved numerical results.
- Saved KL, BVS trade-off, and R-hat plots regenerate without resampling.
- The illustration yields KL 0.12011811695 for OPAD and 0.51181117209 for IS.
- The BVS target expectation matches 3.967976168799272.
- Release files were checked for personal paths, author identifiers, credentials,
  external result-directory dependencies, and LSAC code/data.

Full-budget experiments are invoked with the commands in the README.
