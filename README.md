# Leveraging Joint-Diagonalization in Transform-Learning NMF

Reproduce the results in this paper

## Computations

- Gaps of Notes: run_table3b.sh
- Gaps of GCM: run_table1b and run_table1c
- Complexity: run_table2b.sh, run_table4b.sh

## Plots

- Evoluation of the emprical quantities: read from gcm_gaps_ratio10_itl1000 and notes_gaps_ratio10_itl100_me
- Plots of 8 most sig. atoms: notes_solutions_ratio10_itl100.py and notes_solutions_ratio10_itl100
- Evoluation of CS: gcm_complexity, notes_complexity

## Env: python3

"""
conda install -c conda-forge librosa
conda install numba==0.48
pip install soundfile
python setup.py install # install tlnmf in tlnmf_tsp/ and qn for JD in qndiag_tsp/
"""
