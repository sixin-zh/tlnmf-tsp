export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Case fixed eps
# 1. run best tlnmf, then get best runid, in notes_complexity.ipynb -> runid = 8 for MM, 4 for ME
#python jointdiag/tlnmf2_best_sci_batch.py  -sn nonstationary440_sim1_5k -epsnmf 5e-7 -s 100 -itl 100 -pertl 1 -pernmf 10 -nbruns 10 -nmfme 1

# 2. apply best runid to both TL-NMF and JD+NMF
python jointdiag/tlnmf2b_sci_batch.py  -sn nonstationary440_sim1_5k -tn nonstationary440_sim1_5k -epsnmf 5e-7 -runid 4 -s 100 -itl 100 -pertl 1 -pernmf 10 -nmfme 1
python jointdiag/tlnmfJD2b_sci_batch.py  -sn nonstationary440_sim1_5k -s 100 -ratio 10 -itl 100  -epsc 5e-7 -runid 4 -nmfme 1 

## Case R=10
#python jointdiag/tlnmf2_best_sci_batch.py  -epsnmf 5e-8 -s 100 -itl 500 -pertl 1 -pernmf 10 -nbruns 10
#python jointdiag/tlnmf2b_sci_batch.py  -epsnmf 5e-8 -runid 4 -s 100 -itl 500 -pertl 1 -pernmf 10
#python jointdiag/tlnmfJD2b_sci_batch.py  -s 100 -ratio 10 -itl 500  -epsc 5e-8 -runid 4
#python jointdiag/tlnmfJD2_sci_batch.py  -s 100 -itl 20 -inmf 200 -epsc 5e-8 -runid 4

#python jointdiag/tlnmf2_sci_batch.py  -epsnmf 5e-8 -s 100 -itl 20 -pertl 1 -pernmf 10 -runid 4 -cache 0


