
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

python jointdiag/ex17e_jdnmf_best2_finite.py -S 1 -nbrun 100 -track 1 -epsc 1e-8 -pernmf 10 -nmfme 1
python jointdiag/ex17e_jdnmf_best2_finite.py -S 10 -nbrun 100 -track 1 -epsc 1e-8 -pernmf 10 -nmfme 1
python jointdiag/ex17e_jdnmf_best2_finite.py -S 100 -nbrun 100 -track 1 -epsc 1e-8 -pernmf 10 -nmfme 1
python jointdiag/ex17e_jdnmf_best2_finite.py -S 1000 -nbrun 100 -track 1 -epsc 1e-8 -pernmf 10 -nmfme 1
python jointdiag/ex17e_jdnmf_best2_finite.py -S 5000 -nbrun 100 -track 1 -epsc 1e-8 -pernmf 10 -nmfme 1


