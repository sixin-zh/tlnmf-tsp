DAT=nonstationary440_sim1_5k
ITL=100
#R=1
R=10
P=10
#wid=5
INMF=$( expr "$R" '*' "$ITL" )
#echo "$INMF"

EPS=5e-7

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# SAME EPS, ME
python jointdiag/tlnmf2_best2_sci_batch.py  -epsc $EPS -epsnmf $EPS -s 1 -itl $ITL -pertl 1 -pernmf $R -nbruns $P  -sn $DAT -tn $DAT -nmfme 1
python jointdiag/tlnmf2_best2_sci_batch.py  -epsc $EPS -epsnmf $EPS -s 10 -itl $ITL -pertl 1 -pernmf $R -nbruns $P  -sn $DAT -tn $DAT -nmfme 1
python jointdiag/tlnmf2_best2_sci_batch.py  -epsc $EPS -epsnmf $EPS -s 100 -itl $ITL -pertl 1 -pernmf $R -nbruns $P  -sn $DAT -tn $DAT -nmfme 1
python jointdiag/tlnmf2_best2_sci_batch.py  -epsc $EPS -epsnmf $EPS -s 1000 -itl $ITL -pertl 1 -pernmf $R -nbruns $P  -sn $DAT -tn $DAT -nmfme 1
python jointdiag/tlnmf2_best2_sci_batch.py  -epsc $EPS -epsnmf $EPS -s 5000 -itl $ITL -pertl 1 -pernmf $R -nbruns $P  -sn $DAT -tn $DAT -nmfme 1

python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $INMF -s 1 -epsc $EPS -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT -nmfme 1
python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $INMF -s 10 -epsc $EPS -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT -nmfme 1
python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $INMF -s 100 -epsc $EPS -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT -nmfme 1
python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $INMF -s 1000 -epsc $EPS -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT -nmfme 1
python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $INMF -s 5000 -epsc $EPS -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT -nmfme 1


