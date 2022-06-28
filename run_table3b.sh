DAT=nonstationary440_sim1_5k
#DAT=nonstationary440n3_sim1
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

#EPS=5e-7

# SAME EPS, MM
python jointdiag/tlnmf2_best2_sci_batch.py  -epsc $EPS -epsnmf $EPS -s 1 -itl $ITL -pertl 1 -pernmf $R -nbruns $P  -sn $DAT -tn $DAT
python jointdiag/tlnmf2_best2_sci_batch.py  -epsc $EPS -epsnmf $EPS -s 10 -itl $ITL -pertl 1 -pernmf $R -nbruns $P  -sn $DAT -tn $DAT
python jointdiag/tlnmf2_best2_sci_batch.py  -epsc $EPS -epsnmf $EPS -s 100 -itl $ITL -pertl 1 -pernmf $R -nbruns $P  -sn $DAT -tn $DAT
python jointdiag/tlnmf2_best2_sci_batch.py  -epsc $EPS -epsnmf $EPS -s 1000 -itl $ITL -pertl 1 -pernmf $R -nbruns $P  -sn $DAT -tn $DAT
python jointdiag/tlnmf2_best2_sci_batch.py  -epsc $EPS -epsnmf $EPS -s 5000 -itl $ITL -pertl 1 -pernmf $R -nbruns $P  -sn $DAT -tn $DAT

# JD
python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $INMF -s 1 -epsc $EPS -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT
python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $INMF -s 10 -epsc $EPS -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT
python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $INMF -s 100 -epsc $EPS -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT
python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $INMF -s 1000 -epsc $EPS -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT
python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $INMF -s 5000 -epsc $EPS -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT

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


#python jointdiag/tlnmf2_best2_sci_batch.py  -epsc 1.58113883e-7 -epsnmf $EPS -s 10 -itl $ITL -pertl 1 -pernmf $R -nbruns $P  -sn $DAT -tn $DAT

#python jointdiag/tlnmf2_best2_sci_batch.py  -epsc 5e-8 -epsnmf $EPS -s 100 -itl $ITL -pertl 1 -pernmf $R -nbruns $P -tn $DAT -sn $DAT
#python jointdiag/tlnmf2_best2_sci_batch.py  -epsc 1.5811388300841896e-08 -epsnmf $EPS -s 1000 -itl $ITL -pertl 1 -pernmf $R -nbruns $P -tn $DAT -sn $DAT
#python jointdiag/tlnmf2_best2_sci_batch.py  -epsc 7.071067811865474e-09 -epsnmf $EPS -s 5000 -itl $ITL -pertl 1 -pernmf $R -nbruns $P -tn $DAT -sn $DAT

#python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $ITL -s 1 -epsc $EPS -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT -win $wid
#python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $ITL -s 10 -epsc 1.58113883e-7 -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT -win $wid
#python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $ITL -s 100 -epsc 5e-8 -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT -win $wid
#python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $ITL -s 1000 -epsc 1.5811388300841896e-08 -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT -win $wid
#python jointdiag/tlnmfJD2_best2_sci_batch.py -itl $ITL -inmf $ITL -s 5000 -epsc 7.071067811865474e-09 -epsnmf $EPS -nbruns $P -tn $DAT -sn $DAT -win $wid

# OLD
#python jointdiag/tlnmf2_best2_sci_batch.py  -epsnmf 1.58113883e-7 -s 10 -itl 500 -pertl 1 -pernmf 1 -nbruns 10 -tn $DAT -sn $DAT

#python jointdiag/tlnmfJD2_best2_sci_batch.py -itl 500 -inmf 500 -s 1000 -epsc 1.5811388300841896e-08 -nbruns 10 -tn nonstationary440_sim1_5k -sn nonstationary440_sim1_5k
#python jointdiag/tlnmfJD2_best2_sci_batch.py -itl 500 -inmf 500 -s 5000 -epsc 7.071067811865474e-09 -nbruns 10 -tn nonstationary440_sim1_5k -sn nonstationary440_sim1_5k

#python jointdiag/tlnmfJD2_best2_sci_batch.py -itl 500 -inmf 500 -s 1 -epsc $EPS -nbruns 10 -tn nonstationary440_sim1_5k -sn nonstationary440_sim1_5k
#python jointdiag/tlnmfJD2_best2_sci_batch.py -itl 500 -inmf 500 -s 10 -epsc  1.58113883e-7 -nbruns 10 -tn nonstationary440_sim1_5k -sn nonstationary440_sim1_5k
#python jointdiag/tlnmfJD2_best2_sci_batch.py -itl 500 -inmf 500 -s 100 -epsc 5e-8 -nbruns 10 -tn nonstationary440_sim1_5k -sn nonstationary440_sim1_5k
