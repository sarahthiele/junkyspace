# Welcome to Junky Space

Examples:

python3 rayleigh.py --numsample 1000 --Lcmin 0.1 --satdistro satsold 
--KEkill 150e6 --path sat2021 --event Russia --maxtime 10. --plottime 0.25 
--AMval 0.04 > job1 & disown 

python3 NSBM.py --numsample 1000 --Lcmin 0.1 --satdistro satsold --KEkill 
150e6 --path sat2021 --event Russia --maxtime 10. --plottime 0.25 > job2 & 
disown 

python3 NSBM.py --numsample 1000 --Lcmin 0.003 --satdistro satcon --KEkill 
130e6 --path satall --event India --maxtime 5. --plottime 0.25 > job3 & 
disown 
