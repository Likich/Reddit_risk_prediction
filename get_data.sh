#!/usr/bin/env bash

wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
--save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=16_fhq0v1Q1uWi_M6voHJM6X_ppFbclpx' -O- | \
sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16_fhq0v1Q1uWi_M6voHJM6X_ppFbclpx" \
-O eRisk202_T2_training_data.zip 
unzip eRisk202_T2_training_data.zip