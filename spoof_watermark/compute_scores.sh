python compute_scores.py --inp="images/imagenet/clean" --out="results/treeRing_clean.pkl" --data-cnt=50  --method="treeRing"
python compute_scores.py --inp="images/imagenet/treeRing" --out="results/treeRing_wm.pkl" --data-cnt=50  --method="treeRing"
python compute_scores.py --inp="images/spoofed/treeRing" --out="results/treeRing_spoofed.pkl" --data-cnt=50  --method="treeRing"

python compute_scores.py --inp="images/imagenet/clean" --out="results/watermarkDM_clean.pkl" --data-cnt=50 --method="watermarkDM"
python compute_scores.py --inp="images/imagenet/watermarkDM" --out="results/watermarkDM_wm.pkl" --data-cnt=50  --method="watermarkDM"
python compute_scores.py --inp="images/spoofed/watermarkDM" --out="results/watermarkDM_spoofed.pkl" --data-cnt=50  --method="watermarkDM"

python compute_scores.py --inp="images/imagenet/clean" --out="results/rivaGan_clean.pkl" --data-cnt=50 --method="rivaGan"
python compute_scores.py --inp="images/imagenet/rivaGan" --out="results/rivaGan_wm.pkl" --data-cnt=50  --method="rivaGan"
python compute_scores.py --inp="images/spoofed/rivaGan" --out="results/rivaGan_spoofed.pkl" --data-cnt=50  --method="rivaGan"

python compute_scores.py --inp="images/imagenet/clean" --out="results/dwtDct_clean.pkl" --data-cnt=50 --method="dwtDct"
python compute_scores.py --inp="images/imagenet/dwtDct" --out="results/dwtDct_wm.pkl" --data-cnt=50  --method="dwtDct"
python compute_scores.py --inp="images/spoofed/dwtDct" --out="results/dwtDct_spoofed.pkl" --data-cnt=50  --method="dwtDct"

python compute_scores.py --inp="images/imagenet/clean" --out="results/dwtDctSvd_clean.pkl" --data-cnt=50 --method="dwtDctSvd"
python compute_scores.py --inp="images/imagenet/dwtDctSvd" --out="results/dwtDctSvd_wm.pkl" --data-cnt=50  --method="dwtDctSvd"
python compute_scores.py --inp="images/spoofed/dwtDctSvd" --out="results/dwtDctSvd_spoofed.pkl" --data-cnt=50  --method="dwtDctSvd"

python plot_roc.py