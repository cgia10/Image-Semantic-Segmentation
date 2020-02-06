# Shell script for running the testing code of the baseline model

RESUME='./log/baseline_model.pth.tar'
python3 test_baseline.py --resume $RESUME --data_dir $1 --pred_dir $2