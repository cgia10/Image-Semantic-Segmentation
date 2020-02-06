# Shell script for running the testing code of improved model

RESUME='./log/best_model.pth.tar'
python3 test_improved.py --resume $RESUME --data_dir $1 --pred_dir $2
