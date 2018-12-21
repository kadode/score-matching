#
python main.py --lr 0.01 --epoch 30 --batch_size 1 --num_sample 10000 --dim 2

#python main.py --lr 0.01 --epoch 20 --batch_size 1 --num_sample 10000 --dim 2

convert -delay 100 -loop 0 fig/*.png ./out.gif
