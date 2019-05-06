
#!/bin/bash


python3 predict.py --image "$1" --model output/cnn.model --label-bin output/cnn.pickle --width 64 --height 64



