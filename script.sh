# !/bin/bash

# Train source models
# python train_source_model.py -n C1-SRC -l 1 -m CNN -w 0.0 &
# python train_source_model.py -n C2-SRC -l 2 -m CNN -w 0.0 &
# python train_source_model.py -n C3-SRC -l 3 -m CNN -w 0.0 &
# python train_source_model.py -n C4-SRC -l 4 -m CNN -w 0.0 &
# python train_source_model.py -n C5-SRC -l 5 -m CNN -w 0.0 &
# python train_source_model.py -n C6-SRC -l 6 -m CNN -w 0.0 &
# python train_source_model.py -n C7-SRC -l 7 -m CNN -w 0.0 &
# python train_source_model.py -n C8-SRC -l 8 -m CNN -w 0.0 &
# python train_source_model.py -n C9-SRC -l 9 -m CNN -w 0.0 &

# Baseline
# python train_rpm_model.py -n C1-1-STL -l 1 -m CNN -p weights/C1-1-SRC.pt &
# python train_rpm_model.py -n C2-1-STL -l 2 -m CNN -p weights/C2-1-SRC.pt &
# python train_rpm_model.py -n C4-8-STL -l 4 -m CNN -p weights/C4-8-SRC.pt &

# Visual Prompt
python train_rpm_model.py -n C1-1-VP-STL -l 1 -m CNN -p weights/C1-1-SRC.pt -v &
python train_rpm_model.py -n C2-1-VP-STL -l 2 -m CNN -p weights/C2-1-SRC.pt -v &
python train_rpm_model.py -n C4-8-VP-STL -l 4 -m CNN -p weights/C4-8-SRC.pt -v &

# FC Layer
python train_rpm_model.py -n C1-1-FC-STL -l 1 -m CNN -p weights/C1-1-SRC.pt -f &
python train_rpm_model.py -n C2-1-FC-STL -l 2 -m CNN -p weights/C2-1-SRC.pt -f &
python train_rpm_model.py -n C4-8-FC-STL -l 4 -m CNN -p weights/C4-8-SRC.pt -f &

# VP+FC
python train_rpm_model.py -n C1-1-MIX-STL -l 1 -m CNN -p weights/C1-1-SRC.pt -f -v &
python train_rpm_model.py -n C2-1-MIX-STL -l 2 -m CNN -p weights/C2-1-SRC.pt -f -v &
python train_rpm_model.py -n C4-8-MIX-STL -l 4 -m CNN -p weights/C4-8-SRC.pt -f -v &















# python train_source_model.py -n D-0.0 -m DNN -w 0.0 &
# python train_source_model.py -n D-0.001 -m DNN -w 0.001 &
# python train_source_model.py -n D-0.005 -m DNN -w 0.005 &
# python train_source_model.py -n D-0.01 -m DNN -w 0.01 &
# python train_source_model.py -n D-0.05 -m DNN -w 0.05 &
# python train_source_model.py -n D-0.1 -m DNN -w 0.1 &
# python train_source_model.py -n D-0.5 -m DNN -w 0.5 &
# python train_source_model.py -n D-1.0 -m DNN -w 1.0 &

# python train_source_model.py -n C-0.0 -m CNN -w 0.0 &
# python train_source_model.py -n C-0.001 -m CNN -w 0.001 &
# python train_source_model.py -n C-0.005 -m CNN -w 0.005 &
# python train_source_model.py -n C-0.01 -m CNN -w 0.01 &
# python train_source_model.py -n C-0.05 -m CNN -w 0.05 &
# python train_source_model.py -n C-0.1 -m CNN -w 0.1 &
# python train_source_model.py -n C-0.5 -m CNN -w 0.5 &
# python train_source_model.py -n C-1.0 -m CNN -w 1.0 &

# python train_rpm_model.py -m DNN6 -s weights/D-0.0.pt -d &

# python train_rpm_model.py -m DNN6 -s weights/D-0.0.pt &
# python train_rpm_model.py -m DNN6 -s weights/D-0.1.pt &
# python train_rpm_model.py -m DNN6 -s weights/D-0.01.pt &
# python train_rpm_model.py -m DNN6 -s weights/D-0.001.pt &
# python train_rpm_model.py -m DNN6 -s weights/D-1.0.pt &

# python train_rpm_model.py -m CNN6 -s weights/C-0.0.pt &
# python train_rpm_model.py -m CNN6 -s weights/C-0.1.pt &
# python train_rpm_model.py -m CNN6 -s weights/C-0.01.pt &
# python train_rpm_model.py -m CNN6 -s weights/C-0.001.pt &
# python train_rpm_model.py -m CNN6 -s weights/C-1.0.pt &


