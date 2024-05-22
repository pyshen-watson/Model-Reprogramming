# !/bin/bash
python train_source_model.py -n D1-0.0 -l 1 -m DNN -w 0.0 &
python train_source_model.py -n D2-0.0 -l 2 -m DNN -w 0.0 &
python train_source_model.py -n D3-0.0 -l 3 -m DNN -w 0.0 &
python train_source_model.py -n D4-0.0 -l 4 -m DNN -w 0.0 &
python train_source_model.py -n D5-0.0 -l 5 -m DNN -w 0.0 &
python train_source_model.py -n D6-0.0 -l 6 -m DNN -w 0.0 &
python train_source_model.py -n D7-0.0 -l 7 -m DNN -w 0.0 &
python train_source_model.py -n D8-0.0 -l 8 -m DNN -w 0.0 &
python train_source_model.py -n D9-0.0 -l 9 -m DNN -w 0.0 &

# python train_rpm_model.py -n D1-VP -l 1 -m DNN -p weights/by_depth/src_D1-0.0.pt -v &
# python train_rpm_model.py -n D2-VP -l 2 -m DNN -p weights/by_depth/src_D2-0.0.pt -v &
# python train_rpm_model.py -n D3-VP -l 3 -m DNN -p weights/by_depth/src_D3-0.0.pt -v &
# python train_rpm_model.py -n D4-VP -l 4 -m DNN -p weights/by_depth/src_D4-0.0.pt -v &
# python train_rpm_model.py -n D5-VP -l 5 -m DNN -p weights/by_depth/src_D5-0.0.pt -v &
# python train_rpm_model.py -n D6-VP -l 6 -m DNN -p weights/by_depth/src_D6-0.0.pt -v &
# python train_rpm_model.py -n D7-VP -l 7 -m DNN -p weights/by_depth/src_D7-0.0.pt -v &
# python train_rpm_model.py -n D8-VP -l 8 -m DNN -p weights/by_depth/src_D8-0.0.pt -v &
# python train_rpm_model.py -n D9-VP -l 9 -m DNN -p weights/by_depth/src_D9-0.0.pt -v &

# python train_rpm_model.py -n D1-FC -l 1 -m DNN -p weights/by_depth/src_D1-0.0.pt -f &
# python train_rpm_model.py -n D2-FC -l 2 -m DNN -p weights/by_depth/src_D2-0.0.pt -f &
# python train_rpm_model.py -n D3-FC -l 3 -m DNN -p weights/by_depth/src_D3-0.0.pt -f &
# python train_rpm_model.py -n D4-FC -l 4 -m DNN -p weights/by_depth/src_D4-0.0.pt -f &
# python train_rpm_model.py -n D5-FC -l 5 -m DNN -p weights/by_depth/src_D5-0.0.pt -f &
# python train_rpm_model.py -n D6-FC -l 6 -m DNN -p weights/by_depth/src_D6-0.0.pt -f &
# python train_rpm_model.py -n D7-FC -l 7 -m DNN -p weights/by_depth/src_D7-0.0.pt -f &
# python train_rpm_model.py -n D8-FC -l 8 -m DNN -p weights/by_depth/src_D8-0.0.pt -f &
# python train_rpm_model.py -n D9-FC -l 9 -m DNN -p weights/by_depth/src_D9-0.0.pt -f &

# python train_rpm_model.py -n D1-VPFC -l 1 -m DNN -p weights/by_depth/src_D1-0.0.pt -f -v &
# python train_rpm_model.py -n D2-VPFC -l 2 -m DNN -p weights/by_depth/src_D2-0.0.pt -f -v &
# python train_rpm_model.py -n D3-VPFC -l 3 -m DNN -p weights/by_depth/src_D3-0.0.pt -f -v &
# python train_rpm_model.py -n D4-VPFC -l 4 -m DNN -p weights/by_depth/src_D4-0.0.pt -f -v &
# python train_rpm_model.py -n D5-VPFC -l 5 -m DNN -p weights/by_depth/src_D5-0.0.pt -f -v &
# python train_rpm_model.py -n D6-VPFC -l 6 -m DNN -p weights/by_depth/src_D6-0.0.pt -f -v &
# python train_rpm_model.py -n D7-VPFC -l 7 -m DNN -p weights/by_depth/src_D7-0.0.pt -f -v &
# python train_rpm_model.py -n D8-VPFC -l 8 -m DNN -p weights/by_depth/src_D8-0.0.pt -f -v &
# python train_rpm_model.py -n D9-VPFC -l 9 -m DNN -p weights/by_depth/src_D9-0.0.pt -f -v &


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


