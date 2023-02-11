# JOB_TYPE: 'baseline' or 'semi', decide which kind of job to run
# PERCENT_LABELED_DATA: 1, 5, 10. The ratio of labeled coco data in whole training dataset.
# GPU_NUM: number of gpus to run the job
# for FOLD in 1 2 3 4 5;
# do
#   bash tools/dist_train_partially.sh <JOB_TYPE> ${FOLD} <PERCENT_LABELED_DATA> <GPU_NUM>
# done
for FOLD in 1 ;
do
  bash tools/dist_train_partially.sh yolov3 ${FOLD} 10 1
done