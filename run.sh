# Add the following line to enable test mode; otherwise, it defaults to validation mode
export TEST_MODE=True
export TEST_SAMPLE_COUNT=30
# Use a Hugging Face mirror when huggingface.co is unreachable.
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com

# Baseline: greedy decoding, dataset could be "Controlled_Images_A", "Controlled_Images_B", "COCO_QA_one_obj", "COCO_QA_two_obj", "VG_QA_one_obj", "VG_QA_two_obj", "VSR"
# For "Controlled_Images_A", "Controlled_Images_B", "COCO_QA_one_obj", "COCO_QA_two_obj", use "four" option.
# For "VG_QA_one_obj" and "VG_QA_two_obj", use "six" option. 
# python3 main_aro.py --dataset=Controlled_Images_A --model-name='llava1.5' --download --method=base  --option=four

# For Scaling_Vis on Controlled_A, a weight of 0.8 is used. 
# For Scaling_Vis on Controlled_B, a weight of 0.8 is used.
# For Scaling_Vis on COCO_QA_one_obj, a weight of 1.2 is used.
# For Scaling_Vis on COCO_QA_two_obj, a weight of 1.2 is used.
# For Scaling_Vis on VG_QA_one_obj, a weight of 2.0 is used.
# For Scaling_Vis on VG_QA_two_obj, a weight of 2.0 is used.
# For Scaling_Vis on VSR, a weight of 0.5 is used.
# python3 main_aro.py --dataset=Controlled_Images_A --model-name='llava1.5' --download --method=scaling_vis  --weight=0.8  --option=four

# For Adapt_Vis on COCO_QA_one_obj, weight1=0.5, weight2=1.2, threshold=0.3.
# Layer-range smooth setting:
#   --adjust-method smooth:14-31  -> only smooth between layer 14 and 31
#   --adjust-method smooth:14-27  -> only smooth between layer 14 and 27
python3 main_aro.py --dataset=COCO_QA_one_obj --model-name='llava1.5' --download --method adapt_vis --weight1 0.5 --weight2 1.2 --threshold 0.3 --option=four --adjust-method smooth:14-31
python3 main_aro.py --dataset=COCO_QA_one_obj --model-name='llava1.5' --download --method adapt_vis --weight1 0.5 --weight2 1.2 --threshold 0.3 --option=four --adjust-method smooth:14-27
