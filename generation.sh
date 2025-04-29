python guidance_generation.py --base configs/base/mimic_icustay_base.yaml --gpus 0, --uncond --logdir ${AMLT_BLOB_ROOT}/eICU_mor  -sl 24 --batch_size 128 --max_steps 20000 -lr 0.0001 -s 42 --resume
