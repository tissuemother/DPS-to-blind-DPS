# AI618 YSJ,YRP,SBH
customized time classifier with DPS

실행 방법:

guided-diffusion에서 classifier_train.py 통해 classifier.py 훈련 완료 이후 

python sample_condition.py --model_config=configs/model_config.yaml --diffusion_config=configs/diffusion_config.yaml   --classifier_path ./models/model009999.pt --task_config=configs/{task_config}

로 실행 가능함. model pt는 classifier model 제외하곤 pretrained ffhq 사용했음.
