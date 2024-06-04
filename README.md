# NAVER_boostcamp-mask_classification

### 개요:
- 전염성 질환을 막기 위해 마스크 착용이 요구된다. 그러나 모든 사람의 마스크 착용 상태를 확인하는 것은 인력적으로 어려움이 있다.
- 따라서 마스크 착용 여부를 자동으로 판별할 수 있는 시스템 개발이 필요하게 되었다.
- 본 프로젝트에서는 얼굴 이미지를 통해 성별, 연령, 마스크 착용 여부를 판별하는 시스템 개발을 목적으로 한다.

### 주요 사용기법:
- 해당 task는 mask[normal, mask, incorrect], age[청년, 중년, 장년], gender[male, female] <br/>을 각각 분류하여 18개로 분류하는 multi label classification task이다.
- 각 label별로 최적의 모델을 만들어 해결하는 방법으로 접근 하였다.
- 위 과정에서 여러 모델(VIT-L, Convnext_Large, Efficientnet_b0 and b3) 를 실험하였고 최종적으로 Efficientnet_b0를 채택하였다.
- 입력 이미지에서 배경을 제외한 얼굴 부분에 집중하기 위해 Augmentation으로 Center Crop을 사용하였다. <br/>최종적으로 사용된 Augmentation은 다음과 같다.
  ```python
  train_transform=A.Compose([A.Resize(CFG['IMG_HEIGHT'], CFG['IMG_WIDTH']),
                           A.CenterCrop(300, 220, p=1),
                           A.HorizontalFlip(p=0.3),
                           A.OneOf([
                               A.MotionBlur(p=1),
                               A.OpticalDistortion(p=1),
                               A.GaussNoise(p=1)
                           ], p=0.3),
                           A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.3),
                           A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                           ToTensorV2()])


- 데이터에 불균형 완화를 위해 validation 과정에서 Stratified KFold 를 사용하였다
- 각 Fold에서 얻어진 model에 대해 soft_voting을 수행하였다.
- 서로 다른 모델에서 얻어진 output을 조합하여 18개 class로 반환하였다.

### 결과 & 회고
- label에 따라 별개의 모델을 사용하는 것이 아닌 하나의 모델을 통해 한번에 18개로 분류할 수 있는 방법에 대해 더 많은 최적화를 진행할 필요 있었다. 
- Convnext_Large와 같은 큰 모델으로 좋은 결과를 낸 경험이 많았다. 하지만 예상과 다르게 더 작은 모델인 Efficientnet_b0에서 더 높은 성능이 나왔다.
- Augmentation에서 경험적으로 좋은 결과를 얻었던 기법들을 사용하였다. 입력 데이터의 특성을 잘 파악하여 더 적합한 augmentation을 선정했으면 하는 아쉬움이 있다.
- 도메인에 대한 파악이 이루어지고 실험을 하는게 중요하다고 다시 한번 느꼈다.
- streamlit을 통해 학습된 모델 서빙을 시도하였다. <br/>
![마스크](https://github.com/KANG-dg/NAVER_boostcamp-mask_classification/assets/121837927/7e7647c2-5ae1-4c2c-bafb-d6219f8f12da)

### 이후 추가 시도사항
- Downsampling을 통한 성능향상
- 60대 이상 에서만 모델이 분류를 잘 못하기에 모델에 혼동을 주는 경계값(50대후반)에서 downsampling 진행
- class imbalance를 완화하기 위해 weighted random sampler를 사용(참고: [weighted_random_sampler](https://yeong-jin-data-blog.tistory.com/entry/%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98-%EC%8A%A4%ED%84%B0%EB%94%94-%ED%81%B4%EB%9E%98%EC%8A%A4-%EB%B6%88%EA%B7%A0%ED%98%95-%EB%8B%A4%EB%A3%A8%EA%B8%B0-%EA%B0%80%EC%A4%91-%EB%AC%B4%EC%9E%91%EC%9C%84-%EC%83%98%ED%94%8C%EB%A7%81-%EA%B0%80%EC%A4%91-%EC%86%90%EC%8B%A4-%ED%95%A8%EC%88%98))
- efficientnet v2 사용 고려 해 볼 것
- 데이터 맞는 augmentation 적용 (샤프닝 등으로 사람 얼굴 주름 더 판단하기 쉽게)
