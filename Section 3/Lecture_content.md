# RCNN 계열 Object Detector

## RCNN

**: Regions with CNN features**

Object detection 모델에 처음으로 딥러닝을 적용한 모델이라는 점에 의의가 있음

**<RCNN 모델 개요>**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132945182-7b4ecf1b-c655-4453-968b-b16a16c9b917.GIF width = 800></p>

**1. Selective search**

**2. RCNN Network**
- 2000개의 proposed region을 모두 같은 비율로 warp 
- CNN 모델 적용
- 해당 region의 클래스 분류 (사람인가? 고양이인가?)

**3. SVM classifier / Bounding box regression**

**- RCNN Training**
1. ImageNet으로 Feature Extractor를 Pre-train.
2. Ground Truth와 Selectivesearch의 predicted region의 IOU가 0.5 이상이면 해당 클래스로, 아니면 Background로 분류
3. SVM 적용해서 GT로 학습 한 번 더 진행. IOU가 0.3 이하면 Background로 분류, 0.3 이상이지만 GT가 아니면 학습 후보에서 제외.

**- Bounding box regression** 
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132946559-83e76165-edb0-4f58-974c-f3449e3ad610.png width = 800></p>

- target 값과 Loss 함수를 최소화하도록 하는 regression을 진행함.

**장단점**
- 타 알고리즘에 비해 매우 높은 Detection 정확도 보유
- But, 매우 복잡한 아키텍처와 학습 프로세스 때문에 Detection 속도가 너무 느림. (한 장에 약 50초 소요)

**개선방안**
1. 2000개의 각 region이 아닌 원본 이미지만 CNN으로 처리. Selective search로 proposal된 region만 Feature Map으로 매핑하여 별도 추출.
2. warp 단계가 없었기 때문에 1D Flattened FC Input의 크기 고정이 불가능하다는 문제 발생.
3. 서로 다른 크기를 가진 Region Proposal 이미지를 SPP Net의 고정된 크기 Vector로 변환.


## SPP(Spatial pyramid Pooling)
: spatial pyramid matching(SPM)에 기반을 두는데, 자연어처리에서의 'bag of words'와 비슷한 개념을 활용.

**SPM**
- 원본 데이터를 각 단계별로 분면으로 나누고, 위치(공간)정보를 기반으로 해서 히스토그램을 만듦.
- 그 데이터가 원본 데이터를 변형한 새로운 데이터로써의 역할을 수행함.
- 데이터 크기를 고정할 수 있기 때문에 이미지를 warp 하지 않고도 1D Flattened FC Input의 크기 고정이 가능해짐.

- SPP-Net는 여기서 Max Pooling만 적용해 서로 다른 Feature Map 사이즈를 fixed-length representation함.

## RCNN에 SPP-Net 적용
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132947646-36c64bca-cad6-4f65-baeb-ef45b2ff3ea0.GIF width = 800></p>

- 이미지 한 장이 CNN 2000번을 통과해야했던 RCNN과는 달리 SPP-Net에서는 CNN을 한 번만 통과하면 됨.
- 성능 결과
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132947849-093fb518-1d99-445b-a282-1f4f4acd7e1a.GIF width = 800></p>

