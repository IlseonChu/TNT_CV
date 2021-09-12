# <Section 3> RCNN 계열 Object Detector

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

**- 장단점**
- 타 알고리즘에 비해 매우 높은 Detection 정확도 보유
- But, 매우 복잡한 아키텍처와 학습 프로세스 때문에 Detection 속도가 너무 느림. (한 장에 약 50초 소요)

**- 개선방안**
1. 2000개의 각 region이 아닌 원본 이미지만 CNN으로 처리. Selective search로 proposal된 region만 Feature Map으로 매핑하여 별도 추출.
2. warp 단계가 없었기 때문에 1D Flattened FC Input의 크기 고정이 불가능하다는 문제 발생.
3. 서로 다른 크기를 가진 Region Proposal 이미지를 SPP Net의 고정된 크기 Vector로 변환.


## SPP(Spatial pyramid Pooling)
: spatial pyramid matching(SPM)에 기반을 두는데, 자연어처리에서의 'bag of words'와 비슷한 개념을 활용.

**1. SPM**
- 원본 데이터를 각 단계별로 분면으로 나누고, 위치(공간)정보를 기반으로 해서 히스토그램을 만듦.
- 그 데이터가 원본 데이터를 변형한 새로운 데이터로써의 역할을 수행함.
- 데이터 크기를 고정할 수 있기 때문에 이미지를 warp 하지 않고도 1D Flattened FC Input의 크기 고정이 가능해짐.

- SPP-Net는 여기서 Max Pooling만 적용해 서로 다른 Feature Map 사이즈를 fixed-length representation함.

**2. RCNN에 SPP-Net 적용**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132947646-36c64bca-cad6-4f65-baeb-ef45b2ff3ea0.GIF width = 800></p>

- 이미지 한 장이 CNN 2000번을 통과해야했던 RCNN과는 달리 SPP-Net에서는 CNN을 한 번만 통과하면 됨.
- 성능 결과
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132947849-093fb518-1d99-445b-a282-1f4f4acd7e1a.GIF width = 800></p>

* * *

## Fast RCNN

**<기존 RCNN과의 차별점>**

**1. SPP Layer → ROI Pooling Layer**
-  ROI Pooling : ROI를 고정된 크기(일반적으로 7x7)로 max pooling하는 기법
-  정수형으로 딱 떨어지는 경우 : 분면으로 나누어 max pooling 적용
-  그렇지 않은 경우 : image.resize() 등의 method를 활용하여 7x7로 바로 매핑

**2. SVM → Softmax**
- SVM 대신에 딥러닝 모델 내부에서 Softmax(다중분류함수)를 사용함.
- 그 결과, Classification과 Regression Loss를 함께 반영한 Multi-task Loss 함수 사용이 가능해짐.
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132976566-31bb96ad-0c89-4991-acea-9f64d8d30b88.png width = 800></p>

## Faster RCNN
: RPN + Fast RCNN
- 모든 구성요소가 딥러닝 모델 기반으로 이뤄진 최초의 Object Dection 알고리즘
- Selective Search를 통한 ROI Proposal을 RPN(Region Proposal Network)로 대체

**1. RPN 구현 이슈**
- 주어진 데이터 : Pixel 값, Target : GT Bounding box.
- 어떻게 딥러닝 모델로 SS 수준의 Detection을 구현할 것인가?
- Anchor Box의 도입으로 이슈 해결 

**2. Anchor box**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132977144-b046b54b-4783-476e-8109-e60d4abf706d.GIF width = 500></p>
- 원본 사이즈 이미지가 Feature Map으로 Down sizing되고 grid point 생성
- grid point를 기준으로 각 점마다 9개의 anchor box를 그리게 됨.

## RPN
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132978073-f9276af4-24c1-4636-83ad-4fc0a5fa4df0.GIF width = 800></p>

**1. Network 구조**
- sigmoid classification : 1x1 convolution 적용해서 object인지 background인지 결정
- bbox regression : region proposal 수행

**2. RPN Bounding Box Regression**
- Anchor box를 Reference로 이용하는 방식
- Predicted bounding box와 Positive anchor box와의 좌표 차이와 Ground Truth와 Positive anchor box와의 좌표 차이가 최대한 동일하도록 regression함.

**3. Positive, Negative Anchor box**
- IOU 값이 가장 높은 Anchor box : positive
- 가장 높지 않아도 0.7 이상이면 : positive
- 0.3 이하면 : negetive → Background임을 확실히 하기 위해.
- 0.3 ~ 0.7 사이면 아예 Training 대상에서 제외.

**4. RPN / Faster RCNN Training**
- RPN : 각 128개의 positive/negative anchor box로 구성된 mini batch를 sampling해서 계속해서 학습시킴.
- Faster RCNN : 기본적으로 Alternating Training 구조. 
> RPN 학습 → Fast RCNN의 Classification/Regression 학습 → 다시 Loss 함수값을 feedback 삼아서 RPN fine tuning → Fast RCNN의 Classification/Regression fine tuning 을 계속 반복함.
