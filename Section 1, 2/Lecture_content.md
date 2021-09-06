# <Section 1> Object Detection의 이해 

## Classification, Localization, Detection, Segmentation

![Object detection](https://user-images.githubusercontent.com/89925976/132117220-ba2f94a4-4fd8-4817-b0c2-4f0ba631d543.png)

**1. Classification**

 : 해당 이미지가 원하는 클래스가 맞는지 아닌지만 분류하는 수준.
      
**2. Localization**

 : 이미지 내에 Object가 1개 뿐인 경우에 Bounding box로 이미지 위치 탐색.
      
**3. Detection**

 : Localization과 달리 이미지 내에 여러 Object가 존재할 때 여러 개를 한 번에 탐색.
      
**4. Segmentation**

 : 찾아낸 Object들을 픽셀 단위로 접근하여 Bounding box보다 정확하게 물체를 파악.
     
## Object Localization의 이해

![캡처1](https://user-images.githubusercontent.com/89925976/132119006-db278c6e-f4d6-4dc0-82b4-c0d3b5bb80ec.GIF)

**1. Image classification과 동일한 방식으로 이미지 내 Object의 클래스를 판단.**

**2. Classification과 다른 점은 Feature Map에 별도의 Regression Layer가 나온다는 것.** 

**3. Bounding box regression에서 CNN 가중치가 update 됨에 따라 bbox 크기는 작아지고 모델 성능은 좋아짐.**

**4. 여러 종류의 이미지로 계속 학습 진행.**

* * *

## Object Detection 모델의 구성

![캡처](https://user-images.githubusercontent.com/89925976/132118593-80dfe3bb-a859-42fc-85d5-7ec830356b29.GIF)

**1. Backbone** : VGG나 Resnet 등의 feature extractor를 활용해 원본 이미지의 annotation으로부터 Feature Map을 생성함.

**2. Neck** : Feature Pyramid Network를 통해 작은 object의 정보도 체계화해 분류 가능케 함.

**3. Head** : Network prediction 단계로, image classification과 bounding box regression이 이뤄짐.


## Object Detection Method 1 : Sliding Window

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132119154-f94816a5-8ead-44e2-906e-0501f190aa66.GIF width = 400></p>

- Window를 왼쪽 상단에서 부터 오른쪽 하단으로 이동시키면서 Object를 Detection하는 방식
- 오브젝트 없는 영역도 무조건 슬라이딩. 여러 형태의 Window와 여러 Scale을 가진 이미지를 스캔해서 검출해야 하므로 수행 시간이 오래 걸리고 성능이 낮음.


## Object Detection Method 2 : Selective search

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132120412-e2220c66-b51f-406b-a3d8-717065f237cf.png width = 600></p>
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132120428-2fda025e-b99e-469c-b9c0-9fe4edc39295.GIF width = 600></p>

- Region Proposal의 대표 기법으로, 탐색 속도가 빠르고 예측 성능이 좋음.
- 최초 segmentation은 Pixel intensity에 기반하여 Over segmentation을 수행함.
- 컬러/무늬/크기/형태 등이 유사한 region을 계속해서 묶어나가는 계층적 그룹핑 방법 적용.
- 반복 적용하면 segmentation이 단순화되면서 region proposal을 수행.
 
## Object Detection Method 3 : Non Max Supression (NMS)
 
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132120634-5f0b3ffb-2d2e-48c1-b839-c2106ba00c7e.GIF width = 700></p>

- Bounding box중에 비슷한 위치에 있는 box를 제거하고 가장 적합한 box를 선택하는 기법

**<NMS 수행 로직>**
 
 1. Detected 된 bounding box별로 특정 Confidence threshold 이하 bounding box는 먼저 제거.
 2. 높은 confidence score를 가진 box 순으로 내림차순 정렬
 3. 높은 confidence score를 가진 box와 겹치는 다른 box 간의 IOU 값을 모두 조사하여 특정 threshold 이상인 box를 모두 제거
 4. 남아 있는 box만 선택
 
 **따라서, Confidence score가 높고, IOU threshold가 낮을수록 많은 Box가 제거됨.**
 
 * * *
 
## Intersection Over Union (IOU)

**: Ground Truth Box와 Predicted Box가 겹치는 정도를 0에서 1 사이의 값으로 나타낸 것.**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132120914-f2192981-153a-4b86-916a-c653f52c4960.GIF width = 400></p>

**성능평가 기준**
- Pascal VOC : 0.5를 기준으로 True/False
- MS COCO : (.5:.05:.95)로 IOU 값 달리 적용. 또한 이미지 크기에 따라서도 다른 기준 적용함.

## 오차행렬 (Confusion Matrix)

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132121554-cc08a761-9be2-4987-8956-82fde5c0e0a6.GIF width = 400></p>

- False Negative : Detection 실패한 경우
- True Negative : 지표로 활용 X
- False Positive : 1) wrong class
                   2) IOU < 0.5
                   3) No overlap
- True Positive : Detection 성공한 경우

## 정밀도와 재현율

**: Object Detection 성능 평가 지표**

**정밀도**
: Prediction과 Ground Truth가 얼마나 일치하는가.

**= TP/(FP+TP)**

**재현율**
: 실제 Object를 얼마나 빠트리지 않고 정확히 잡아내는가

**= TP/(FN+TP)**

**Confidence 임계값에 따른 정밀도 & 재현율**
> 임계값 낮을수록 bounding box 난사. 정밀도 감소, 재현율 증가

> 임계값 높을수록 bounding box 신중. 정밀도 증가, 재현율 감소

## mean Average Precision (mAP)

**<정밀도 재현율 곡선>**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132122554-7905d9d3-3ebe-4ecf-9885-330a3d50c6ca.png width = 400></p>

- Recall 값에 대응하는 Precision 값 나타낸 그래프.
- 그래프 아래 면적이 AP 값이고, 여러 오브젝트들의 AP 값을 평균낸 것이 mAP.

* * *

# <Section 2> Object Detection과 Segmentation을 위한 주요 데이터 세트 및 OpenCV 소개

## 주요 데이터 세트
대부분의 Detection / Segmentation 딥러닝 패키지는 아래 데이터 세트로 Pretrained 되어 배포됨.

**1. PASCAL VOC**
 : XML Format. 20개 카테고리
 
**2. MS COCO**
 : json Format. 80개 카테고리
 
**3. Google Open Images**
 : csv Format. 600개 카테고리
 
## PASCAL VOC

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132167700-09383ae4-651d-4c88-b4fb-1e9a055a796d.GIF width = 400></p>

PASCAL 2012가 기준이며, Dataset 구조는 다음과 같다.
- Annotations : XML 포맷이며, 개별 XML파일은 한 개 image에 대한 Annotation 정보를 가지고 있음. 확장자 xml을 제외한 파일명은 image 파일명(확장자 jpg를 제외한)과 동일하게 매핑
- ImageSet : 어떤 이미지를 train, test, trainval, val에 사용할 것인지에 대한 매핑 정보를 개별 오브젝트별로 파일로 가지고 있음.
- JPEGImages : Detection과 Segmentation에 사용될 원본 이미지로, Annotations과 짝을 이룸.
- SegmentationClass : Semantic Segmentation에 사용될 masking 이미지
- SegmentationObject : Instance Segmentation에 사용될 masking 이미지

## MS COCO

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132167765-5ce59594-90bb-49bf-b8e9-438fbf877821.GIF width = 400></p>
- 가장 자주 쓰이는 데이터 세트로, Tensorflow Object Detection API 및 많은 오픈 소스 계열의 주요 패키지들은 COCO Dataset으로 Pretrained 된 모델을 제공함.
 
- 80개 Object Category가 있으며, 300K개의 Image들과 1.5M개의 object들이 존재한다.
 
- 하나의 image에 여러 object가 포함되어 있으며 타 데이터 세트에 비해 난이도가 높은 데이터를 제공함.

**1. 이미지 파일**

 학습용(train2017), 검증용(val2017), 테스트용(test2017) 파일로 구성되어 있음.
 
**2. JSON Annotation 파일** 

json 포맷인 하나의 파일에 annotation 정보 모두 들어있음.

> info : COCO Dataset 생성 일자등을 가지는 헤더 정보임.
> 
> license : 이미지 파일들의 라이선스에 대한 정보
> 
> images : 모든 이미지들의 id, 파일명, 이미지 너비, 높이 정보
> 
> annotations : 대상 image 및 object id, Segmentation, bounding box, 픽셀 영역등의 상세 정보
> 
> categories : 80개 오브젝트 카테고리에 대한 id, 이름, Group을 가짐.

## Open CV

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132168978-982f0b55-e63f-494b-a219-a1d197963aba.GIF width = 400></p>

- 인텔이 초기 개발을 주도했음
- Windows, 리눅스, Mac OS X, 안드로이드, i-OS 등 다양한 플랫폼에서 사용 가능
- 방대한 Computer Vision 라이브러리와 손쉬운 인터페이스 제공이 특징임

## Object Detection 네트워크 개요

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132175616-4ec7f3fc-2204-49ab-aabf-a2a08caf84d1.GIF width = 400></p>

- Feature Extractor Network (VGG, RESNET, INCEPTION 등) : 보통 ImageNet 데이터 세트 기반으로 Pretrained 됨.
- Object Detection Network : 보통 Pascal VOC 또는 MS COCO 데이터 세트 기반으로 Pretrained 됨.

**Image Resolution, FPS, Detection 성능 상관 관계**

Image resolution이 높을수록 Object Detection 성능은 높아지고 FPS는 낮아진다.

따라서, FPS가 높아야하는 경우에는 Object Detection 성능이 낮아질 수 밖에 없다.
