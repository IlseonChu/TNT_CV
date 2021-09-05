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

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132119154-f94816a5-8ead-44e2-906e-0501f190aa66.GIF width = 400)</p>

- Window를 왼쪽 상단에서 부터 오른쪽 하단으로 이동시키면서 Object를 Detection하는 방식
- 오브젝트 없는 영역도 무조건 슬라이딩. 여러 형태의 Window와 여러 Scale을 가진 이미지를 스캔해서 검출해야 하므로 수행 시간이 오래 걸리고 성능이 낮음.

      
## Object Detection Method 2 : Selective search
      
      


