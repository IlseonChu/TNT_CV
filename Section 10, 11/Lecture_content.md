# <Section 11> RetinaNet과 EfficientDet

## RetinaNet

### RetinaNet의 특징

**1. One Stage Detector로서의 장점인 빠른 Detection 속도 유지하면서, 단점이었던 Detection 성능 저하 문제는 개선**

**2. 수행 시간이 YOLO나 SSD보다 느리지만 Faster RCNN보다는 빠름**

**3. 수행 성능은 타 Detection 모델 보다 뛰어남. 특히 타 One Stage Detector보다 작은 오브젝트에 대한 Detection 능력이 뛰어남**

### Focal Loss

**1. Cross Entropy**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/140595629-7664aed2-a2a7-4e0b-a13b-ba9a579797b8.png width = 1000></p>

**- Detection 잘한 것은 더 정확하게, Detection 못한 것은 덜 못하게 하는 방향으로 학습된다.**


**2. Class imbalance**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/140595767-a47f9378-9e2c-4e17-a64c-9de642182086.png width = 600></p>

  - Background / 크고 선명한 물체 : Easy Example
  - 작고 불분명한 물체 : Hard Example

**-> Cross Entropy Loss는 Hard Examlple이 Easy Example보다 적은 위 그림과 같은 상황에서 Hard Example의 Detection 성능 개선이 아니라 Easy Example의 정확도를 더 높이는 방향으로 학습을 진행시킴. Background나 큰 물체는 이미 정확한 Detection이 수행되었기 때문에 이러한 성능 개선은 전체 성능을 높인다하더라도 유의미한 학습이 아님.**

**3. Focal Loss**

그래서 제안된 개념이 Focal Loss. Cross Entropy에 가중치를 부여하여 Class Imbalance issue를 해결.

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/140596142-644d87f9-0d6b-431b-9396-390b78db3504.png width = 1000></p>

**확실하게 Detection된 Object들에 대해서는 매우 작은 가중치를 부여해 더이상 성능개선이 일어나지 않도록 조절함.**


### FPN(Feature Pyramid Network)

![image](https://user-images.githubusercontent.com/89925976/140596610-00affb24-7cc5-4a4b-b35e-a7179b15964b.png)

- ResNet(BackBone) : Bottom-up
- FPN(Neck) : Top-down

**Top-down 방식으로 upsampling을 하면서 skip connection(더 좋은 resolution을 갖는, 즉 더 많은 정보량을 갖는 하위 이미지와 confusion)을 통해 Predict 성능을 높임**


## EfficientDet

### EfficientDet의 특징

**: 적은 연산 수, 적은 파라미터 수에 비해 상대적으로 타 모델보다 높은 모델 예측 성능을 나타냄. 즉, 연산량이 적은 데 비해 예측 성능은 뛰어난 모델임.**

### EfficientNet
**: Backbone에 해당하며, 네트웍의 깊이(Depth), 필터 수(Width), 이미지 Resolution 크기를 함께 최적으로 조합하여 모델 성능 극대화.**

<EfficientNet 아키텍처>

![image](https://user-images.githubusercontent.com/89925976/140597216-358d6765-8309-48e8-b13f-ea0cc6aada30.png)

- 필터 수, 네트웍 깊이를 일정 수준 이상 늘려도 성능 향상 미비 (단 Resolution의 경우 어느 정도 약간씩 성능 향상이 지속됨)
- ImageNet 데이터세트 기준 80% 정확도에서 개별 Scaling 요소를 증가 시키더라도 성능 향상이 어려움을 발견함.
- 이에 복합적으로 Scaling 하는 Compound Scaling 기법 도입.

### Compound Scaling

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/140597259-43e17d5a-2c08-4cc6-8e40-10d690e8a0b4.png width = 500></p>

- 최초에는 𝝋를 1로 고정하고 grid search 기반으로 𝛼, 𝛽, 𝛾의 최적 값을 찾아냄. EfficientNetB0의 경우 𝛼 =1.2, 𝛽=1.1, 𝛾=1.15 임.
- 다음으로 𝜶, 𝜷, 𝜸을 고정하고 𝜑을 증가 시켜가면서 EfficientB1~ B7까지 Scale up 구성.

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/140597300-c6487f30-0bf7-43d6-9d3e-ee8dc964050b.png width = 500></p>

**더 빠른 Inference 속도와 더 높은 예측 성능을 보여줌**

### BiFPN













