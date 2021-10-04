# <Section 5> MMDetection의 이해와 Faster RCNN 적용 실습 2

### Config 대분류 및 주요 설정 내역

**1. Dataset**

: dataset의 type(CustomDataset, CocoDataset등), train/val/test Dataset 유형, data_root, train/val/test Dataset의 주요 파라미터 설정(type, ann_file, img_prefix, pipeline 등)

**2. Model**

: Object Detection Model 의 backbone, neck, dense head, roi extractor, roi head 주요 영역별로 세부 설정. 

**3. Schedule**

: optimizer 유형 설정(SGD, Adam, Rmsprop등), 최초 learning 설정 학습 중 동적 Learning rate 적용 정책 설정( step, cyclic, CosineAnnealing등) train 시 epochs 횟수

**4. Run time**

: 주로 hook(callback)관련 설정. 학습 중 checkpoint 파일, log 파일 생성을 위한 interval epochs수

### Data Pipeline의 구성
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/135808138-747bf3f3-75d9-4e9d-8b17-63fbed5855e3.png width = 1000></p>

- 초록 글씨 : 새로운 key 값이 추가되는 부분
- 노란 글씨 : 기존 key 값이 update 되는 부분

# <Section 6> SSD(Single Shot Detector)

### SSD Network 구조
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/135812882-e3c0b1ad-30c1-4d99-b85f-1cf04948d633.png width = 800></p>

### Multi Scale Feature Layer
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/135813306-9ccfc3bb-2ecb-4d4f-a08d-5e003c8e410b.png width = 800></p>

- 한 이미지 내에는 서로 다른 크기의 object들이 존재함.
- 따라서, 같은 크기의 sliding window로 한 그 object를 모두 detection 하기 어려움.
- Feature map의 크기를 줄여나가면서 detection을 반복함으로써 문제 해결

### Default(Anchor) Box
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/135815727-dcab7002-52e8-4330-9e93-5ede3f4b5f95.png width = 600></p>

- anchor box가 feature map 정보를 통해 object가 있을 곳을 예측해줌.
- 각 anchor box 별로 box 내 object가 무엇인지 클래스 분류
- GT box 예측을 위해 좌표 수정

### 
