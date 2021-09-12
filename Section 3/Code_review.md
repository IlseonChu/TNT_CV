# <Section 3> Code Review

## Open CV Deep Neural Network 장단점

**1. 장점**
- 딥러닝 개발 프레임 워크 없이 쉽게 Inference 구현 가능
- OpenCV에서 지원하는 다양한 Computer vision 처리 API와 Deep learning을 쉽게 결합

**2. 단점**
- GPU 지원 기능이 약함.
- Google에서 NVIDIA GPU 지원을 발표했으나 아직 환경 구성/ 설치가 어려움.
- OpenCV는 모델을 학습할 수 있는 방법을 제공하지 않으며 오직 Inference만 가능함.
- CPU 기반에서 Inference 속도가 개선되었으나, GPU(NVIDIA) 지원 여건이 안정적이지 않음.

## Inference 수행 로직

**1. 가중치 모델 파일과 환경 설정 파일을 로딩하고 Inference network 모델 생성**

`cvNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')`

**2. Input 이미지 Preprocessing 후, Inference network 모델에 입력**

`cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))`

**3. Inference network 모델에서 Output 추출**

`networkOutput = cvNet.forward()`

**4. 추출된 output에서 detect한 정보를 기반으로 원본 image위에 Bounding Box 및 캡션 시각화**

`for detection in networkOutput[0,0]:`

* * *

## 이미지 Object Detection 실습 코드

**1. DNN 패키지에서 readNetFromTensorflow()로 tensorflow inference 모델을 로딩**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132988478-1ffdeab1-79e6-4465-aa49-49517afbcc09.GIF width = 800></p>

- OpenCV는 딥러닝 가중치 모델을 생성하지 않고 타 Framework에서 생성된 모델을 변환하여 로딩함.
- DNN 패키지는 readNetFrom**FrameworkName**(가중치 모델파일, 환경 파일) API를 제공함.

**2. MS COCO 데이터 세트의 클래스id별 클래스명 지정**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132988859-8be490ef-dd84-4550-9749-8c853f9b05d1.GIF width = 1200></p>

- 클래스명이 할당되지 않은 클래스 id가 있어서, 0~79 / 0~90 / 1~91번 등 DNN 모델별 매핑방식이 다름.

**3. 실제 Object Detection 수행**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132989199-a4c03f22-c52c-436b-b1e2-7750c0a17e8f.GIF width = 1000></p>

- height를 `rows`에, width를 `cols`에 저장하고, 원본 이미지 변형을 피하기 위해 `draw_img`에 복사본 저장.
- `cv_net.setInput(cv2.dnn.blobFromImage(img, swapRB=True, crop=False))`로 inference network에 이미지 배열을 입력.
- 이 때, `swapRB=True, crop=False`이므로 BGR이 RGB로 변환되고 이미지 size는 변형되지 않음.
- `forward()` 메서드로 Object Detection을 수행하고 결과를 `cv_out`으로 반환

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132990307-4105303a-f358-405d-9848-abfc34aa7eaa.GIF width = 400></p>

**<`cv_out[i]`의 의미>**
- 0 : None
- 1 : class id
- 2 : confidence
- 3 : xmin
- 4 : ymin
- 5 : xmax
- 6 : ymax

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132989493-8ce5df0e-f5d0-43aa-b90c-03ff730960da.GIF width = 800></p>

- confidence 0.5 이상인 object에 대해서만 bounding box을 그리고 caption 달기. 
- for 구문에서, `cv_out[0,0,:,:]`로 전체 xmin, ymin에 대해 iteration 함.
- downsizing된 이미지를 `rows`와 `cols` 곱해줌으로써 원본 크기로 복원.

**<출력 결과>**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132990522-b3bb9a89-b9c9-4218-8b2a-b291bc811719.GIF width = 600></p>

**4. 단일 이미지의 object detection을 함수로 생성**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132991614-6e722c31-92bb-4f25-a1c8-d1a2be8e6d55.GIF width = 800></p>
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132991616-f62d8015-f152-4cdb-bd4a-f15bc4f54d05.GIF width = 800></p>

- 3에서 했던 로직을 그대로 사용.
- 추가적으로 Detection 시간 정보만 추가한 함수를 생성.

**5. 다른 이미지에 적용한 결과**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132991819-2cafd5cf-9602-466d-8629-eb3539fdff4e.GIF width = 800></p>

* * *

## 비디오 Object Detection 실습 코드

**이미지 Detection 코드와 매우 유사. 다만, frame 별로 같은 작업을 반복한다는 차이점이 존재.**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132992698-74a93bbe-7c6a-4cf6-aa45-94f63734099b.GIF width = 800></p>
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132992699-0b6d3595-4db3-4521-bcc5-a292ecbf5117.GIF width = 800></p>

- `cap = cv2.VideoCapture(input_path)`로 video를 frame 별로 capture.
- `codec = cv2.VideoWriter_fourcc(*'XVID')` 로 비디오 생성
- `get_detected_img` 함수를 사용해 frame 별로 image detection 진행.
- `.release()`로 bounding box 시각화된 video 파일 생성. 
