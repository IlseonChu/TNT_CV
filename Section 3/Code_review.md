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

## 이미지 Object Detection 실습 코드

https://user-images.githubusercontent.com/89925976/132988478-1ffdeab1-79e6-4465-aa49-49517afbcc09.GIF

