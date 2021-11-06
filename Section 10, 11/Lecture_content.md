# <Section 11> RetinaNet과 EfficientDet

### RetinaNet의 특징

**1. One Stage Detector로서의 장점인 빠른 Detection 속도 유지하면서, 단점이었던 Detection 성능 저하 문제는 개선**

**2. 수행 시간이 YOLO나 SSD보다 느리지만 Faster RCNN보다는 빠름**

**3. 수행 성능은 타 Detection 모델 보다 뛰어남. 특히 타 One Stage Detector보다 작은 오브젝트에 대한 Detection 능력이 뛰어남**

### Focal Loss

**1. Cross Entropy**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/140595629-7664aed2-a2a7-4e0b-a13b-ba9a579797b8.png width = 1000></p>
