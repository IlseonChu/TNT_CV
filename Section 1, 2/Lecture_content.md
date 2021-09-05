# Section 1 lecture content
*Topic : Object detection / Segmentation / Region proposal / IOU / NMS / mAP*

- ### Classification, Localization, Detection, Segmentation

![이미지] C:\Users\cls12\Desktop\Section 1, 2\Object detection

1. Classification

      : 해당 이미지가 원하는 클래스가 맞는지 아닌지만 분류하는 수준.
      
2. Localization

      : 이미지 내에 Object가 1개 뿐인 경우에 Bounding box로 이미지 위치 탐색.
      
3. Detection

      : Localization과 달리 이미지 내에 여러 Object가 존재할 때 여러 개를 한 번에 탐색.
      
4. Segmentation

      : 찾아낸 Object들을 픽셀 단위로 접근하여 Bounding box보다 정확하게 물체를 파악.
      
- ### Object Detection의 구성요소
