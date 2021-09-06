# Section 1 & 2 Code Review


## Selective search 기법
먼저 `!mkdir` 과 `!wget-O`를 사용해 directory를 생성하고, 그 directory 내에 해당 url의 이미지를 특정 이름으로 저장함.

**1. 오드리헵번 이미지를 cv2로 로드하고 matplotlib으로 시각화**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132178554-7063d1e8-fd72-42df-9704-fe11bd30302b.GIF width = 400></p>

- 오드리헵번 이미지를 `cv2.imread()` 으로 읽어 변수에 저장하고, `plt.show()`로 시각화.

**2. Region Proposal(후보 영역)에 대한 정보 보기**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132179256-996f9d5a-d3c5-440c-8f38-459e3e43b809.GIF width = 600></p>

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132179512-09b0347e-dd23-469b-baca-cba8bbc1b3d1.GIF width = 500></p>

Region : 세부 원소로 딕셔너리를 가지고 있는 리스트

- rect : x,y 시작 좌표와 너비, 높이 값을 의미. 이 값이 Detected Object 후보를 나타내는 Bounding box임.
- size : segment로 select된 Object의 크기
- labels : 해당 rect로 지정된 Bounding Bow 내에 있는 오브젝트들의 고유 ID

**3. Bounding Box 시각화하고 크기가 큰 Box들만 추출하기**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132180241-89922069-1b19-44e0-9715-28c54594e00c.GIF width = 800></p>

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132180622-14f5149e-fadb-487b-9a41-3e2a6bd35855.GIF width = 800></p>

`rectangle()` 에 이미지 좌표정보 입력하여 Bounding box를 그려주고 나서, `cand_rects = [cand['rect'] for cand in regions if cand['size'] > 10000]`로 특정 크기 이상의 Bounding Box만 추출함.

**4. 결과**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132181930-e15f4651-eec2-487e-a6cd-e72bb3385ad9.png width = 800></p>

## IOU 구하기

**1. `cand_box`와 `gt_box`를 입력받아서 직접 IOU 값을 계산해내는 함수 생성**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132182674-149bdda6-6126-46e1-9389-35265e469724.GIF width = 500></p>

IOU = intersection / union 이므로 maximum, minimum 값 더하고 빼서 구할 수 있음.

**2. 특정 IOU 값 이상인 Bounding Box만 추출**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132183326-7a8832a8-651a-4c18-9f2b-dc4795668afe.GIF width = 800></p>

`compute_iou` 함수로 각 후보 Bounding Box의 IOU 값 계산하고, 특정 값 이상의 Bounding Box의 정보만 추출하고 시각화함.

**3. 결과**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132184009-a0c61ff1-8870-441a-ad82-b452d69e72f9.GIF width = 400></p>

* * *

## PASCAL VOC

**1. 압축파일 로딩**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132190198-b5472c4a-5051-4996-8bee-7677109318cb.GIF width = 600></p>

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132190712-e2523436-a9cc-4756-9154-246652c01d88.GIF width = 600></p>

- 디렉토리 생성 후 압축파일을 저장하고 `!tar -xvf`로 압축을 해제함.
- `head -n 5` 로 맨 앞 5개 파일의 이름을 출력할 수 있음.


**2. 디렉토리 내 임의의 Image 파일과 Annotation 파일 보기**

**- Image 파일**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132191984-b73e05c2-37c2-47f2-8c84-7992eab14117.GIF width = 600></p>

`.imread()`, `.cvtColor()`, `.imshow()` 사용해 이미지 시각화

`os.path.join()`으로 상세 파일과 디렉토리를 지정함.

**- Annotation 파일**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132192085-0783dc0b-c8b6-4c7c-ae40-175e7cbed9ff.GIF width = 400></p>

`!cat` 사용해서 annotation 파일을 볼 수 있음.

**3. ElementTree를 이용해 XML 파싱하기**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132193504-2ead7915-63b7-4f35-8c79-16d23ac5e624.GIF width = 400></p>

- VOC2012 파일, Annotation 파일, JPEGImages 파일의 경로를 변수에 저장하는 코드

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132193521-af9aa4c4-0857-4ef2-90cc-f2d5414a3feb.GIF width = 600></p>

- 변수에 임의의 파일을 parsing 하여 Element를 생성한 뒤에, `tree.getroot()`로 node를 탐색함.
- filename / size / width / height / bounding box 좌표값 에 대한 정보를 얻게 됨.

**4. Bounding box 시각화**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132193548-02763e7c-baa8-46fb-a9e6-accc36b8cc87.GIF width = 800></p>

- Bounding box 좌표 정보와 Object의 이름이 담긴 `objects_list`를 바탕으로 Bounding box 시각화

**5. 결과**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132196380-1f01e428-d4b2-48ed-8919-c0ef22fd5b5d.GIF width = 600></p>


* * *

## OpenCV 이미지 로딩

**1. `.imread()`로 이미지 읽어오기**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132184602-67303657-e43f-47e3-be20-59c54b0145fa.GIF width = 600></p>

이 때, 원본 RGB 이미지는 BGR 형태의 numpy array로 반환되기 때문에 색감이 달라질 수 있음. (Red -> Blue)

**2. `.cvtColor()`로 BGR을 RGB로 변환하기**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132185717-6595d806-dfe7-48e6-bcfb-923c8d455107.GIF width = 500></p>

**3. `.imwrite()`로 이미지 배열을 파일에 저장하기**

`.imwrite()`를 사용하면 `.imread()`로 읽은 이미지를 RGB로 변환하지 않아도 RGB로 저장할 수 있음.

## OpenCV 영상 처리

**1. VideoCapture와 VideoWriter**
<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132186695-345049a4-6bdf-41f8-ba86-78d2bb839e1f.GIF width = 800></p>

- `cap = cv2.VideoCapture(video_input_path)` 로 입력받은 video의 다양한 속성을 가져오고, `cap.read()`로 모든 frame을 읽음.
- `vid_writer = cv2.VideoWriter(video_output_path, codec, vid_fps, vid_size)` 로 frame을 특정 포맷의 동영상으로 write 할 수 있음.

**2. Frame 별 Object Detection**

<p align = "center"><img src = https://user-images.githubusercontent.com/89925976/132188633-dc114b0b-cfa8-4aee-8db7-3d5950f2714b.png width = 600></p>

- while loop로 모든 frame에서 Object Detection을 수행하고, 캡션으로 프레임 수를 출력함. (Bounding box와 프레임 캡션을 시각화)
