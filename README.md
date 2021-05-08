# autonomous_driving_mid_project
## 차량지능기초 전반부 프로젝트 
Mediapipe를 활용한 단일 2D 카메라를 이용한 근거리 인물 인식 및 주의 전달

### Setup
```
pip install -r requirements.txt
```

### run
mediapipe_human_detect.py를 실행하면 동영상 입력 창이 실행됩니다.  
동영상 파일을 선택하면 프로세스가 실행되고, annotated 영상이 만들어집니다.
```
python3 mediapipe_human_detect.py
```

## Introduction
 본 프로젝트에서는 mediapipe의 pose와 face detection을 활용하여  
다른 센서 없이 2d 카메라를 이용해 근거리에서 인물을 탐지하고 시스템에 주의할 것을 전달하는 시스템의 구현을 목표로 합니다.  
 저속 주행이나 주차시 등 근거리 인물 탐지가 필요한 상황, 특히 벽이나 다른 주차되어있는 차량이 아닌 인물만을 탐지해야 할 경우 활용 가능합니다.
 
 ***
 mediapipe의 pose를 활용해 인물의 전신을 인식하고, face detection을 한번 더 사용해 미처 인식하지 못한 인물이나 상반신 중 일부만 보이는 인물,  
 혹은 아동을 추가로 인식하고 일정 크기 이상일 경우 카메라에 근접한 것으로 인지해 시스템에 주의할 것을 전달합니다.
 
 이 때, 화면과 신체의 크기 비율을 계산하여 근접 여부를 결정하는데,  
 신체가 작은 아동의 경우 pose만으로 계산하면 threshold를 넘지 못하는 경우가 발생 할 수 있으므로  
 성인과 아동의 신장 길이 차이보다 머리 크기 차이 비율이 더 적은 것에 착안해 
 얼굴 영역을 활용해 추가로 근접 여부를 결정하도록 하였습니다.  
 > #### 6세와 17세의 평균 신장, 평균 머리 길이 차이  
 > 평균 신장 차이: 남성 53cm, 여성 41cm  
 > 평균 머리 길이 차이: 남성 2.762cm, 여성 1.206cm  
 > (출처: 이영숙. (1999). 한국 청소년의 신체 성장 특징과 체격 변화(제1보). 해부·생물인류학, 12(1), 175-186.)

## TODO
mediapipe_human_detect.py 코드 중 warning 함수는  
현재 영상 중 주의할 장면의 프레임을 저장하는 것으로 구현해 두었습니다.  
해당 함수를 수정하여 활용할 수 있습니다.
```python
def warning(data):
    # TODO: 경고 전송
    os.makedirs(data['output_dir'], exist_ok=True)
    cv2.imwrite(os.path.join(data['output_dir'], '{}.png'.format(data['i'])), data['image'])
```
