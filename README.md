# CREAM IONIQ 차선 검출 시스템

YOLO 기반 차선 세그멘테이션과 BEV 변환을 활용한 실시간 차선 검출 및 추적 시스템

## 📁 패키지 구조

```
cream_ioniq/
├── config/                          # 설정 파일들
│   └── lane_detection_config.yaml   # 메인 설정 파일
├── launch/                          # 런치 파일들
│   └── cream_ioniq_node.launch      # 메인 런치 파일
├── scripts/                         # 실행 스크립트들
│   ├── cream_ioniq_node.py          # 기존 모놀리식 노드
│   └── cream_ioniq_node_modular.py  # 새로운 모듈형 노드 (권장)
└── src/cream_ioniq/                 # Python 모듈들
    ├── core/                        # 핵심 알고리즘
    │   ├── bev_transform.py         # BEV 변환
    │   └── lane_fitting.py          # 차선 피팅
    ├── visualization/               # 시각화
    │   └── lane_visualizer.py       # RViz 마커 생성
    └── utils/                       # 유틸리티
        └── image_processing.py      # 이미지 전처리
```

## 🚀 사용법

### 1. 모듈형 노드 실행 (권장)
```bash
roslaunch cream_ioniq cream_ioniq_node.launch use_modular:=true
```

### 2. 기존 노드 실행
```bash
roslaunch cream_ioniq cream_ioniq_node.launch use_modular:=false
```

### 3. 커스텀 설정 파일 사용
```bash
roslaunch cream_ioniq cream_ioniq_node.launch config_file:=/path/to/your/config.yaml
```

## 🛠️ 빌드 방법

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## 📊 발행되는 토픽들

- `/lane_seg` - 원본 차선 세그멘테이션
- `/lane_seg_processed` - 전처리된 차선 마스크
- `/lane_seg_bev` - BEV 변환된 차선 마스크
- `/original_bev` - BEV 변환된 원본 이미지
- `/lane_path` - 차선 경로 (nav_msgs/Path)
- `/lane_fitted_bev` - 피팅 결과가 그려진 BEV
- `/lane_markers` - 차선 마커 (RViz용)
- `/lane_boundary_markers` - 차선 경계 마커
- `/lane_info_markers` - 차선 정보 텍스트

## ⚙️ 주요 설정 파라미터

- `yolo.weights_path` - YOLO 모델 경로
- `camera.image_topic` - 입력 카메라 토픽
- `lane_fitting.polynomial_order` - 다항식 차수 (기본: 3)
- `temporal_smoothing.alpha` - 시간 평활화 계수 (기본: 0.7)

## 🎨 특징

- **모듈화된 구조**: 각 기능이 독립적인 모듈로 분리
- **한국어 주석**: 자연스러운 한국어로 작성된 코드와 주석
- **시간적 평활화**: 안정적인 차선 검출을 위한 이동평균 필터링
- **RViz 시각화**: 다양한 마커로 차선 상태를 직관적으로 표시
- **설정 기반**: YAML 파일로 모든 파라미터 조정 가능

## 의존성

- ROS Noetic
- Python 3
- PyTorch
- OpenCV
- NumPy
- scikit-learn
- PyYAML

## 제작자

CREAM IONIQ Team - Chaemin Park
