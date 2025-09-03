import cv2
import rospy
import numpy as np
import json
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# 전역 변수 초기화
points = []
paused_frame = None
bridge = CvBridge()

# 실제 거리 설정 (여기서 수정하세요)
REAL_WIDTH = 7.4    # 실제 가로 길이 (미터)
REAL_HEIGHT = 20.0  # 실제 세로 길이 (미터)
BEV_HEIGHT = 800    # BEV 출력 세로 픽셀

# 원본 해상도 설정
ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1080

# 카메라 매트릭스
camera_matrix = np.array([
    [976.8601881, 0.0, 944.38126699],
    [0.0, 979.28933183, 588.12563818],
    [0.0, 0.0, 1.0]
])

# 왜곡 계수
dist_coeffs = np.array([-0.36123691, 0.16505182, -0.00177654, 0.00029295, 0.0])

# 새로운 카메라 매트릭스
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT), 1, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT)
)

def calculate_bev_parameters():
    """BEV 파라미터 계산"""
    real_width = REAL_WIDTH
    real_height = REAL_HEIGHT
    bev_height = BEV_HEIGHT
    
    # 실제 비율에 맞춰서 BEV 가로 픽셀 자동 계산
    aspect_ratio = real_width / real_height
    bev_width = int(bev_height * aspect_ratio)
    
    # 실제 스케일 계산
    scale_x = real_width / bev_width
    scale_y = real_height / bev_height  
    real_scale = (scale_x + scale_y) / 2
    
    print(f"실제 영역: {real_width}m x {real_height}m")
    print(f"BEV 크기: {bev_width} x {bev_height} 픽셀")
    print(f"계산된 스케일: {real_scale:.4f} m/px")
    
    return real_scale, real_width, real_height, bev_width, bev_height

def create_bev_config_and_preview(real_scale, real_width, real_height, bev_width, bev_height):
    """BEV config 파일 생성"""
    
    # 목적지 포인트 (전체 BEV 영역 사용)
    dst_points = [
        [0, 0],
        [bev_width, 0],
        [bev_width, bev_height],
        [0, bev_height]
    ]
    
    src_points = np.array(points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    # 원근 변환 매트릭스 계산
    transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # BEV 설정 딕셔너리
    bev_config = {
        "transformation_matrix": transformation_matrix.tolist(),
        "source_points": points,
        "destination_points": dst_points.tolist(),
        "real_world_scale": real_scale,
        "real_dimensions": {
            "width_meters": real_width,
            "height_meters": real_height
        },
        "bev_output_size": {
            "width": bev_width,
            "height": bev_height
        },
        "camera_params": {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist()
        }
    }
    
    # JSON 파일로 저장
    config_path = "/home/mini/catkin_ws/src/YOLOPv2_ROS/bev_config.json"
    with open(config_path, 'w') as f:
        json.dump(bev_config, f, indent=2)
    
    print(f"BEV 설정 저장: {config_path}")
    
    # BEV 변환 결과 미리보기
    if paused_frame is not None:
        source_image = paused_frame
    else:
        # 더미 이미지 생성 (카메라 없을 때)
        print("카메라 이미지가 없어 더미 이미지를 사용합니다.")
        source_image = np.zeros((ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 3), dtype=np.uint8)
        source_image.fill(50)
        cv2.rectangle(source_image, (300, 400), (1620, 950), (80, 80, 80), -1)
        cv2.line(source_image, (600, 400), (700, 950), (255, 255, 255), 3)
        cv2.line(source_image, (960, 400), (960, 950), (255, 255, 255), 3)
        cv2.line(source_image, (1320, 400), (1220, 950), (255, 255, 255), 3)
    
    # BEV 변환 수행
    bev_image = cv2.warpPerspective(source_image, transformation_matrix, (bev_width, bev_height))
    
    
    # BEV 결과 이미지 표시
    try:
        cv2.namedWindow("BEV Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("BEV Result", 800, 600)
        cv2.imshow("BEV Result", bev_image)
        print("BEV 변환 결과 확인. 아무 키나 누르면 종료...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"창 표시 오류: {e}")
        print("이미지 파일로만 저장됩니다.")
    
    # 결과 이미지 저장
    save_path = "/home/mini/catkin_ws/src/YOLOPv2_ROS/bev_result.jpg"
    cv2.imwrite(save_path, bev_image)
    print(f"BEV 결과 이미지 저장: {save_path}")

def mouse_callback(event, x, y, flags, param):
    global points, paused_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) == 0:
            paused_frame = param.copy()
            print(f"이미지 캡처됨: {paused_frame.shape}")
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)} selected at: ({x}, {y})")
        if len(points) == 4:
            print("4개 포인트 선택 완료!")
            rospy.signal_shutdown("Points selected")

def image_callback(msg):
    global paused_frame
    if paused_frame is None:
        try:
            frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Could not convert image: {e}")
            return

        if frame.shape[1] != ORIGINAL_WIDTH or frame.shape[0] != ORIGINAL_HEIGHT:
            frame = cv2.resize(frame, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))

        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        x, y, w, h = roi
        undistorted_frame = undistorted_frame[y:y+h, x:x+w]

        cv2.imshow("Frame", undistorted_frame)
        cv2.setMouseCallback("Frame", mouse_callback, undistorted_frame)
    else:
        cv2.imshow("Frame", paused_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User exit")

def main():
    global points
    
    print("=== BEV 캘리브레이션 도구 ===")
    print("4개 포인트를 순서대로 클릭하세요 (좌상 -> 우상 -> 우하 -> 좌하)")
    print("카메라 영상을 기다리는 중...")
    
    rospy.init_node('compressed_image_listener', anonymous=True)
    rospy.Subscriber("/gmsl_camera/dev/video1/compressed", CompressedImage, image_callback)
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    
    cv2.destroyAllWindows()
    
    # 4개 포인트가 선택되었으면 config 생성
    if len(points) == 4:
        real_scale, real_width, real_height, bev_width, bev_height = calculate_bev_parameters()
        create_bev_config_and_preview(real_scale, real_width, real_height, bev_width, bev_height)
        print("BEV 캘리브레이션 완료!")
    else:
        print(f"포인트 선택이 완료되지 않았습니다. ({len(points)}/4)")

if __name__ == '__main__':
    main()