#!/usr/bin/env python3
"""
CREAM IONIQ 차선 검출 메인 노드

모듈화된 컴포넌트들을 사용해서 차선을 검출하고 
RViz에 시각화하는 통합 노드임.

기능:
- YOLO 기반 차선 세그멘테이션
- BEV 변환 및 차선 피팅
- 시간적 평활화로 안정적인 검출
- ROS 토픽 발행 및 마커 시각화
"""

import rospy
import rospkg
import yaml
import os
import sys
import cv2
import numpy as np
import torch

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge

# 패키지 내 모듈들 불러오기
# Add the YOLOPv2 utils directory to the Python path
yolopv2_ros_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_dir = os.path.join(yolopv2_ros_dir, 'utils')
sys.path.append(utils_dir)

from utils import (
    time_synchronized, select_device, 
    lane_line_mask, driving_area_mask, letterbox
)

sys.path.append(os.path.join(rospkg.RosPack().get_path('cream_ioniq'), 'src'))

from cream_ioniq.core.bev_transform import BEVTransform
from cream_ioniq.core.lane_fitting import LaneFitter
from cream_ioniq.visualization.lane_visualizer import LaneVisualizer
from cream_ioniq.utils.image_processing import preprocess_lane_mask


class CreamIoniqNode:
    """메인 차선 검출 노드 클래스
    
    설정 파일을 읽어서 필요한 컴포넌트들을 초기화하고
    카메라 이미지를 받아서 차선을 검출하는 역할함.
    """
    
    def __init__(self):
        """노드 초기화 및 컴포넌트 설정"""
        rospy.init_node('cream_ioniq_node')
        rospy.loginfo("CREAM IONIQ 차선 검출 시스템 시작")
        
        # 설정 파일 로드
        self.config = self._load_config()
        
        # ROS 관련 초기화
        self.bridge = CvBridge()
        self.frame_id = "base_link"
        
        # YOLO 모델 로드
        self.yolo_model = self._load_yolo_model()
        
        # 컴포넌트 초기화
        self.bev_transform = BEVTransform(self.config['bev_transform']['config_path'])
        self.lane_fitter = LaneFitter(self.config)
        self.visualizer = LaneVisualizer(self.config)
        
        # ROS 퍼블리셔 설정
        self._setup_publishers()
        
        # 카메라 구독자 설정
        self._setup_subscribers()
        
        rospy.loginfo("모든 컴포넌트 초기화 완료")
    
    def _load_config(self):
        """설정 파일을 읽어서 파라미터 로드함"""
        try:
            config_path = rospy.get_param('~config_path', 
                os.path.join(rospkg.RosPack().get_path('cream_ioniq'), 
                           'config/lane_detection_config.yaml'))
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            rospy.loginfo(f"설정 파일 로드: {config_path}")
            return config
            
        except Exception as e:
            rospy.logerr(f"설정 파일 로드 실패: {e}")
            rospy.signal_shutdown("설정 파일 오류")
            return {}
    
    def _load_yolo_model(self):
        """YOLO 모델 로드 및 초기화"""
        try:
            weights_path = self.config['yolo']['weights_path']
            device = self.config['yolo']['device']
            use_half = self.config['yolo']['use_half_precision']
            
            rospy.loginfo(f"YOLO 모델 로드 중: {weights_path}")
            
            # 디바이스 설정 (YOLOPv2 방식 사용)
            self.device = select_device(device)
            
            # 모델 로드
            model = torch.jit.load(weights_path)
            model = model.to(self.device)
            
            # Half precision 설정
            self.half = self.device.type != 'cpu'
            if self.half and use_half:
                model.half()
                rospy.loginfo("Half precision 모드 활성화")
            
            model.eval()
            
            # 모델 초기화를 위한 더미 입력
            if self.device.type != 'cpu':
                model(torch.zeros(1, 3, self.config['yolo']['input_size'], self.config['yolo']['input_size']).to(self.device).type_as(next(model.parameters())))
            
            rospy.loginfo(f"YOLO 모델 로드 완료 (device: {device})")
            return model
            
        except Exception as e:
            rospy.logerr(f"YOLO 모델 로드 실패: {e}")
            rospy.signal_shutdown("모델 로드 오류")
            return None
    
    def _setup_publishers(self):
        """ROS 퍼블리셔들 초기화"""
        topics = self.config['topics']['publishers']
        
        self.publishers = {
            'lane_seg': rospy.Publisher(topics['lane_segmentation'], Image, queue_size=1),
            'lane_seg_processed': rospy.Publisher(topics['lane_segmentation_processed'], Image, queue_size=1),
            'lane_seg_bev': rospy.Publisher(topics['lane_segmentation_bev'], Image, queue_size=1),
            'original_bev': rospy.Publisher(topics['original_bev'], Image, queue_size=1),
            'lane_path': rospy.Publisher(topics['lane_path'], Path, queue_size=1),
            'lane_fitted_bev': rospy.Publisher(topics['lane_fitted_bev'], Image, queue_size=1),
            'lane_markers': rospy.Publisher(topics['lane_markers'], MarkerArray, queue_size=1),
            'lane_boundary_markers': rospy.Publisher(topics['lane_boundary_markers'], MarkerArray, queue_size=1),
            'lane_info_markers': rospy.Publisher(topics['lane_info_markers'], MarkerArray, queue_size=1)
        }
        
        rospy.loginfo("퍼블리셔 설정 완료")
    
    def _setup_subscribers(self):
        """카메라 구독자 설정"""
        image_topic = self.config['camera']['image_topic']
        use_compressed = self.config['camera']['use_compressed']
        
        if use_compressed:
            self.image_sub = rospy.Subscriber(
                image_topic, CompressedImage, self._compressed_image_callback, queue_size=1
            )
            rospy.loginfo(f"압축 이미지 구독: {image_topic}")
        else:
            self.image_sub = rospy.Subscriber(
                image_topic, Image, self._image_callback, queue_size=1
            )
            rospy.loginfo(f"일반 이미지 구독: {image_topic}")
    
    def _compressed_image_callback(self, msg):
        """압축된 이미지 콜백 함수"""
        try:
            # 압축 이미지를 OpenCV 형태로 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                self._process_image(cv_image, msg.header)
            
        except Exception as e:
            rospy.logwarn(f"압축 이미지 처리 오류: {e}")
    
    def _image_callback(self, msg):
        """일반 이미지 콜백 함수"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self._process_image(cv_image, msg.header)
            
        except Exception as e:
            rospy.logwarn(f"이미지 처리 오류: {e}")
    
    def _process_image(self, cv_image, header):
        """메인 이미지 처리 파이프라인"""
        try:
            # 원본 이미지 저장 (BEV용, 기존 코드 방식)
            original_for_bev = cv_image.copy()
            if original_for_bev.shape[1] != 1920 or original_for_bev.shape[0] != 1080:
                original_for_bev = cv2.resize(original_for_bev, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            
            # 2단계: YOLO 세그멘테이션
            seg_mask = self._run_yolo_segmentation(cv_image)
            if seg_mask is None:
                return
            
            # 세그멘테이션 결과 발행
            self._publish_image(self.publishers['lane_seg'], seg_mask, header, 'mono8')
            
            # 3단계: 마스크 전처리
            processed_mask = preprocess_lane_mask(
                seg_mask,
                aggressive_removal=self.config['image_preprocessing']['aggressive_horizontal_removal'],
                apply_thinning=self.config['image_preprocessing']['apply_thinning']
            )
            
            # 전처리된 마스크 발행
            self._publish_image(self.publishers['lane_seg_processed'], processed_mask, header, 'mono8')
            
            # 4단계: BEV 변환 (기존 코드 방식으로 처리된 마스크 사용)
            # 차선 마스크를 1920x1080으로 크기 조정
            processed_mask_1920 = cv2.resize(processed_mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
            
            # BEV 변환
            original_bev = self.bev_transform.transform_to_bev(
                original_for_bev, 
                apply_undistort=self.config['bev_transform']['apply_undistortion']
            )
            seg_bev = self.bev_transform.transform_to_bev(processed_mask_1920, apply_undistort=True)
            
            if original_bev is None or seg_bev is None:
                rospy.logwarn("BEV 변환 실패")
                return
            
            # BEV 이미지들 발행
            self._publish_image(self.publishers['original_bev'], original_bev, header)
            self._publish_image(self.publishers['lane_seg_bev'], seg_bev, header, 'mono8')
            
            # 5단계: 차선 피팅 및 시각화
            self._fit_and_visualize_lanes(seg_bev, original_bev, header)
            
        except Exception as e:
            rospy.logwarn(f"이미지 처리 파이프라인 오류: {e}")
    
    def _run_yolo_segmentation(self, image):
        """YOLO 모델로 차선 세그멘테이션 수행 (기존 코드 방식)"""
        try:
            input_size = self.config['yolo']['input_size']
            
            # YOLO 추론용 이미지 크기 조정 (기존 코드 방식)
            im0s = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)
            
            # Letterbox 처리 (기존 코드 방식)
            img, ratio, (dw, dh) = letterbox(im0s, input_size, stride=32)
            
            # BGR -> RGB, HWC -> CHW
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            
            # 텐서 변환
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # 추론 (기존 코드 방식)
            t1 = time_synchronized()
            [pred, anchor_grid], seg, ll = self.yolo_model(img)
            t2 = time_synchronized()
            
            if self.config['debug']['enable_debug_logs']:
                rospy.loginfo(f"YOLO 추론 시간: {(t2-t1)*1000:.1f}ms")
                rospy.loginfo(f"출력 형태 - pred: {pred.shape if hasattr(pred, 'shape') else type(pred)}")
                rospy.loginfo(f"출력 형태 - seg: {seg.shape if hasattr(seg, 'shape') else type(seg)}")  
                rospy.loginfo(f"출력 형태 - ll: {ll.shape if hasattr(ll, 'shape') else type(ll)}")
            
            # 세그멘테이션 마스크 추출 (기존 코드 방식)
            ll_seg_mask = lane_line_mask(ll)
            
            # 마스크를 8비트 이미지로 변환
            ll_seg_mask_8u = np.zeros(ll_seg_mask.shape, dtype=np.uint8)
            ll_seg_mask_8u[ll_seg_mask > 0] = 255
            
            # 원본 이미지 크기로 복원
            original_height, original_width = image.shape[:2]
            seg_mask = cv2.resize(ll_seg_mask_8u, (original_width, original_height))
            
            # 최종 마스크 상태 확인
            if self.config['debug']['enable_debug_logs']:
                white_pixels = cv2.countNonZero(seg_mask)
                total_pixels = seg_mask.shape[0] * seg_mask.shape[1]
                percentage = (white_pixels / total_pixels) * 100
                rospy.loginfo(f"최종 세그멘테이션 마스크: {white_pixels}/{total_pixels} ({percentage:.2f}%) 흰색 픽셀")
            
            return seg_mask
            
        except Exception as e:
            rospy.logwarn(f"YOLO 추론 실패: {e}")
            return None
    
    def _fit_and_visualize_lanes(self, seg_bev, original_bev, header):
        """차선 피팅 및 시각화 수행"""
        try:
            # BEV 스케일 정보 가져오기
            bev_config = self.bev_transform.get_config_info()
            if bev_config is None:
                rospy.logwarn("BEV 설정 정보 없음")
                return
            
            real_scale = bev_config['real_scale']
            
            # BEV 마스크에서 차선 점들 추출
            from cream_ioniq.utils.image_processing import extract_lane_points_from_bev
            lane_points = extract_lane_points_from_bev(seg_bev, real_scale)
            
            # 차선 피팅 수행
            fitting_result = self.lane_fitter.fit_lanes(lane_points)
            
            if fitting_result is None or fitting_result[0] is None:
                rospy.loginfo("차선 검출 실패")
                # 빈 마커 배열 발행 (이전 마커들 정리)
                empty_markers = MarkerArray()
                self.publishers['lane_markers'].publish(empty_markers)
                self.publishers['lane_boundary_markers'].publish(empty_markers)
                self.publishers['lane_info_markers'].publish(empty_markers)
                return
            
            x_pred, y_pred_l, y_pred_r = fitting_result
            
            # Path 메시지 생성 및 발행
            path_msg = self.lane_fitter.create_path_message(x_pred, y_pred_l, y_pred_r, header)
            self.publishers['lane_path'].publish(path_msg)
            
            # 시각화 마커들 생성
            markers = self.visualizer.create_lane_markers(
                x_pred, y_pred_l, y_pred_r, rospy.Time.now(), header.frame_id
            )
            
            boundary_markers = self.visualizer.create_lane_boundary_markers(
                x_pred, y_pred_l, y_pred_r, rospy.Time.now(), header.frame_id
            )
            
            info_markers = self.visualizer.create_lane_info_text_marker(
                x_pred, y_pred_l, y_pred_r, rospy.Time.now(), header.frame_id
            )
            
            # 마커들 발행
            self.publishers['lane_markers'].publish(markers)
            self.publishers['lane_boundary_markers'].publish(boundary_markers)
            self.publishers['lane_info_markers'].publish(info_markers)
            
            # BEV에 피팅 결과 그리기
            from cream_ioniq.utils.image_processing import draw_fitted_lanes_on_bev
            fitted_bev = draw_fitted_lanes_on_bev(original_bev, x_pred, y_pred_l, y_pred_r, real_scale)
            self._publish_image(self.publishers['lane_fitted_bev'], fitted_bev, header)
            
            if self.config['debug']['enable_debug_logs']:
                left_count = len(y_pred_l) if y_pred_l is not None else 0
                right_count = len(y_pred_r) if y_pred_r is not None else 0
                rospy.loginfo(f"차선 검출 성공: 좌측 {left_count}개, 우측 {right_count}개 점")
            
        except Exception as e:
            rospy.logwarn(f"차선 피팅/시각화 오류: {e}")
    
    def _publish_image(self, publisher, image, header, encoding="bgr8"):
        """이미지를 ROS 토픽으로 발행"""
        try:
            # 헤더 설정
            img_msg = self.bridge.cv2_to_imgmsg(image, encoding)
            img_msg.header = header
            img_msg.header.frame_id = self.frame_id
            
            # 발행
            publisher.publish(img_msg)
            
        except Exception as e:
            rospy.logwarn(f"이미지 발행 실패: {e}")
    
    def run(self):
        """노드 실행 루프"""
        rospy.loginfo("차선 검출 시스템 가동 중...")
        
        # 성능 모니터링 (옵션)
        if self.config['debug']['measure_performance']:
            rate = rospy.Rate(10)  # 10Hz로 성능 체크
            
            while not rospy.is_shutdown():
                # 메모리 사용량이나 처리 시간 등을 로그로 출력할 수 있음
                rate.sleep()
        else:
            # 일반 실행
            rospy.spin()


def main():
    """메인 함수"""
    try:
        node = CreamIoniqNode()
        node.run()
        
    except KeyboardInterrupt:
        rospy.loginfo("사용자 종료 요청")
    except Exception as e:
        rospy.logerr(f"노드 실행 중 오류: {e}")
    finally:
        rospy.loginfo("CREAM IONIQ 시스템 종료")


if __name__ == '__main__':
    main()