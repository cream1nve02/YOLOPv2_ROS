#!/usr/bin/env python3
"""
Bird's Eye View Transformation Module
사용법: bev_config.json을 로드해서 실시간 버드아이뷰 변환 수행
yolopv2_node에서 import해서 사용할 수 있습니다.
"""

import json
import cv2
import numpy as np
import os

class BEVTransform:
    def __init__(self, config_path="/home/mini/catkin_ws/src/YOLOPv2_ROS/bev_config.json"):
        """
        BEV 변환 클래스 초기화
        
        Args:
            config_path (str): bev_config.json 파일 경로
        """
        self.config = None
        self.transformation_matrix = None
        self.bev_size = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.real_scale = None
        
        self.load_config(config_path)
    
    def load_config(self, config_path):
        """BEV 설정 파일 로드"""
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # 변환 매트릭스 로드
            self.transformation_matrix = np.array(self.config['transformation_matrix'])
            
            # 출력 크기 설정
            self.bev_size = (
                self.config['bev_output_size']['width'],
                self.config['bev_output_size']['height']
            )
            
            # 카메라 매개변수 로드
            self.camera_matrix = np.array(self.config['camera_params']['camera_matrix'])
            self.dist_coeffs = np.array(self.config['camera_params']['dist_coeffs'])
            
            # 실제 스케일 정보
            self.real_scale = self.config['real_world_scale']
            
            print(f"BEV config loaded successfully:")
            print(f"- Real scale: {self.real_scale} m/pixel")
            print(f"- Real dimensions: {self.config['real_dimensions']['width_meters']:.2f}m x {self.config['real_dimensions']['height_meters']:.2f}m")
            print(f"- Output size: {self.bev_size[0]}x{self.bev_size[1]}")
            
            return True
            
        except Exception as e:
            print(f"Error loading BEV config: {e}")
            print("Please run points.py first to generate bev_config.json")
            return False
    
    def undistort_image(self, image):
        """
        이미지 왜곡 보정 및 ROI 적용 (points.py와 동일한 처리)
        
        Args:
            image: 입력 이미지
            
        Returns:
            왜곡 보정 및 ROI 적용된 이미지
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
        
        # 새로운 카메라 매트릭스 계산 (points.py와 동일)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, 
            (image.shape[1], image.shape[0]), 1, 
            (image.shape[1], image.shape[0])
        )
        
        # 왜곡 보정
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        
        # ROI 적용 (points.py와 동일한 크롭핑)
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def transform_to_bev(self, image, apply_undistort=True):
        """
        이미지를 버드아이뷰로 변환
        
        Args:
            image: 입력 이미지 (원본 카메라 이미지)
            apply_undistort (bool): 왜곡 보정 적용 여부
            
        Returns:
            버드아이뷰 변환된 이미지
        """
        if self.transformation_matrix is None:
            print("Error: BEV configuration not loaded")
            return None
        
        # 왜곡 보정 적용 (선택사항)
        if apply_undistort:
            image = self.undistort_image(image)
        
        # 버드아이뷰 변환 적용
        bev_image = cv2.warpPerspective(image, self.transformation_matrix, self.bev_size)
        
        return bev_image
    
    def get_real_coordinates(self, pixel_x, pixel_y):
        """
        BEV 이미지의 픽셀 좌표를 실제 좌표(미터)로 변환
        
        Args:
            pixel_x, pixel_y: BEV 이미지에서의 픽셀 좌표
            
        Returns:
            (real_x, real_y): 실제 좌표 (미터)
        """
        if self.real_scale is None:
            return None, None
        
        # BEV 이미지의 중심을 기준점으로 설정
        center_x = self.bev_size[0] // 2
        center_y = self.bev_size[1] // 2
        
        # 픽셀 좌표를 실제 좌표로 변환
        real_x = (pixel_x - center_x) * self.real_scale
        real_y = (center_y - pixel_y) * self.real_scale  # Y축 반전 (이미지 좌표계 -> 실제 좌표계)
        
        return real_x, real_y
    
    def get_pixel_coordinates(self, real_x, real_y):
        """
        실제 좌표(미터)를 BEV 이미지의 픽셀 좌표로 변환
        
        Args:
            real_x, real_y: 실제 좌표 (미터)
            
        Returns:
            (pixel_x, pixel_y): BEV 이미지에서의 픽셀 좌표
        """
        if self.real_scale is None:
            return None, None
        
        center_x = self.bev_size[0] // 2
        center_y = self.bev_size[1] // 2
        
        pixel_x = int(center_x + real_x / self.real_scale)
        pixel_y = int(center_y - real_y / self.real_scale)  # Y축 반전
        
        return pixel_x, pixel_y
    
    def is_loaded(self):
        """설정이 제대로 로드되었는지 확인"""
        return self.transformation_matrix is not None
    
    def get_config_info(self):
        """설정 정보 반환"""
        if self.config is None:
            return None
        
        return {
            'real_scale': self.real_scale,
            'real_dimensions': self.config['real_dimensions'],
            'bev_size': self.bev_size,
            'source_points': self.config['source_points'],
            'destination_points': self.config['destination_points'],
            'transformation_matrix': self.transformation_matrix.tolist()
        }

# 사용 예제 함수들
def test_bev_transform():
    """BEV 변환 테스트"""
    # BEV 변환기 초기화
    bev_transformer = BEVTransform()
    
    if not bev_transformer.is_loaded():
        print("BEV configuration not loaded. Please run points.py first.")
        return
    
    # 샘플 이미지가 있다면 테스트
    sample_image_path = "/home/mini/catkin_ws/src/YOLOPv2_ROS/bev_sample.jpg"
    if os.path.exists(sample_image_path):
        # 원본 이미지 로드 (BEV 변환 전 이미지를 시뮬레이션)
        print("Testing with sample image...")
        
        # 실제로는 카메라에서 받은 원본 이미지를 사용
        # 여기서는 테스트용으로 임시 이미지 생성
        test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        test_image.fill(100)  # 회색 배경
        
        # BEV 변환 적용
        bev_result = bev_transformer.transform_to_bev(test_image)
        
        if bev_result is not None:
            cv2.imwrite("/home/mini/catkin_ws/src/YOLOPv2_ROS/test_bev_result.jpg", bev_result)
            print("BEV transformation test completed. Result saved as test_bev_result.jpg")
        else:
            print("BEV transformation failed")
    else:
        print("No sample image found. Run points.py first to generate sample images.")

if __name__ == "__main__":
    test_bev_transform()