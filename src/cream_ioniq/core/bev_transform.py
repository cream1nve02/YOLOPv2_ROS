#!/usr/bin/env python3
"""
버드아이뷰 변환 모듈

카메라 이미지를 위에서 내려다본 시점(BEV)으로 변환하는 모듈임.
미리 캘리브레이션된 설정 파일을 사용해서 실시간으로 변환을 수행함.
차선 검출에서 거리 측정과 곡률 계산이 정확해지는 장점이 있음.
"""

import json
import cv2
import numpy as np
import os
from typing import Optional, Tuple, Dict, Any


class BEVTransform:
    """버드아이뷰 변환을 담당하는 클래스
    
    카메라 캘리브레이션 정보와 변환 매트릭스를 사용해서
    일반 카메라 이미지를 위에서 내려다본 시점으로 바꿔주는 역할함.
    """
    
    def __init__(self, config_path: str):
        """BEV 변환기 초기화
        
        Args:
            config_path: 캘리브레이션 설정 파일 경로
        """
        # 설정 관련 변수들
        self.config: Optional[Dict[str, Any]] = None
        self.transformation_matrix: Optional[np.ndarray] = None
        self.bev_size: Optional[Tuple[int, int]] = None
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.real_scale: Optional[float] = None
        
        # 설정 파일 불러오기
        self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        """설정 파일을 읽어서 변환에 필요한 정보들을 준비함
        
        Args:
            config_path: JSON 설정 파일 경로
            
        Raises:
            FileNotFoundError: 설정 파일이 없을 때
            KeyError: 필수 키가 설정 파일에 없을 때
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"BEV 설정 파일을 찾을 수 없음: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # 변환 매트릭스 준비
            self.transformation_matrix = np.array(
                self.config['transformation_matrix'], dtype=np.float32
            )
            
            # BEV 출력 크기 설정
            bev_config = self.config['bev_output_size']
            self.bev_size = (bev_config['width'], bev_config['height'])
            
            # 카메라 파라미터 설정
            cam_params = self.config['camera_params']
            self.camera_matrix = np.array(cam_params['camera_matrix'], dtype=np.float32)
            self.dist_coeffs = np.array(cam_params['dist_coeffs'], dtype=np.float32)
            
            # 실제 스케일 정보 (미터/픽셀)
            self.real_scale = float(self.config['real_world_scale'])
            
            print(f"BEV 변환 설정 로드 완료: {self.bev_size[0]}x{self.bev_size[1]} "
                  f"스케일: {self.real_scale:.4f}m/px")
                  
        except KeyError as e:
            raise KeyError(f"설정 파일에 필수 키가 없음: {e}")
        except Exception as e:
            raise RuntimeError(f"설정 파일 로드 중 오류 발생: {e}")
    
    def is_loaded(self) -> bool:
        """변환기가 제대로 초기화되었는지 확인
        
        Returns:
            초기화 성공 여부
        """
        return self.transformation_matrix is not None
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """카메라 왜곡 보정을 수행함
        
        카메라 렌즈로 인한 왜곡을 제거해서 
        더 정확한 BEV 변환이 가능하도록 함.
        
        Args:
            image: 원본 카메라 이미지
            
        Returns:
            왜곡이 보정된 이미지
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
        
        h, w = image.shape[:2]
        
        # 최적 카메라 매트릭스 계산
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        # 왜곡 보정 수행
        undistorted = cv2.undistort(
            image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix
        )
        
        # ROI 적용 (필요한 부분만 잘라냄)
        x, y, roi_w, roi_h = roi
        if roi_w > 0 and roi_h > 0:
            undistorted = undistorted[y:y+roi_h, x:x+roi_w]
        
        return undistorted
    
    def transform_to_bev(self, image: np.ndarray, apply_undistort: bool = True) -> Optional[np.ndarray]:
        """이미지를 버드아이뷰로 변환함
        
        Args:
            image: 입력 이미지
            apply_undistort: 왜곡 보정 적용 여부
            
        Returns:
            BEV 변환된 이미지 (실패시 None)
        """
        if not self.is_loaded():
            print("BEV 변환기가 초기화되지 않음")
            return None
        
        # 왜곡 보정 (옵션)
        if apply_undistort:
            image = self.undistort_image(image)
        
        # BEV 변환 수행
        try:
            bev_image = cv2.warpPerspective(
                image, 
                self.transformation_matrix, 
                self.bev_size,
                flags=cv2.INTER_LINEAR
            )
            return bev_image
            
        except Exception as e:
            print(f"BEV 변환 실패: {e}")
            return None
    
    def get_config_info(self) -> Optional[Dict[str, Any]]:
        """현재 설정 정보를 반환함
        
        다른 모듈에서 BEV 관련 정보가 필요할 때 사용함.
        
        Returns:
            설정 정보 딕셔너리 (로드 안된 경우 None)
        """
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
    
    def pixel_to_real_coords(self, bev_x: int, bev_y: int) -> Tuple[float, float]:
        """BEV 픽셀 좌표를 실제 미터 단위로 변환함
        
        Args:
            bev_x: BEV 이미지의 X 픽셀 좌표
            bev_y: BEV 이미지의 Y 픽셀 좌표
            
        Returns:
            (실제 X 좌표(m), 실제 Y 좌표(m))
        """
        if self.bev_size is None or self.real_scale is None:
            return (0.0, 0.0)
        
        # BEV 이미지에서 실제 좌표계로 변환
        # Y=0이 이미지 상단, X=0이 이미지 왼쪽
        real_x = (self.bev_size[1] - bev_y) * self.real_scale  # 전방 거리
        real_y = (bev_x - self.bev_size[0]/2) * self.real_scale  # 좌우 위치
        
        return (real_x, real_y)
    
    def real_to_pixel_coords(self, real_x: float, real_y: float) -> Tuple[int, int]:
        """실제 미터 좌표를 BEV 픽셀 좌표로 변환함
        
        Args:
            real_x: 실제 전방 거리(m)
            real_y: 실제 좌우 위치(m)
            
        Returns:
            (BEV X 픽셀, BEV Y 픽셀)
        """
        if self.bev_size is None or self.real_scale is None:
            return (0, 0)
        
        # 실제 좌표계에서 BEV 픽셀로 변환
        bev_x = int(self.bev_size[0]/2 + real_y / self.real_scale)
        bev_y = int(self.bev_size[1] - real_x / self.real_scale)
        
        return (bev_x, bev_y)


def test_bev_transform() -> None:
    """BEV 변환기 테스트 함수
    
    개발/디버깅 목적으로 사용하는 테스트 함수임.
    """
    config_path = "/home/mini/catkin_ws/src/cream_ioniq/bev_config.json"
    
    try:
        bev = BEVTransform(config_path)
        
        if bev.is_loaded():
            print("BEV 변환기 테스트 통과!")
            info = bev.get_config_info()
            if info:
                print(f"실제 영역: {info['real_dimensions']['width_meters']}m x {info['real_dimensions']['height_meters']}m")
                print(f"BEV 크기: {info['bev_size'][0]} x {info['bev_size'][1]}")
        else:
            print("BEV 변환기 초기화 실패")
            
    except Exception as e:
        print(f"테스트 실패: {e}")


if __name__ == "__main__":
    test_bev_transform()