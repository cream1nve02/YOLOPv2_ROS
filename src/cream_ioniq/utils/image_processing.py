#!/usr/bin/env python3
"""
이미지 처리 유틸리티

차선 검출에 필요한 이미지 전처리, 마스크 정리, 좌표 변환 등
공통으로 사용되는 이미지 처리 함수들을 모아놓은 모듈임.
"""

import cv2
import numpy as np
import rospy
from typing import Tuple, Optional


def extract_lane_points_from_bev(bev_mask: np.ndarray, real_scale: float) -> np.ndarray:
    """BEV 마스크에서 실제 좌표계 차선 점들을 추출함
    
    세그멘테이션 마스크의 흰색 픽셀들을 찾아서
    실제 미터 단위 좌표로 변환하는 함수임.
    
    Args:
        bev_mask: BEV 변환된 차선 마스크 (그레이스케일)
        real_scale: 미터/픽셀 변환 비율
        
    Returns:
        4xN 형태의 차선 점들 [x, y, z, w] (실제 좌표계)
    """
    # 마스크에 흰색 픽셀이 없으면 빈 결과 반환
    if cv2.countNonZero(bev_mask) == 0:
        return np.zeros((4, 10))
    
    # 흰색 픽셀들의 좌표 찾기
    lane_pixels = cv2.findNonZero(bev_mask)
    if lane_pixels is None:
        return np.zeros((4, 10))
    
    lane_pixels = lane_pixels.reshape([-1, 2])  # (N, 2) 형태로 변환
    
    # BEV 픽셀 좌표를 실제 좌표계로 변환
    bev_height, bev_width = bev_mask.shape
    
    # BEV 좌표계 변환
    # Y=0이 이미지 상단(멀리), Y=height가 하단(가까이)
    # X=0이 이미지 왼쪽, X=width/2가 중앙
    x_real = (bev_height - lane_pixels[:, 1]) * real_scale  # 전방 거리
    y_real = (bev_width/2 - lane_pixels[:, 0]) * real_scale  # 좌우 위치 (중앙 기준)
    
    # 유효한 범위의 점들만 선택 (앞쪽 20m, 좌우 3m 이내)
    valid_mask = (x_real > 0) & (x_real < 20) & (np.abs(y_real) < 3)
    x_real = x_real[valid_mask]
    y_real = y_real[valid_mask]
    
    if len(x_real) == 0:
        return np.zeros((4, 10))
    
    # 4xN 형태로 변환 (동차 좌표)
    xyz_points = np.array([
        x_real,                    # X: 전방 거리
        y_real,                    # Y: 좌우 위치
        np.zeros_like(x_real),     # Z: 높이 (지면 기준)
        np.ones_like(x_real)       # W: 동차 좌표
    ])
    
    return xyz_points


def aggressive_horizontal_removal(mask: np.ndarray) -> np.ndarray:
    """매우 공격적인 가로선 제거
    
    정지선이나 횡단보도 같은 가로 방향의 선들을 
    강력하게 제거하는 함수임. 차선 검출 정확도 향상을 위해 사용함.
    
    Args:
        mask: 입력 마스크 (이진 이미지)
        
    Returns:
        가로선이 제거된 마스크
    """
    try:
        filtered_mask = mask.copy()
        h, w = mask.shape
        
        # 행별로 스캔해서 가로선 검출 및 제거
        for y in range(h):
            row = mask[y, :]
            white_pixels = np.sum(row > 0)
            
            # 해당 행에서 흰 픽셀이 너무 많으면 가로선으로 판단
            if white_pixels > w * 0.15:  # 행의 15% 이상이 흰색
                # 연속된 흰 픽셀 구간들 찾기
                white_regions = []
                start = None
                
                for x in range(w):
                    if row[x] > 0:
                        if start is None:
                            start = x
                    else:
                        if start is not None:
                            white_regions.append((start, x-1))
                            start = None
                            
                if start is not None:
                    white_regions.append((start, w-1))
                
                # 긴 가로 구간들 제거 (30픽셀 이상)
                for start_x, end_x in white_regions:
                    length = end_x - start_x + 1
                    if length > 30:
                        filtered_mask[y, start_x:end_x+1] = 0
        
        # 세로 방향으로만 연결된 픽셀 보존
        vertical_kernel = np.array([[0, 1, 0],
                                   [0, 1, 0],
                                   [0, 1, 0]], dtype=np.uint8)
        vertical_preserved = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, vertical_kernel)
        
        # 대각선 방향도 어느정도 보존
        diagonal_kernel = np.array([[1, 0, 0],
                                   [0, 1, 0], 
                                   [0, 0, 1]], dtype=np.uint8)
        diagonal_preserved = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, diagonal_kernel)
        
        # 세로선과 대각선 결합
        result = cv2.bitwise_or(vertical_preserved, diagonal_preserved)
        
        return result
        
    except Exception as e:
        rospy.logwarn(f"가로선 제거 중 오류: {e}")
        return mask


def filter_horizontal_lines(
    mask: np.ndarray, 
    horizontal_ratio_threshold: float = 2.5, 
    min_height_threshold: int = 20
) -> np.ndarray:
    """연결 컴포넌트 분석으로 가로선 제거
    
    각 연결된 영역의 가로세로 비율을 계산해서
    가로로 긴 영역들을 제거함.
    
    Args:
        mask: 입력 마스크
        horizontal_ratio_threshold: 가로선 판단 기준 (가로/세로 비율)
        min_height_threshold: 최소 높이 기준
        
    Returns:
        가로선이 필터링된 마스크
    """
    try:
        filtered_mask = mask.copy()
        
        # 연결된 컴포넌트 분석
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        for i in range(1, num_labels):  # 0은 배경이므로 제외
            x, y, w, h, area = stats[i]
            
            should_remove = False
            
            # 가로세로 비율이 임계값보다 큰 경우
            if h > 0 and (w / h) >= horizontal_ratio_threshold:
                should_remove = True
            
            # 높이가 너무 작은 경우 (얇은 가로선)
            elif h <= 5 and w > 50:
                should_remove = True  
            
            # 화면 하단의 가로로 긴 객체 (정지선 가능성)
            elif y > mask.shape[0] * 0.7 and h > 0 and (w / h) >= 2.0:
                should_remove = True
            
            if should_remove:
                filtered_mask[labels == i] = 0
        
        return filtered_mask
        
    except Exception as e:
        rospy.logwarn(f"가로선 필터링 중 오류: {e}")
        return mask


def remove_horizontal_lines_morphology(mask: np.ndarray) -> np.ndarray:
    """모폴로지 연산으로 가로선 제거
    
    구조 요소(커널)를 사용해서 가로선만 검출하고 제거함.
    세로선과 대각선은 최대한 보존함.
    
    Args:
        mask: 입력 마스크
        
    Returns:
        가로선이 제거된 마스크
    """
    try:
        # 가로 방향 커널로 가로선 검출
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
        
        # 세로 방향 커널로 세로선 보존
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 30))  
        vertical_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
        
        # 원본에서 가로선 제거
        filtered_mask = cv2.subtract(mask, horizontal_lines)
        
        # 세로선은 다시 추가
        result = cv2.bitwise_or(filtered_mask, vertical_lines)
        
        return result
        
    except Exception as e:
        rospy.logwarn(f"모폴로지 가로선 제거 중 오류: {e}")
        return mask


def thin_lanes(mask: np.ndarray) -> np.ndarray:
    """차선을 얇게 만드는 스켈레톤화 함수
    
    두꺼운 차선 마스크를 1픽셀 두께의 중심선으로 변환함.
    BEV 변환 후 정확한 피팅을 위해 사용함.
    
    Args:
        mask: 입력 마스크 (두꺼운 차선)
        
    Returns:
        얇게 만들어진 차선 마스크
    """
    try:
        # 작은 커널로 끊어진 선 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 스켈레톤화 수행
        skeleton = skeletonize_zhang_suen(closed)
        
        # 스켈레톤을 아주 조금만 두껍게 (완전히 얇으면 노이즈에 취약)
        thicken_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thickened = cv2.dilate(skeleton, thicken_kernel, iterations=1)
        
        return thickened
        
    except Exception as e:
        rospy.logwarn(f"차선 얇게 만들기 실패: {e}")
        return mask


def skeletonize_zhang_suen(img: np.ndarray) -> np.ndarray:
    """Zhang-Suen 스켈레톤화 알고리즘
    
    이진 이미지를 1픽셀 두께의 스켈레톤으로 변환하는 
    표준 알고리즘임. 형태는 유지하면서 두께만 줄임.
    
    Args:
        img: 이진 이미지
        
    Returns:
        스켈레톤화된 이미지
    """
    img = img.copy()
    img[img > 0] = 1  # 0과 1로 정규화
    
    skeleton = np.zeros(img.shape, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    # 반복적으로 침식과 팽창을 수행
    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        
        # 더 이상 변화가 없으면 종료
        if cv2.countNonZero(img) == 0:
            break
    
    skeleton[skeleton > 0] = 255
    return skeleton


def draw_fitted_lanes_on_bev(
    bev_image: np.ndarray, 
    x_pred: Optional[np.ndarray], 
    y_pred_l: Optional[np.ndarray], 
    y_pred_r: Optional[np.ndarray], 
    real_scale: float
) -> np.ndarray:
    """BEV 이미지에 피팅된 차선을 그리는 함수
    
    피팅 결과를 BEV 이미지 위에 시각적으로 표시해서
    디버깅이나 검증 목적으로 사용함.
    
    Args:
        bev_image: BEV 변환된 이미지
        x_pred, y_pred_l, y_pred_r: 피팅된 차선 좌표들
        real_scale: 미터/픽셀 변환 비율
        
    Returns:
        차선이 그려진 BEV 이미지
    """
    if x_pred is None or y_pred_l is None or y_pred_r is None:
        return bev_image
    
    # 그레이스케일이면 컬러로 변환
    if len(bev_image.shape) == 2:
        bev_with_lanes = cv2.cvtColor(bev_image.copy(), cv2.COLOR_GRAY2BGR)
    else:
        bev_with_lanes = bev_image.copy()
    
    bev_height, bev_width = bev_image.shape[:2]
    
    # 실제 좌표를 BEV 픽셀 좌표로 변환해서 그리기
    for i in range(len(x_pred)):
        # 좌측 차선 - 실제 좌표를 픽셀 좌표로 변환
        pixel_x_l = int(bev_width/2 - y_pred_l[i] / real_scale)
        pixel_y_l = int(bev_height - x_pred[i] / real_scale)
        
        # 우측 차선
        pixel_x_r = int(bev_width/2 - y_pred_r[i] / real_scale)
        pixel_y_r = int(bev_height - x_pred[i] / real_scale)
        
        # BEV 이미지 범위 내에서만 그리기
        if 0 <= pixel_x_l < bev_width and 0 <= pixel_y_l < bev_height:
            cv2.circle(bev_with_lanes, (pixel_x_l, pixel_y_l), 3, (0, 255, 0), -1)  # 좌측: 초록
        
        if 0 <= pixel_x_r < bev_width and 0 <= pixel_y_r < bev_height:
            cv2.circle(bev_with_lanes, (pixel_x_r, pixel_y_r), 3, (0, 0, 255), -1)  # 우측: 빨강
    
    return bev_with_lanes


def preprocess_lane_mask(
    mask: np.ndarray, 
    aggressive_removal: bool = True,
    apply_thinning: bool = True
) -> np.ndarray:
    """차선 마스크 전처리 파이프라인
    
    여러 단계의 전처리를 순서대로 적용해서
    깔끔한 차선 마스크를 만드는 함수임.
    
    Args:
        mask: 원본 차선 마스크
        aggressive_removal: 강력한 가로선 제거 적용 여부
        apply_thinning: 스켈레톤화 적용 여부
        
    Returns:
        전처리된 차선 마스크
    """
    processed = mask.copy()
    
    # 1단계: 강력한 가로선 제거
    if aggressive_removal:
        processed = aggressive_horizontal_removal(processed)
    
    # 2단계: 모폴로지 연산으로 추가 가로선 제거
    processed = remove_horizontal_lines_morphology(processed)
    
    # 3단계: 컴포넌트 분석으로 남은 가로선 제거  
    processed = filter_horizontal_lines(processed, horizontal_ratio_threshold=1.8)
    
    # 4단계: 차선을 얇게 만들기 (옵션)
    if apply_thinning:
        processed = thin_lanes(processed)
    
    return processed