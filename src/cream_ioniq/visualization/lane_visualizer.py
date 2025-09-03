#!/usr/bin/env python3
"""
차선 시각화 모듈

차선 피팅 결과를 RViz에서 볼 수 있도록 MarkerArray로 변환하는 모듈임.
실제 미터 단위의 차선 데이터를 받아서 3D 마커로 만들어줌.
차선, 중앙선, 경계면, 정보 텍스트 등 다양한 시각화를 제공함.
"""

import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from typing import Optional


class LaneVisualizer:
    """차선을 RViz 마커로 시각화하는 클래스
    
    차선 피팅 결과를 받아서 3D 공간에 표시할 수 있는
    여러 종류의 마커들을 생성함. 실제 미터 단위를 사용해서
    정확한 크기로 표시됨.
    """
    
    def __init__(self, config: dict = None):
        """시각화기 초기화
        
        Args:
            config: 설정 딕셔너리 (시각화 관련 파라미터)
        """
        self.marker_id_counter = 0
        
        # 설정값 로드
        if config is not None and 'visualization' in config:
            vis_config = config['visualization']
            self.marker_lifetime = vis_config.get('marker_lifetime', 0.2)
            self.line_width = vis_config.get('line_width', 0.1)
            self.point_size = vis_config.get('point_size', 0.1)
            self.left_lane_color = vis_config.get('left_lane_color', [0.0, 1.0, 0.0, 0.8])
            self.right_lane_color = vis_config.get('right_lane_color', [1.0, 0.0, 0.0, 0.8])
            self.center_line_color = vis_config.get('center_line_color', [0.0, 0.0, 1.0, 0.6])
            self.lane_area_color = vis_config.get('lane_area_color', [0.5, 0.5, 0.5, 0.3])
        else:
            # 기본값 사용
            self.marker_lifetime = 0.2
            self.line_width = 0.1
            self.point_size = 0.1
            self.left_lane_color = [0.0, 1.0, 0.0, 0.8]
            self.right_lane_color = [1.0, 0.0, 0.0, 0.8]
            self.center_line_color = [0.0, 0.0, 1.0, 0.6]
            self.lane_area_color = [0.5, 0.5, 0.5, 0.3]
    
    def create_lane_markers(
        self, 
        x_pred: np.ndarray, 
        y_pred_l: np.ndarray, 
        y_pred_r: np.ndarray, 
        timestamp, 
        frame_id: str = 'ego_car'
    ) -> MarkerArray:
        """차선 피팅 결과를 MarkerArray로 변환함
        
        Args:
            x_pred: 예측된 x 좌표 배열 (실제 미터 단위)
            y_pred_l: 좌측 차선 y 좌표 배열 (실제 미터 단위)
            y_pred_r: 우측 차선 y 좌표 배열 (실제 미터 단위)
            timestamp: ROS 타임스탬프
            frame_id: 좌표계 프레임 ID
            
        Returns:
            차선들을 나타내는 마커 배열
        """
        if x_pred is None or y_pred_l is None or y_pred_r is None:
            return MarkerArray()
        
        marker_array = MarkerArray()
        
        # 좌측 차선 마커
        left_color = ColorRGBA(*self.left_lane_color)
        left_marker = self._create_line_strip_marker(
            x_pred, y_pred_l, timestamp, frame_id,
            marker_id=0,
            color=left_color,
            marker_name="left_lane"
        )
        marker_array.markers.append(left_marker)
        
        # 우측 차선 마커
        right_color = ColorRGBA(*self.right_lane_color)
        right_marker = self._create_line_strip_marker(
            x_pred, y_pred_r, timestamp, frame_id,
            marker_id=1,
            color=right_color,
            marker_name="right_lane"
        )
        marker_array.markers.append(right_marker)
        
        # 차선 중앙선 마커
        y_center = (y_pred_l + y_pred_r) / 2.0
        center_color = ColorRGBA(*self.center_line_color)
        center_marker = self._create_line_strip_marker(
            x_pred, y_center, timestamp, frame_id,
            marker_id=2,
            color=center_color,
            marker_name="center_lane",
            line_width=self.line_width / 2.0  # 중앙선은 절반 두께
        )
        marker_array.markers.append(center_marker)
        
        # 차선 점들을 개별 구체로도 표시
        left_points_color = ColorRGBA(self.left_lane_color[0], self.left_lane_color[1], self.left_lane_color[2], 1.0)
        left_points_marker = self._create_points_marker(
            x_pred, y_pred_l, timestamp, frame_id,
            marker_id=3,
            color=left_points_color,
            marker_name="left_lane_points"
        )
        marker_array.markers.append(left_points_marker)
        
        right_points_color = ColorRGBA(self.right_lane_color[0], self.right_lane_color[1], self.right_lane_color[2], 1.0)
        right_points_marker = self._create_points_marker(
            x_pred, y_pred_r, timestamp, frame_id,
            marker_id=4,
            color=right_points_color,
            marker_name="right_lane_points"
        )
        marker_array.markers.append(right_points_marker)
        
        return marker_array
    
    def _create_line_strip_marker(
        self, 
        x_coords: np.ndarray, 
        y_coords: np.ndarray, 
        timestamp, 
        frame_id: str,
        marker_id: int, 
        color: ColorRGBA, 
        marker_name: str, 
        line_width: float = None
    ) -> Marker:
        """선 형태의 마커를 생성함
        
        Args:
            x_coords, y_coords: 선의 좌표들
            timestamp: 타임스탬프
            frame_id: 프레임 ID
            marker_id: 마커 고유 ID
            color: 선 색상
            marker_name: 마커 네임스페이스
            line_width: 선 두께 (미터)
            
        Returns:
            LINE_STRIP 타입 마커
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = timestamp
        marker.ns = marker_name
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # 마커 크기 설정 (실제 미터 단위)
        if line_width is None:
            line_width = self.line_width
        marker.scale.x = line_width  # 선의 두께
        marker.scale.y = 0.0
        marker.scale.z = 0.0
        
        # 색상 설정
        marker.color = color
        
        # 점들을 마커에 추가
        for i in range(len(x_coords)):
            point = Point()
            point.x = float(x_coords[i])  # 실제 미터 단위
            point.y = float(y_coords[i])  # 실제 미터 단위
            point.z = 0.0
            marker.points.append(point)
        
        # 마커 지속 시간
        marker.lifetime = rospy.Duration(self.marker_lifetime)
        
        return marker
    
    def _create_points_marker(
        self, 
        x_coords: np.ndarray, 
        y_coords: np.ndarray, 
        timestamp, 
        frame_id: str,
        marker_id: int, 
        color: ColorRGBA, 
        marker_name: str, 
        point_size: float = None
    ) -> Marker:
        """점들로 이루어진 마커를 생성함
        
        Args:
            x_coords, y_coords: 점들의 좌표
            timestamp: 타임스탬프
            frame_id: 프레임 ID
            marker_id: 마커 고유 ID
            color: 점 색상
            marker_name: 마커 네임스페이스
            point_size: 점 크기 (미터)
            
        Returns:
            POINTS 타입 마커
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = timestamp
        marker.ns = marker_name
        marker.id = marker_id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # 마커 크기 설정
        if point_size is None:
            point_size = self.point_size
        marker.scale.x = point_size  # 점의 크기
        marker.scale.y = point_size  # 점의 크기
        marker.scale.z = 0.0
        
        # 색상 설정
        marker.color = color
        
        # 점들을 마커에 추가
        for i in range(len(x_coords)):
            point = Point()
            point.x = float(x_coords[i])  # 실제 미터 단위
            point.y = float(y_coords[i])  # 실제 미터 단위
            point.z = 0.0
            marker.points.append(point)
        
        # 마커 지속 시간
        marker.lifetime = rospy.Duration(self.marker_lifetime)
        
        return marker
    
    def create_lane_boundary_markers(
        self, 
        x_pred: np.ndarray, 
        y_pred_l: np.ndarray, 
        y_pred_r: np.ndarray, 
        timestamp, 
        frame_id: str = 'ego_car'
    ) -> MarkerArray:
        """차선 경계를 면으로 표시하는 마커들을 생성함
        
        차선 사이의 공간을 반투명한 면으로 채워서
        주행 가능 영역을 명확하게 보여줌.
        
        Args:
            x_pred, y_pred_l, y_pred_r: 차선 좌표들
            timestamp: 타임스탬프
            frame_id: 프레임 ID
            
        Returns:
            차선 경계를 표시하는 마커 배열
        """
        if x_pred is None or y_pred_l is None or y_pred_r is None:
            return MarkerArray()
        
        marker_array = MarkerArray()
        
        # 차선 사이 공간을 채우는 삼각형 면들
        lane_area_marker = Marker()
        lane_area_marker.header.frame_id = frame_id
        lane_area_marker.header.stamp = timestamp
        lane_area_marker.ns = "lane_area"
        lane_area_marker.id = 10
        lane_area_marker.type = Marker.TRIANGLE_LIST
        lane_area_marker.action = Marker.ADD
        
        # 반투명 회색 면
        lane_area_marker.color = ColorRGBA(0.5, 0.5, 0.5, 0.3)
        lane_area_marker.scale.x = 1.0
        lane_area_marker.scale.y = 1.0
        lane_area_marker.scale.z = 1.0
        
        # 인접한 점들 사이를 삼각형으로 연결
        for i in range(len(x_pred) - 1):
            # 첫 번째 삼각형 (좌하 -> 우하 -> 좌상)
            p1 = Point(x=float(x_pred[i]), y=float(y_pred_l[i]), z=0.05)
            p2 = Point(x=float(x_pred[i]), y=float(y_pred_r[i]), z=0.05)
            p3 = Point(x=float(x_pred[i+1]), y=float(y_pred_l[i+1]), z=0.05)
            lane_area_marker.points.extend([p1, p2, p3])
            
            # 두 번째 삼각형 (우하 -> 우상 -> 좌상)
            p4 = Point(x=float(x_pred[i]), y=float(y_pred_r[i]), z=0.05)
            p5 = Point(x=float(x_pred[i+1]), y=float(y_pred_r[i+1]), z=0.05)
            p6 = Point(x=float(x_pred[i+1]), y=float(y_pred_l[i+1]), z=0.05)
            lane_area_marker.points.extend([p4, p5, p6])
        
        lane_area_marker.lifetime = rospy.Duration(0.2)
        marker_array.markers.append(lane_area_marker)
        
        return marker_array
    
    def create_lane_info_text_marker(
        self, 
        x_pred: np.ndarray, 
        y_pred_l: np.ndarray, 
        y_pred_r: np.ndarray, 
        timestamp, 
        frame_id: str = 'ego_car'
    ) -> MarkerArray:
        """차선 정보를 텍스트로 표시하는 마커를 생성함
        
        차선 폭, 예측 거리 등의 정보를 3D 공간에 텍스트로 표시함.
        
        Args:
            x_pred, y_pred_l, y_pred_r: 차선 좌표들
            timestamp: 타임스탬프
            frame_id: 프레임 ID
            
        Returns:
            차선 정보 텍스트 마커
        """
        if x_pred is None or y_pred_l is None or y_pred_r is None:
            return MarkerArray()
        
        marker_array = MarkerArray()
        
        # 차선 폭 계산
        lane_widths = y_pred_l - y_pred_r
        avg_lane_width = np.mean(lane_widths)
        max_distance = float(x_pred[-1])
        
        # 텍스트 마커 생성
        text_marker = Marker()
        text_marker.header.frame_id = frame_id
        text_marker.header.stamp = timestamp
        text_marker.ns = "lane_info"
        text_marker.id = 20
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        # 텍스트 위치 (첫 번째 점 위쪽 2미터)
        text_marker.pose.position.x = float(x_pred[0])
        text_marker.pose.position.y = float((y_pred_l[0] + y_pred_r[0]) / 2.0)
        text_marker.pose.position.z = 2.0
        text_marker.pose.orientation.w = 1.0
        
        # 텍스트 내용
        text_marker.text = (f"차선폭: {avg_lane_width:.2f}m\\n"
                           f"예측거리: {max_distance:.1f}m\\n"
                           f"점개수: {len(x_pred)}개")
        
        # 텍스트 스타일
        text_marker.scale.z = 0.5  # 텍스트 크기
        text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)  # 흰색
        text_marker.lifetime = rospy.Duration(0.2)
        
        marker_array.markers.append(text_marker)
        
        return marker_array
    
    def clear_all_markers(self, timestamp, frame_id: str = 'ego_car') -> MarkerArray:
        """모든 마커를 지우는 MarkerArray 생성
        
        차선 검출이 실패했을 때나 초기화할 때 사용함.
        
        Args:
            timestamp: 타임스탬프
            frame_id: 프레임 ID
            
        Returns:
            모든 마커를 DELETE하는 배열
        """
        marker_array = MarkerArray()
        
        # 기본 마커들 삭제 (ID 0~4)
        for marker_id in range(5):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.id = marker_id
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        
        # 추가 마커들 삭제 (면, 텍스트)
        for marker_id in [10, 20]:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.id = marker_id
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        
        return marker_array
    
    def create_detection_status_marker(
        self, 
        confidence: float, 
        timestamp, 
        frame_id: str = 'ego_car'
    ) -> MarkerArray:
        """차선 검출 상태를 표시하는 마커를 생성함
        
        검출 신뢰도에 따라 색상이 바뀌는 상태 표시기를 만들어줌.
        
        Args:
            confidence: 검출 신뢰도 (0~1)
            timestamp: 타임스탬프
            frame_id: 프레임 ID
            
        Returns:
            상태 표시 마커
        """
        marker_array = MarkerArray()
        
        # 상태 표시 구체
        status_marker = Marker()
        status_marker.header.frame_id = frame_id
        status_marker.header.stamp = timestamp
        status_marker.ns = "detection_status"
        status_marker.id = 30
        status_marker.type = Marker.SPHERE
        status_marker.action = Marker.ADD
        
        # 차량 앞 오른쪽에 위치
        status_marker.pose.position.x = 1.0
        status_marker.pose.position.y = -2.0
        status_marker.pose.position.z = 1.5
        status_marker.pose.orientation.w = 1.0
        
        # 크기 설정
        status_marker.scale.x = 0.3
        status_marker.scale.y = 0.3
        status_marker.scale.z = 0.3
        
        # 신뢰도에 따른 색상 (빨강 -> 노랑 -> 초록)
        if confidence > 0.7:
            status_marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)  # 초록 (좋음)
        elif confidence > 0.3:
            status_marker.color = ColorRGBA(1.0, 1.0, 0.0, 0.8)  # 노랑 (보통)
        else:
            status_marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)  # 빨강 (나쁨)
        
        status_marker.lifetime = rospy.Duration(0.2)
        marker_array.markers.append(status_marker)
        
        return marker_array