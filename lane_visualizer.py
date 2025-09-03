#!/usr/bin/env python3

import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point


class LaneVisualizer:
    """차선을 실제 스케일로 MarkerArray로 시각화하는 클래스"""
    
    def __init__(self):
        self.marker_id_counter = 0
    
        
    def create_lane_markers(self, x_pred, y_pred_l, y_pred_r, timestamp, frame_id='ego_car'):
        """
        차선 피팅 결과를 MarkerArray로 변환
        
        Args:
            x_pred: 예측된 x 좌표 배열 (실제 미터 단위)
            y_pred_l: 예측된 좌측 차선 y 좌표 배열 (실제 미터 단위)
            y_pred_r: 예측된 우측 차선 y 좌표 배열 (실제 미터 단위)
            timestamp: ROS 타임스탬프
            frame_id: 좌표계 프레임 ID
            
        Returns:
            MarkerArray: 좌우 차선을 나타내는 마커 배열
        """
        if x_pred is None or y_pred_l is None or y_pred_r is None:
            return MarkerArray()
            
        marker_array = MarkerArray()
        
        # 좌측 차선 마커 생성 (초록색)
        left_marker = self._create_line_strip_marker(
            x_pred, y_pred_l, timestamp, frame_id,
            marker_id=0,
            color=ColorRGBA(0.0, 1.0, 0.0, 0.8),  # 초록색
            marker_name="left_lane"
        )
        marker_array.markers.append(left_marker)
        
        # 우측 차선 마커 생성 (빨간색)
        right_marker = self._create_line_strip_marker(
            x_pred, y_pred_r, timestamp, frame_id,
            marker_id=1,
            color=ColorRGBA(1.0, 0.0, 0.0, 0.8),  # 빨간색
            marker_name="right_lane"
        )
        marker_array.markers.append(right_marker)
        
        # 차선 중앙선 마커 생성
        y_center = (y_pred_l + y_pred_r) / 2.0
        center_marker = self._create_line_strip_marker(
            x_pred, y_center, timestamp, frame_id,
            marker_id=2,
            color=ColorRGBA(0.0, 0.0, 1.0, 0.6),  # 파란색
            marker_name="center_lane",
            line_width=0.05
        )
        marker_array.markers.append(center_marker)
        
        # 차선 점들을 개별 구체로도 표시
        left_points_marker = self._create_points_marker(
            x_pred, y_pred_l, timestamp, frame_id,
            marker_id=3,
            color=ColorRGBA(0.0, 0.8, 0.0, 1.0),  # 진한 초록색
            marker_name="left_lane_points"
        )
        marker_array.markers.append(left_points_marker)
        
        right_points_marker = self._create_points_marker(
            x_pred, y_pred_r, timestamp, frame_id,
            marker_id=4,
            color=ColorRGBA(0.8, 0.0, 0.0, 1.0),  # 진한 빨간색
            marker_name="right_lane_points"
        )
        marker_array.markers.append(right_points_marker)
        
        return marker_array
    
    def _create_line_strip_marker(self, x_coords, y_coords, timestamp, frame_id, 
                                  marker_id, color, marker_name, line_width=0.1):
        """LINE_STRIP 타입 마커 생성"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = timestamp
        marker.ns = marker_name
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # 마커 스케일 설정 (실제 미터 단위)
        marker.scale.x = line_width  # 선의 두께
        marker.scale.y = 0.0
        marker.scale.z = 0.0
        
        # 색상 설정
        marker.color = color
        
        # 점들 추가
        for i in range(len(x_coords)):
            point = Point()
            point.x = float(x_coords[i])  # 실제 미터 단위
            point.y = float(y_coords[i])  # 실제 미터 단위
            point.z = 0.0
            marker.points.append(point)
        
        # 마커 지속 시간 설정
        marker.lifetime = rospy.Duration(0.2)  # 200ms
        
        return marker
    
    def _create_points_marker(self, x_coords, y_coords, timestamp, frame_id, 
                              marker_id, color, marker_name, point_size=0.1):
        """POINTS 타입 마커 생성"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = timestamp
        marker.ns = marker_name
        marker.id = marker_id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # 마커 스케일 설정
        marker.scale.x = point_size  # 점의 크기
        marker.scale.y = point_size  # 점의 크기
        marker.scale.z = 0.0
        
        # 색상 설정
        marker.color = color
        
        # 점들 추가
        for i in range(len(x_coords)):
            point = Point()
            point.x = float(x_coords[i])  # 실제 미터 단위
            point.y = float(y_coords[i])  # 실제 미터 단위
            point.z = 0.0
            marker.points.append(point)
        
        # 마커 지속 시간 설정
        marker.lifetime = rospy.Duration(0.2)  # 200ms
        
        return marker
    
    def create_lane_boundary_markers(self, x_pred, y_pred_l, y_pred_r, timestamp, frame_id='ego_car'):
        """
        차선 경계를 더 명확하게 표시하는 추가 마커들
        
        Returns:
            MarkerArray: 차선 경계를 표시하는 추가 마커들
        """
        if x_pred is None or y_pred_l is None or y_pred_r is None:
            return MarkerArray()
            
        marker_array = MarkerArray()
        
        # 차선 폭을 나타내는 반투명 면 생성 (TRIANGLE_LIST)
        lane_area_marker = Marker()
        lane_area_marker.header.frame_id = frame_id
        lane_area_marker.header.stamp = timestamp
        lane_area_marker.ns = "lane_area"
        lane_area_marker.id = 10
        lane_area_marker.type = Marker.TRIANGLE_LIST
        lane_area_marker.action = Marker.ADD
        
        # 면의 색상 (반투명 회색)
        lane_area_marker.color = ColorRGBA(0.5, 0.5, 0.5, 0.3)
        lane_area_marker.scale.x = 1.0
        lane_area_marker.scale.y = 1.0
        lane_area_marker.scale.z = 1.0
        
        # 삼각형들로 차선 영역 생성
        for i in range(len(x_pred) - 1):
            # 첫 번째 삼각형 (좌하 -> 우하 -> 좌상)
            p1 = Point()
            p1.x = float(x_pred[i])
            p1.y = float(y_pred_l[i])
            p1.z = 0.05
            
            p2 = Point()
            p2.x = float(x_pred[i])
            p2.y = float(y_pred_r[i])
            p2.z = 0.05
            
            p3 = Point()
            p3.x = float(x_pred[i+1])
            p3.y = float(y_pred_l[i+1])
            p3.z = 0.05
            
            lane_area_marker.points.extend([p1, p2, p3])
            
            # 두 번째 삼각형 (우하 -> 우상 -> 좌상)
            p4 = Point()
            p4.x = float(x_pred[i])
            p4.y = float(y_pred_r[i])
            p4.z = 0.05
            
            p5 = Point()
            p5.x = float(x_pred[i+1])
            p5.y = float(y_pred_r[i+1])
            p5.z = 0.05
            
            p6 = Point()
            p6.x = float(x_pred[i+1])
            p6.y = float(y_pred_l[i+1])
            p6.z = 0.05
            
            lane_area_marker.points.extend([p4, p5, p6])
        
        lane_area_marker.lifetime = rospy.Duration(0.2)
        marker_array.markers.append(lane_area_marker)
        
        return marker_array
    
    def create_lane_info_text_marker(self, x_pred, y_pred_l, y_pred_r, timestamp, frame_id='ego_car'):
        """
        차선 정보를 텍스트로 표시하는 마커
        
        Returns:
            MarkerArray: 차선 정보 텍스트 마커
        """
        if x_pred is None or y_pred_l is None or y_pred_r is None:
            return MarkerArray()
            
        marker_array = MarkerArray()
        
        # 차선 폭 계산
        lane_widths = y_pred_l - y_pred_r
        avg_lane_width = np.mean(lane_widths)
        
        # 텍스트 마커 생성
        text_marker = Marker()
        text_marker.header.frame_id = frame_id
        text_marker.header.stamp = timestamp
        text_marker.ns = "lane_info"
        text_marker.id = 20
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        # 텍스트 위치 (첫 번째 점 위쪽)
        text_marker.pose.position.x = float(x_pred[0])
        text_marker.pose.position.y = float((y_pred_l[0] + y_pred_r[0]) / 2.0)
        text_marker.pose.position.z = 2.0  # 높이 2미터
        
        text_marker.pose.orientation.w = 1.0
        
        # 텍스트 내용
        text_marker.text = f"Lane Width: {avg_lane_width:.2f}m\nLength: {float(x_pred[-1]):.1f}m"
        
        # 텍스트 스타일
        text_marker.scale.z = 0.5  # 텍스트 크기
        text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)  # 흰색
        
        text_marker.lifetime = rospy.Duration(0.2)
        marker_array.markers.append(text_marker)
        
        return marker_array
    
    def clear_all_markers(self, timestamp, frame_id='ego_car'):
        """
        모든 마커를 지우는 MarkerArray 생성
        
        Returns:
            MarkerArray: 모든 마커를 DELETE하는 배열
        """
        marker_array = MarkerArray()
        
        # 기본 마커들 삭제
        for marker_id in range(5):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.id = marker_id
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        
        # 추가 마커들 삭제
        for marker_id in [10, 20]:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.id = marker_id
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        
        return marker_array