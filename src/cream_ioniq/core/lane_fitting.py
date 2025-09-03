#!/usr/bin/env python3
"""
차선 피팅 및 시간적 평활화 모듈

BEV 이미지에서 추출된 차선 점들을 곡선으로 피팅하고,
시간에 따른 변화를 부드럽게 만드는 모듈임.
RANSAC을 사용한 강건한 피팅과 이동평균 필터를 적용함.
"""

import numpy as np
import rospy
from sklearn import linear_model
import random
from typing import Optional, Tuple, List
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

# sklearn 경고 무시 (성능상 이유)
simplefilter("ignore", category=ConvergenceWarning)


class LaneFitter:
    """차선 곡선 피팅 및 시간적 평활화를 담당하는 클래스
    
    BEV 좌표계의 차선 점들을 받아서 좌우 차선을 구분하고,
    각각을 3차 곡선으로 피팅함. 시간적 평활화도 적용해서
    부드러운 차선 추적이 가능하도록 함.
    """
    
    def __init__(
        self,
        config: dict = None,
        order: int = 3,
        alpha: float = 10.0,
        lane_width: float = 3.7,
        y_margin: float = 0.8,
        x_range: float = 20.0,
        dx: float = 0.5,
        min_pts: int = 8,
        max_trials: int = 50,
        temporal_alpha: float = 0.7
    ):
        """차선 피터 초기화
        
        Args:
            config: 설정 딕셔너리 (제공되면 이 값을 우선 사용)
            order: 다항식 차수 (기본 3차)
            alpha: LASSO 정규화 강도
            lane_width: 기본 차선 폭 (미터)
            y_margin: 좌우 분류 마진 (미터)
            x_range: 예측 범위 (미터)
            dx: 예측 간격 (미터)
            min_pts: 피팅에 필요한 최소 점 수
            max_trials: RANSAC 최대 시도 횟수
            temporal_alpha: 시간적 평활화 계수 (0~1, 클수록 이전 값에 의존)
        """
        # config 딕셔너리가 있으면 거기서 값 가져옴
        if config is not None:
            lane_config = config.get('lane_fitting', {})
            temporal_config = config.get('temporal_smoothing', {})
            
            self.order = lane_config.get('polynomial_order', order)
            self.lane_width = lane_config.get('default_lane_width', lane_width)
            self.y_margin = lane_config.get('classification_margin', y_margin)
            self.x_range = lane_config.get('prediction_range', x_range)
            self.dx = lane_config.get('prediction_step', dx)
            self.min_pts = lane_config.get('min_points', min_pts)
            self.max_trials = lane_config.get('max_trials', max_trials)
            self.temporal_alpha = temporal_config.get('alpha', temporal_alpha)
            alpha = lane_config.get('lasso_alpha', alpha)
        else:
            # 기본값 사용
            self.order = order
            self.lane_width = lane_width
            self.y_margin = y_margin
            self.x_range = x_range
            self.dx = dx
            self.min_pts = min_pts
            self.max_trials = max_trials
            self.temporal_alpha = temporal_alpha
        
        # ROS Path 메시지용
        self.lane_path = Path()
        
        # 시간적 평활화를 위한 이전 프레임 데이터
        self.prev_y_pred_l: Optional[np.ndarray] = None
        self.prev_y_pred_r: Optional[np.ndarray] = None
        self.prev_x_pred: Optional[np.ndarray] = None
        self.detection_confidence = 0.0  # 검출 신뢰도 (0~1)
        self.no_detection_count = 0      # 연속 검출 실패 횟수
        
        # RANSAC 회귀기 초기화
        self._setup_ransac_regressors(alpha)
        self._initialize_models()
    
    def _setup_ransac_regressors(self, alpha: float) -> None:
        """RANSAC 회귀기들을 설정함
        
        sklearn 버전에 따라 파라미터 이름이 다를 수 있어서
        두 가지 방식으로 시도함.
        """
        try:
            # sklearn 1.2+ 버전용
            self.ransac_left = linear_model.RANSACRegressor(
                estimator=linear_model.Lasso(alpha=alpha),
                max_trials=self.max_trials,
                loss='absolute_error',
                min_samples=self.min_pts,
                residual_threshold=self.y_margin
            )
            
            self.ransac_right = linear_model.RANSACRegressor(
                estimator=linear_model.Lasso(alpha=alpha),
                max_trials=self.max_trials,
                loss='absolute_error',
                min_samples=self.min_pts,
                residual_threshold=self.y_margin
            )
        except TypeError:
            # sklearn 구버전용
            self.ransac_left = linear_model.RANSACRegressor(
                base_estimator=linear_model.Lasso(alpha=alpha),
                max_trials=self.max_trials,
                loss='absolute_loss',
                min_samples=self.min_pts,
                residual_threshold=self.y_margin
            )
            
            self.ransac_right = linear_model.RANSACRegressor(
                base_estimator=linear_model.Lasso(alpha=alpha),
                max_trials=self.max_trials,
                loss='absolute_loss',
                min_samples=self.min_pts,
                residual_threshold=self.y_margin
            )
    
    def _initialize_models(self) -> None:
        """RANSAC 모델들을 기본값으로 초기화함
        
        처음 실행할 때나 재초기화가 필요할 때 사용함.
        한국 도로 기준 차선 폭인 3.7m를 기본으로 함.
        """
        # 더미 데이터로 초기 모델 생성
        X = np.stack([np.arange(0, 2, 0.02)**i for i in reversed(range(1, self.order+1))]).T
        y_l = (self.lane_width / 2) * np.ones_like(np.arange(0, 2, 0.02))   # 좌측: +1.85m
        y_r = -(self.lane_width / 2) * np.ones_like(np.arange(0, 2, 0.02))  # 우측: -1.85m
        
        self.ransac_left.fit(X, y_l)
        self.ransac_right.fit(X, y_r)
    
    def _preprocess_points(self, lane_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """차선 점들을 전처리하고 좌우로 분류함
        
        거리별로 샘플링하고, 중심선 기준으로 좌우를 나누는 작업을 함.
        복잡한 RANSAC 예측 대신 단순한 Y좌표 기준 분류를 사용함.
        
        Args:
            lane_pts: 4xN 형태의 차선 점들 [x, y, z, w]
            
        Returns:
            (x_left, y_left, x_right, y_right): 좌우로 분류된 점들
        """
        # 거리별 샘플링으로 균등한 분포 만들기
        idx_list = []
        for d in np.arange(0, self.x_range, self.dx):
            # 해당 거리 범위의 점들 찾기
            distance_mask = np.logical_and(lane_pts[0, :] >= d, lane_pts[0, :] < d + self.dx)
            idx_full_list = np.where(distance_mask)[0].tolist()
            
            # 해당 구간에서 랜덤 샘플링
            if len(idx_full_list) > 0:
                sample_count = max(1, min(self.min_pts, len(idx_full_list)))
                idx_list += random.sample(idx_full_list, sample_count)
        
        if len(idx_list) == 0:
            rospy.logwarn("차선 점 샘플링 실패 - 빈 결과 반환")
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # 샘플링된 점들 추출
        lane_pts = lane_pts[:, idx_list]
        x_g = lane_pts[0, :].copy()
        y_g = lane_pts[1, :].copy()
        
        if len(x_g) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # 좌표 분포 로깅 (디버깅용)
        rospy.loginfo_throttle(3.0, 
            f"Y 좌표 분포: {np.min(y_g):.2f} ~ {np.max(y_g):.2f} (평균: {np.mean(y_g):.2f})")
        
        # 중심선(Y=0) 기준으로 좌우 분류
        # 좌측: Y > 0 (양수), 우측: Y < 0 (음수)
        center_y = 0.0
        left_mask = y_g > center_y
        right_mask = y_g < center_y
        
        x_left = x_g[left_mask]
        y_left = y_g[left_mask]
        x_right = x_g[right_mask]
        y_right = y_g[right_mask]
        
        # 분류 결과 로깅
        rospy.loginfo_throttle(2.0, f"중심 기준 분류 - 좌측: {len(x_left)}개, 우측: {len(x_right)}개")
        
        return x_left, y_left, x_right, y_right
    
    def _fit_current_frame(self, x_left: np.ndarray, y_left: np.ndarray, 
                          x_right: np.ndarray, y_right: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """현재 프레임 데이터만으로 차선 피팅을 수행함
        
        Args:
            x_left, y_left: 좌측 차선 점들
            x_right, y_right: 우측 차선 점들
            
        Returns:
            (x_pred, y_pred_l, y_pred_r): 피팅된 차선 곡선들 (실패시 None)
        """
        try:
            # 충분한 점이 있는 경우에만 피팅 수행
            if len(y_left) >= self.min_pts:
                X_left = np.stack([x_left**i for i in reversed(range(1, self.order+1))]).T
                self.ransac_left.fit(X_left, y_left)
            
            if len(y_right) >= self.min_pts:
                X_right = np.stack([x_right**i for i in reversed(range(1, self.order+1))]).T
                self.ransac_right.fit(X_right, y_right)
                
        except Exception as e:
            rospy.logwarn(f"RANSAC 피팅 실패: {e}")
            return None, None, None
        
        # 예측할 X 좌표 범위 생성 (차량 바로 앞부터)
        x_pred = np.arange(0.5, self.x_range, self.dx).astype(np.float32)
        X_pred = np.stack([x_pred**i for i in reversed(range(1, self.order+1))]).T
        
        try:
            # 곡선 예측
            y_pred_l = self.ransac_left.predict(X_pred)
            y_pred_r = self.ransac_right.predict(X_pred)
        except Exception as e:
            rospy.logwarn(f"곡선 예측 실패: {e}")
            return None, None, None
        
        # 양쪽 차선이 모두 검출된 경우 차선 폭 업데이트
        if len(y_left) >= self.min_pts and len(y_right) >= self.min_pts:
            self._update_lane_width(y_pred_l, y_pred_r)
        
        # 한쪽 차선이 없는 경우 다른 쪽 기준으로 보완
        if len(y_left) < self.min_pts:
            y_pred_l = y_pred_r + self.lane_width
        
        if len(y_right) < self.min_pts:
            y_pred_r = y_pred_l - self.lane_width
        
        return x_pred, y_pred_l, y_pred_r
    
    def _update_lane_width(self, y_pred_l: np.ndarray, y_pred_r: np.ndarray) -> None:
        """실제 측정된 차선 폭으로 기본값을 업데이트함
        
        Args:
            y_pred_l: 좌측 차선 예측값
            y_pred_r: 우측 차선 예측값
        """
        measured_width = np.mean(y_pred_l - y_pred_r)
        
        # 합리적인 범위에서만 업데이트 (3.0m ~ 4.5m)
        if 3.0 <= measured_width <= 4.5:
            # 급격한 변화 방지를 위한 가중 평균
            self.lane_width = 0.9 * self.lane_width + 0.1 * measured_width
            rospy.loginfo_throttle(5.0, 
                f"차선 폭 업데이트: {self.lane_width:.2f}m (측정값: {measured_width:.2f}m)")
        else:
            rospy.logwarn_throttle(5.0, 
                f"비현실적인 차선 폭 감지: {measured_width:.2f}m, 기존값 유지: {self.lane_width:.2f}m")
    
    def _apply_temporal_smoothing(self, current_x_pred: np.ndarray, 
                                 current_y_pred_l: np.ndarray, 
                                 current_y_pred_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """시간적 평활화를 적용해서 부드러운 차선 변화를 만듦
        
        Args:
            current_x_pred, current_y_pred_l, current_y_pred_r: 현재 프레임 예측값들
            
        Returns:
            평활화된 예측값들
        """
        # 첫 번째 프레임이거나 이전 데이터가 없는 경우
        if self.prev_y_pred_l is None or self.prev_y_pred_r is None:
            self.prev_x_pred = current_x_pred.copy()
            self.prev_y_pred_l = current_y_pred_l.copy()
            self.prev_y_pred_r = current_y_pred_r.copy()
            return current_x_pred, current_y_pred_l, current_y_pred_r
        
        # 이동평균 필터 적용: 이전값 * alpha + 현재값 * (1 - alpha)
        smoothed_y_pred_l = (self.temporal_alpha * self.prev_y_pred_l + 
                            (1 - self.temporal_alpha) * current_y_pred_l)
        smoothed_y_pred_r = (self.temporal_alpha * self.prev_y_pred_r + 
                            (1 - self.temporal_alpha) * current_y_pred_r)
        
        # 이전 프레임 데이터 업데이트
        self.prev_x_pred = current_x_pred.copy()
        self.prev_y_pred_l = smoothed_y_pred_l.copy()
        self.prev_y_pred_r = smoothed_y_pred_r.copy()
        
        rospy.loginfo_throttle(3.0, f"시간적 평활화 적용 (alpha={self.temporal_alpha})")
        
        return current_x_pred, smoothed_y_pred_l, smoothed_y_pred_r
    
    def _handle_no_detection(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """차선 검출 실패 시 이전 프레임 데이터를 활용함
        
        Returns:
            이전 프레임 데이터 또는 None
        """
        self.no_detection_count += 1
        self.detection_confidence = max(0.0, self.detection_confidence - 0.3)
        
        # 연속 실패가 너무 많으면 데이터 리셋
        if self.no_detection_count > 10:
            rospy.logwarn("연속 검출 실패 횟수 초과 - 시간적 데이터 리셋")
            self.prev_y_pred_l = None
            self.prev_y_pred_r = None
            self.prev_x_pred = None
            self.detection_confidence = 0.0
            return None, None, None
        
        # 이전 프레임 데이터가 있으면 재사용
        if self.prev_y_pred_l is not None and self.prev_y_pred_r is not None:
            rospy.logwarn_throttle(1.0, 
                f"차선 미검출 - 이전 프레임 사용 (신뢰도: {self.detection_confidence:.2f})")
            return (self.prev_x_pred.copy(), 
                   self.prev_y_pred_l.copy(), 
                   self.prev_y_pred_r.copy())
        
        return None, None, None
    
    def fit_lanes(self, lane_pts: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """메인 차선 피팅 함수
        
        BEV 좌표계의 차선 점들을 받아서 좌우 차선을 피팅하고
        시간적 평활화까지 적용한 최종 결과를 반환함.
        
        Args:
            lane_pts: 4xN 형태의 차선 점들 [x, y, z, w]
            
        Returns:
            (x_pred, y_pred_l, y_pred_r): 피팅된 차선들 (실패시 None)
        """
        # 입력 데이터 검증
        if lane_pts.size == 0:
            return self._handle_no_detection()
        
        # 점들을 전처리하고 좌우 분류
        x_left, y_left, x_right, y_right = self._preprocess_points(lane_pts)
        
        rospy.loginfo_throttle(1.0, f"전처리 결과 - 좌측: {len(x_left)}개, 우측: {len(x_right)}개")
        
        # 양쪽 모두 점이 없으면 이전 데이터 활용
        if len(x_left) == 0 and len(x_right) == 0:
            rospy.logwarn("양쪽 차선 모두 점 없음")
            return self._handle_no_detection()
        
        # 현재 프레임에서 차선 피팅 시도
        current_result = self._fit_current_frame(x_left, y_left, x_right, y_right)
        
        if current_result[0] is None:
            return self._handle_no_detection()
        
        current_x_pred, current_y_pred_l, current_y_pred_r = current_result
        
        # 시간적 평활화 적용
        smoothed_result = self._apply_temporal_smoothing(
            current_x_pred, current_y_pred_l, current_y_pred_r
        )
        
        # 성공적인 검출 후 상태 업데이트
        self.detection_confidence = min(1.0, self.detection_confidence + 0.2)
        self.no_detection_count = 0
        
        return smoothed_result
    
    def create_path_message(self, x_pred: Optional[np.ndarray], 
                           y_pred_l: Optional[np.ndarray], 
                           y_pred_r: Optional[np.ndarray], 
                           frame_id: str = 'ego_car') -> Path:
        """ROS Path 메시지를 생성함
        
        Args:
            x_pred, y_pred_l, y_pred_r: 피팅된 차선 데이터
            frame_id: ROS 프레임 ID
            
        Returns:
            차선 중앙을 따라가는 Path 메시지
        """
        self.lane_path = Path()
        self.lane_path.header.frame_id = frame_id
        
        if x_pred is None or y_pred_l is None or y_pred_r is None:
            return self.lane_path
        
        # 차선 중앙선을 따라가는 경로 생성
        for i in range(len(x_pred)):
            pose = PoseStamped()
            pose.pose.position.x = float(x_pred[i])
            pose.pose.position.y = float((y_pred_l[i] + y_pred_r[i]) / 2.0)
            pose.pose.position.z = 0.0
            
            # 기본 orientation (나중에 차선 각도 계산 추가 가능)
            pose.pose.orientation.w = 1.0
            
            self.lane_path.poses.append(pose)
        
        return self.lane_path
    
    def get_lane_info(self) -> dict:
        """현재 차선 상태 정보를 반환함
        
        Returns:
            차선 상태 정보 딕셔너리
        """
        return {
            'lane_width': self.lane_width,
            'detection_confidence': self.detection_confidence,
            'no_detection_count': self.no_detection_count,
            'has_previous_data': self.prev_y_pred_l is not None
        }