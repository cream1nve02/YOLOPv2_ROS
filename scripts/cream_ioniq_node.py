#!/usr/bin/env python3
import rospy
import torch
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import os
import sys
import json
import math
from sklearn import linear_model
import random
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray

# ignore sklearn warnings
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# Add the YOLOPv2 utils directory to the Python path
yolopv2_ros_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_dir = os.path.join(yolopv2_ros_dir, 'utils')
sys.path.append(utils_dir)

from utils import (
    time_synchronized, select_device, 
    lane_line_mask, driving_area_mask, letterbox
)

# BEV Config Transform import
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bev_transform import BEVTransform as BEVConfigTransform
    from lane_visualizer import LaneVisualizer
    BEV_CONFIG_AVAILABLE = True
except ImportError as e:
    BEV_CONFIG_AVAILABLE = False
    print(f"BEV Config Transform not available: {e}")


def rotationMtx(yaw, pitch, roll):
    R_x = np.array([[1,         0,              0,                0],
                    [0,         math.cos(roll), -math.sin(roll) , 0],
                    [0,         math.sin(roll), math.cos(roll)  , 0],
                    [0,         0,              0,               1]])
                     
    R_y = np.array([[math.cos(pitch),    0,      math.sin(pitch) , 0],
                    [0,                  1,      0               , 0],
                    [-math.sin(pitch),   0,      math.cos(pitch) , 0],
                    [0,         0,              0,               1]])
    
    R_z = np.array([[math.cos(yaw),    -math.sin(yaw),    0,    0],
                    [math.sin(yaw),    math.cos(yaw),     0,    0],
                    [0,                0,                 1,    0],
                    [0,         0,              0,               1]])
                     
    R = np.matmul(R_x, np.matmul(R_y, R_z))
    return R

def traslationMtx(x, y, z):
    M = np.array([[1,         0,              0,               x],
                  [0,         1,              0,               y],
                  [0,         0,              1,               z],
                  [0,         0,              0,               1]])
    return M

def project2img_mtx(params_cam):
    if params_cam["ENGINE"]=='UNITY':
        fc_x = params_cam["HEIGHT"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))
        fc_y = params_cam["HEIGHT"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))
    else:
        fc_x = params_cam["WIDTH"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))
        fc_y = params_cam["WIDTH"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))

    cx = params_cam["WIDTH"]/2
    cy = params_cam["HEIGHT"]/2
    
    R_f = np.array([[fc_x,  0,      cx],
                    [0,     fc_y,   cy]])
    return R_f

class BEVTransform:
    def __init__(self, params_cam, xb=10.0, zb=10.0):
        self.xb = xb
        self.zb = zb
        self.theta = np.deg2rad(params_cam["PITCH"])
        self.width = params_cam["WIDTH"]
        self.height = params_cam["HEIGHT"]
        self.x = params_cam["X"]
        
        if params_cam["ENGINE"]=="UNITY":
            self.alpha_r = np.deg2rad(params_cam["FOV"]/2)
            self.fc_y = params_cam["HEIGHT"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))
            self.alpha_c = np.arctan2(params_cam["WIDTH"]/2, self.fc_y)
            self.fc_x = self.fc_y
        else:
            self.alpha_c = np.deg2rad(params_cam["FOV"]/2)
            self.fc_x = params_cam["WIDTH"]/(2*np.tan(np.deg2rad(params_cam["FOV"]/2)))
            self.alpha_r = np.arctan2(params_cam["HEIGHT"]/2, self.fc_x)
            self.fc_y = self.fc_x
            
        self.h = params_cam["Z"] + 0.34
        self.n = float(params_cam["WIDTH"])
        self.m = float(params_cam["HEIGHT"])
        
        self.RT_b2g = np.matmul(np.matmul(traslationMtx(xb, 0, zb), rotationMtx(np.deg2rad(-90), 0, 0)),
                                rotationMtx(0, 0, np.deg2rad(180)))
        
        self.proj_mtx = project2img_mtx(params_cam)
        self._build_tf(params_cam)

    def calc_Xv_Yu(self, U, V):
        Xv = self.h*(np.tan(self.theta)*(1-2*(V-1)/(self.m-1))*np.tan(self.alpha_r)-1)/\
            (-np.tan(self.theta)+(1-2*(V-1)/(self.m-1))*np.tan(self.alpha_r))
        Yu = (1-2*(U-1)/(self.n-1))*Xv*np.tan(self.alpha_c)
        return Xv, Yu

    def _build_tf(self, params_cam):
        v = np.array([params_cam["HEIGHT"]*0.5, params_cam["HEIGHT"]]).astype(np.float32)
        u = np.array([0, params_cam["WIDTH"]]).astype(np.float32)
        U, V = np.meshgrid(u, v)
        Xv, Yu = self.calc_Xv_Yu(U, V)
        
        xyz_g = np.concatenate([Xv.reshape([1,-1]) + params_cam["X"],
                                Yu.reshape([1,-1]),
                                np.zeros_like(Yu.reshape([1,-1])),
                                np.ones_like(Yu.reshape([1,-1]))], axis=0)
        
        xyz_bird = np.matmul(np.linalg.inv(self.RT_b2g), xyz_g)
        xyi = self.project_pts2img(xyz_bird)
        
        src_pts = np.concatenate([U.reshape([-1, 1]), V.reshape([-1, 1])], axis=1).astype(np.float32)
        dst_pts = xyi.astype(np.float32)
        
        self.perspective_tf = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.perspective_inv_tf = cv2.getPerspectiveTransform(dst_pts, src_pts)

    def warp_bev_img(self, img):
        img_warp = cv2.warpPerspective(img, self.perspective_tf, (self.width, self.height), flags=cv2.INTER_LINEAR)
        return img_warp
        
    def project_pts2img(self, xyz_bird):
        xc, yc, zc = xyz_bird[0,:].reshape([1,-1]), xyz_bird[1,:].reshape([1,-1]), xyz_bird[2,:].reshape([1,-1])
        xn, yn = xc/(zc+0.0001), yc/(zc+0.0001)
        xyi = np.matmul(self.proj_mtx, np.concatenate([xn, yn, np.ones_like(xn)], axis=0))
        xyi = xyi[0:2,:].T
        return xyi


class CURVEFit:
    def __init__(
        self,
        order=3,
        alpha=10,
        lane_width=4.0,  # BEV config의 실제 차로 폭에 맞춤 (4미터)
        y_margin=0.5,
        x_range=20,  # BEV config의 실제 길이에 맞춤 (20미터)
        dx=0.5,
        min_pts=50,
        max_tri=5,
        temporal_alpha=0.7  # 시간적 평활화 계수 (0.7 = 이전 프레임 70% + 현재 프레임 30%)
    ):
        self.order = order
        self.lane_width = lane_width
        self.y_margin = y_margin
        self.x_range = x_range
        self.dx = dx
        self.min_pts = min_pts
        self.max_trials = max_tri
        self.temporal_alpha = temporal_alpha

        self.lane_path = Path()
        
        # 시간적 평활화를 위한 이전 프레임 데이터 저장
        self.prev_y_pred_l = None
        self.prev_y_pred_r = None
        self.prev_x_pred = None
        self.detection_confidence = 0.0  # 검출 신뢰도 (0~1)
        self.no_detection_count = 0      # 연속 검출 실패 카운트

        try:
            # sklearn 1.2+ 버전
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
            # sklearn 구버전 호환
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
        
        self._init_model()

    def _init_model(self):
        X = np.stack([np.arange(0, 2, 0.02)**i for i in reversed(range(1, self.order+1))]).T
        # 3.7m 차선 폭에 맞춰 초기화
        y_l = 1.85*np.ones_like(np.arange(0, 2, 0.02))   # 좌측: +1.85m (3.7/2)
        y_r = -1.85*np.ones_like(np.arange(0, 2, 0.02))  # 우측: -1.85m

        self.ransac_left.fit(X, y_l)
        self.ransac_right.fit(X, y_r)

    def preprocess_pts(self, lane_pts):
        # 기존 샘플링 전략 유지
        idx_list = []

        for d in np.arange(0, self.x_range, self.dx):
            idx_full_list = np.where(np.logical_and(lane_pts[0, :]>=d, lane_pts[0, :]<d+self.dx))[0].tolist()
            # 포인트가 적을 때는 최소 1개라도 선택하도록 수정
            if len(idx_full_list) > 0:
                sample_count = max(1, min(self.min_pts, len(idx_full_list)))
                idx_list += random.sample(idx_full_list, sample_count)

        if len(idx_list) == 0:
            rospy.logwarn("No points found in any segment")
            return np.array([]), np.array([]), np.array([]), np.array([])

        lane_pts = lane_pts[:, idx_list]
        
        x_g = np.copy(lane_pts[0, :])
        y_g = np.copy(lane_pts[1, :])

        if len(x_g) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # 실제 y 좌표 분포 확인 (디버깅용)
        rospy.loginfo_throttle(3.0, f"Y coordinate range: min={np.min(y_g):.2f}, max={np.max(y_g):.2f}, mean={np.mean(y_g):.2f}")

        # **간단한 중심 기준 좌우 분류**
        # 차량(BEV 하단 중심)을 기준으로 좌우 구분
        center_y = 0.0  # 차량 중심선 (Y=0)
        
        # 좌측: Y > 0 (양수), 우측: Y < 0 (음수)
        left_mask = y_g > center_y
        right_mask = y_g < center_y
        
        x_left = x_g[left_mask]
        y_left = y_g[left_mask]
        x_right = x_g[right_mask]
        y_right = y_g[right_mask]
        
        rospy.loginfo_throttle(2.0, f"Center-based classification - Left: {len(x_left)}, Right: {len(x_right)}")
        if len(y_left) > 0:
            rospy.loginfo_throttle(3.0, f"Left Y range: {np.min(y_left):.2f} to {np.max(y_left):.2f}")
        else:
            rospy.loginfo_throttle(3.0, "Left Y range: None")
            
        if len(y_right) > 0:
            rospy.loginfo_throttle(3.0, f"Right Y range: {np.min(y_right):.2f} to {np.max(y_right):.2f}")
        else:
            rospy.loginfo_throttle(3.0, "Right Y range: None")

        return x_left, y_left, x_right, y_right

    def fit_curve(self, lane_pts):
        if lane_pts.size == 0:
            # 검출 실패 시 이전 프레임 데이터 사용
            return self._handle_no_detection()
            
        x_left, y_left, x_right, y_right = self.preprocess_pts(lane_pts)
        
        # 디버깅 정보 추가
        rospy.loginfo_throttle(1.0, f"Preprocessed points - Left: {len(x_left)}, Right: {len(x_right)}")
        
        if len(y_left)==0 or len(y_right)==0:
            rospy.logwarn_throttle(1.0, "One side has no points, using fallback classification")
            # 중심 기준 분류는 이미 적용되었으므로, 빈 쪽에 가상 점 추가하지 않음
            # RANSAC 모델은 기존 초기화된 상태 유지

        if len(x_left) == 0 and len(x_right) == 0:
            rospy.logwarn("No points found on either side after preprocessing")
            return self._handle_no_detection()

        # 현재 프레임에서 차선 피팅 시도
        current_x_pred, current_y_pred_l, current_y_pred_r = self._fit_current_frame(x_left, y_left, x_right, y_right)
        
        if current_x_pred is None:
            return self._handle_no_detection()
        
        # 시간적 평활화 적용
        smoothed_x_pred, smoothed_y_pred_l, smoothed_y_pred_r = self._apply_temporal_smoothing(
            current_x_pred, current_y_pred_l, current_y_pred_r
        )
        
        # 성공적인 검출 후 상태 업데이트
        self.detection_confidence = min(1.0, self.detection_confidence + 0.2)
        self.no_detection_count = 0
        
        return smoothed_x_pred, smoothed_y_pred_l, smoothed_y_pred_r

    def _fit_current_frame(self, x_left, y_left, x_right, y_right):
        """현재 프레임 데이터로만 차선 피팅"""
        try:
            if len(y_left) >= self.ransac_left.min_samples:
                X_left = np.stack([x_left**i for i in reversed(range(1, self.order+1))]).T
                self.ransac_left.fit(X_left, y_left)
            
            if len(y_right) >= self.ransac_right.min_samples:
                X_right = np.stack([x_right**i for i in reversed(range(1, self.order+1))]).T
                self.ransac_right.fit(X_right, y_right)
        except:
            return None, None, None
    
        # predict the curve (내 코드 유지: 차량 바로 앞부터 시작)
        x_pred = np.arange(0.5, self.x_range, self.dx).astype(np.float32)
        X_pred = np.stack([x_pred**i for i in reversed(range(1, self.order+1))]).T
        
        try:
            y_pred_l = self.ransac_left.predict(X_pred)
            y_pred_r = self.ransac_right.predict(X_pred)
        except:
            return None, None, None

        if len(y_left) >= self.ransac_left.min_samples and len(y_right) >= self.ransac_right.min_samples:
            self.update_lane_width(y_pred_l, y_pred_r)

        if len(y_left) < self.ransac_left.min_samples:
            y_pred_l = y_pred_r + self.lane_width

        if len(y_right) < self.ransac_right.min_samples:
            y_pred_r = y_pred_l - self.lane_width

        return x_pred, y_pred_l, y_pred_r
    
    def _apply_temporal_smoothing(self, current_x_pred, current_y_pred_l, current_y_pred_r):
        """이전 프레임과 현재 프레임의 가중 평균"""
        if self.prev_y_pred_l is None or self.prev_y_pred_r is None:
            # 첫 번째 프레임이거나 이전 데이터 없음
            self.prev_x_pred = current_x_pred.copy()
            self.prev_y_pred_l = current_y_pred_l.copy()
            self.prev_y_pred_r = current_y_pred_r.copy()
            return current_x_pred, current_y_pred_l, current_y_pred_r
        
        # 이동평균 필터: previous * alpha + current * (1 - alpha)
        smoothed_y_pred_l = self.temporal_alpha * self.prev_y_pred_l + (1 - self.temporal_alpha) * current_y_pred_l
        smoothed_y_pred_r = self.temporal_alpha * self.prev_y_pred_r + (1 - self.temporal_alpha) * current_y_pred_r
        
        # 이전 프레임 데이터 업데이트
        self.prev_x_pred = current_x_pred.copy()
        self.prev_y_pred_l = smoothed_y_pred_l.copy()
        self.prev_y_pred_r = smoothed_y_pred_r.copy()
        
        rospy.loginfo_throttle(3.0, f"Applied temporal smoothing with alpha={self.temporal_alpha}")
        
        return current_x_pred, smoothed_y_pred_l, smoothed_y_pred_r
    
    def _handle_no_detection(self):
        """검출 실패 시 이전 프레임 데이터 사용"""
        self.no_detection_count += 1
        self.detection_confidence = max(0.0, self.detection_confidence - 0.3)
        
        # 연속 검출 실패가 너무 많으면 데이터 리셋
        if self.no_detection_count > 10:
            rospy.logwarn("Too many consecutive detection failures, resetting temporal data")
            self.prev_y_pred_l = None
            self.prev_y_pred_r = None
            self.prev_x_pred = None
            self.detection_confidence = 0.0
            return None, None, None
        
        # 이전 프레임 데이터가 있으면 그것을 사용
        if self.prev_y_pred_l is not None and self.prev_y_pred_r is not None:
            rospy.logwarn_throttle(1.0, f"No detection, using previous frame data (confidence: {self.detection_confidence:.2f})")
            return self.prev_x_pred.copy(), self.prev_y_pred_l.copy(), self.prev_y_pred_r.copy()
        
        return None, None, None

    def update_lane_width(self, y_pred_l, y_pred_r):
        # 실제 측정된 차선폭 계산
        measured_width = np.mean(y_pred_l - y_pred_r)
        
        # 합리적인 범위 내에서만 업데이트 (3.0m ~ 4.5m)
        if 3.0 <= measured_width <= 4.5:
            # 급격한 변화 방지: 기존 값과 측정값의 가중 평균
            self.lane_width = 0.9 * self.lane_width + 0.1 * measured_width
            rospy.loginfo_throttle(5.0, f"Lane width updated: {self.lane_width:.2f}m (measured: {measured_width:.2f}m)")
        else:
            rospy.logwarn_throttle(5.0, f"Unrealistic lane width detected: {measured_width:.2f}m, keeping current: {self.lane_width:.2f}m")

    def write_path_msg(self, x_pred, y_pred_l, y_pred_r, frame_id='/map'):
        self.lane_path = Path()
        self.lane_path.header.frame_id = frame_id

        for i in range(len(x_pred)):
            tmp_pose = PoseStamped()
            tmp_pose.pose.position.x = x_pred[i]
            tmp_pose.pose.position.y = (0.5)*(y_pred_l[i] + y_pred_r[i])
            tmp_pose.pose.position.z = 0
            tmp_pose.pose.orientation.x = 0
            tmp_pose.pose.orientation.y = 0
            tmp_pose.pose.orientation.z = 0
            tmp_pose.pose.orientation.w = 1
            self.lane_path.poses.append(tmp_pose)

    def get_lane_path(self):
        return self.lane_path


def extract_lane_points_from_bev(bev_mask, real_scale):
    """BEV 마스크에서 실제 좌표계 차선 점들을 추출"""
    if cv2.countNonZero(bev_mask) == 0:
        return np.zeros((4, 10))  # 빈 배열 반환
    
    # 흰색 픽셀 찾기
    lane_pixels = cv2.findNonZero(bev_mask)
    if lane_pixels is None:
        return np.zeros((4, 10))
    
    lane_pixels = lane_pixels.reshape([-1, 2])
    
    # BEV 픽셀 좌표를 실제 좌표계로 변환
    bev_height, bev_width = bev_mask.shape
    
    # BEV config에서 destination_points가 [0,0], [160,0], [160,800], [0,800]이므로
    # 픽셀 좌표를 실제 미터 좌표로 변환
    # Y=0이 상단, Y=800이 하단 (앞쪽이 하단)
    x_real = (bev_height - lane_pixels[:, 1]) * real_scale  # Y픽셀 -> X실제 (앞방향, 하단이 멀리)
    y_real = (bev_width/2 - lane_pixels[:, 0]) * real_scale  # X픽셀 -> Y실제 (좌우방향, 중심기준, 부호 반전)
    
    # 앞쪽 방향만 선택 (x > 0) 및 범위 제한
    valid_mask = (x_real > 0) & (x_real < 20) & (np.abs(y_real) < 3)
    x_real = x_real[valid_mask]
    y_real = y_real[valid_mask]
    
    if len(x_real) == 0:
        return np.zeros((4, 10))
    
    # 4xN 형태로 반환 (x, y, z=0, w=1)
    xyz_g = np.array([
        x_real,
        y_real,
        np.zeros_like(x_real),
        np.ones_like(x_real)
    ])
    
    return xyz_g



def draw_fitted_lanes_on_bev(bev_image, x_pred, y_pred_l, y_pred_r, real_scale):
    """BEV 이미지에 fitting된 차선을 그리는 함수"""
    if x_pred is None or y_pred_l is None or y_pred_r is None:
        return bev_image
    
    bev_with_lanes = cv2.cvtColor(bev_image.copy(), cv2.COLOR_GRAY2BGR) if len(bev_image.shape) == 2 else bev_image.copy()
    
    bev_height, bev_width = bev_image.shape[:2]
    
    # 실제 좌표를 BEV 픽셀 좌표로 변환 (extract_lane_points_from_bev와 동일한 방식)
    for i in range(len(x_pred)):
        # 좌측 차선 - 실제 좌표를 픽셀 좌표로 변환
        pixel_x_l = int(bev_width/2 - y_pred_l[i] / real_scale)  # y_real -> x_pixel (부호 반전)
        pixel_y_l = int(bev_height - x_pred[i] / real_scale)      # x_real -> y_pixel (역변환)
        
        # 우측 차선
        pixel_x_r = int(bev_width/2 - y_pred_r[i] / real_scale)  # y_real -> x_pixel (부호 반전)
        pixel_y_r = int(bev_height - x_pred[i] / real_scale)      # x_real -> y_pixel (역변환)
        
        # BEV 이미지 범위 내에서만 그리기
        if 0 <= pixel_x_l < bev_width and 0 <= pixel_y_l < bev_height:
            cv2.circle(bev_with_lanes, (pixel_x_l, pixel_y_l), 3, (0, 255, 0), -1)  # 초록색 (좌측)
        
        if 0 <= pixel_x_r < bev_width and 0 <= pixel_y_r < bev_height:
            cv2.circle(bev_with_lanes, (pixel_x_r, pixel_y_r), 3, (0, 0, 255), -1)  # 빨간색 (우측)
    
    return bev_with_lanes

class BEVNode:
    def __init__(self):
        rospy.init_node('cream_ioniq_node', anonymous=True)
        
        # 파라미터 로드
        self.weights = rospy.get_param('~weights', '')
        self.img_size = rospy.get_param('~img_size', 640)
        self.image_topic = rospy.get_param('~image_topic', '/image_raw')
        self.compressed_input = rospy.get_param('~compressed_input', False)
        self.device = rospy.get_param('~device', 'cpu')
        
        # 모델 파일 존재 확인
        if not os.path.exists(self.weights):
            rospy.logerr(f"Model weights file not found: {self.weights}")
            return
            
        # 디바이스 설정
        self.device = select_device(self.device)
        
        # 모델 로딩
        rospy.loginfo("Loading YOLOPv2 model...")
        self.model = torch.jit.load(self.weights)
        self.model = self.model.to(self.device)
        self.half = self.device.type != 'cpu'
        
        if self.half:
            self.model.half()
        self.model.eval()
        
        # 모델 초기화를 위한 더미 입력
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))
        
        # cv_bridge 초기화
        self.bridge = CvBridge()
        
        # BEV Transform 초기화
        self.bev_config_transform = None
        self.setup_bev_transform()
        
        # Lane fitting 초기화 (시간적 평활화 추가)
        self.curve_fitter = CURVEFit(
            order=3,
            alpha=5,         # 내 코드 유지
            lane_width=3.7,  # 차선 폭을 3.7m로 설정
            y_margin=0.8,    # 차선 분리를 위해 더 엄격하게 설정
            x_range=20,      # 내 코드 유지 (더 긴 범위)
            dx=0.5,
            min_pts=8,       # 적은 포인트 상황에서도 작동하도록 감소
            max_tri=50,      # 더 많은 시도로 정확도 향상
            temporal_alpha=0.7  # 이동평균 필터: 70% 이전 + 30% 현재
        )
        
        # Lane visualizer 초기화
        self.lane_visualizer = LaneVisualizer()
        
        # 발행자 초기화
        self.lane_seg_pub = rospy.Publisher('lane_seg', Image, queue_size=1)
        self.lane_seg_processed_pub = rospy.Publisher('lane_seg_processed', Image, queue_size=1)
        self.lane_seg_bev_pub = rospy.Publisher('lane_seg_bev', Image, queue_size=1)
        self.original_bev_pub = rospy.Publisher('original_bev', Image, queue_size=1)
        self.lane_path_pub = rospy.Publisher('lane_path', Path, queue_size=1)
        self.lane_fitted_bev_pub = rospy.Publisher('lane_fitted_bev', Image, queue_size=1)
        self.lane_markers_pub = rospy.Publisher('lane_markers', MarkerArray, queue_size=1)
        self.lane_boundary_markers_pub = rospy.Publisher('lane_boundary_markers', MarkerArray, queue_size=1)
        self.lane_info_markers_pub = rospy.Publisher('lane_info_markers', MarkerArray, queue_size=1)
        
        # 구독자 초기화
        if self.compressed_input:
            self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        else:
            self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        
        rospy.loginfo("CREAM IONIQ node initialized. Waiting for images...")
        
    def setup_bev_transform(self):
        try:
            if BEV_CONFIG_AVAILABLE:
                bev_config_path = rospy.get_param('~bev_config_path', 
                    '/home/mini/catkin_ws/src/cream_ioniq/bev_config.json')
                
                self.bev_config_transform = BEVConfigTransform(bev_config_path)
                if self.bev_config_transform.is_loaded():
                    rospy.loginfo("BEV Config Transform initialized successfully")
                    config_info = self.bev_config_transform.get_config_info()
                    if config_info:
                        real_w = config_info['real_dimensions']['width_meters']
                        real_h = config_info['real_dimensions']['height_meters']
                        rospy.loginfo(f"Real-world scale: {config_info['real_scale']} m/px")
                        rospy.loginfo(f"Real-world area: {real_w:.1f}m x {real_h:.1f}m")
                else:
                    rospy.logerr("BEV Config Transform failed to load")
                    self.bev_config_transform = None
            else:
                rospy.logerr("BEV Config Transform not available")
                self.bev_config_transform = None
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize BEV Config Transform: {e}")
            self.bev_config_transform = None
    
    def aggressive_horizontal_removal(self, mask):
        """매우 공격적인 가로선 제거"""
        try:
            filtered_mask = mask.copy()
            h, w = mask.shape
            
            # 1. 행별 스캔으로 가로선 검출 및 제거
            for y in range(h):
                row = mask[y, :]
                white_pixels = np.sum(row > 0)
                
                # 해당 행에서 흰 픽셀이 너무 많으면 가로선으로 판단
                if white_pixels > w * 0.15:  # 행의 15% 이상이 흰색
                    # 연속된 흰 픽셀 구간 찾기
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
                    
                    # 긴 가로 구간들 제거
                    for start_x, end_x in white_regions:
                        length = end_x - start_x + 1
                        if length > 30:  # 30픽셀 이상 연속된 가로선
                            filtered_mask[y, start_x:end_x+1] = 0
            
            # 2. 세로 방향으로만 연결된 픽셀 보존
            vertical_kernel = np.array([[0, 1, 0],
                                       [0, 1, 0],
                                       [0, 1, 0]], dtype=np.uint8)
            vertical_preserved = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, vertical_kernel)
            
            # 3. 대각선 방향도 어느정도 보존
            diagonal_kernel = np.array([[1, 0, 0],
                                       [0, 1, 0], 
                                       [0, 0, 1]], dtype=np.uint8)
            diagonal_preserved = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, diagonal_kernel)
            
            # 세로선과 대각선 결합
            result = cv2.bitwise_or(vertical_preserved, diagonal_preserved)
            
            
            return result
            
        except Exception as e:
            rospy.logwarn(f"Error in aggressive horizontal removal: {e}")
            return mask
    
    def filter_horizontal_lines(self, mask, horizontal_ratio_threshold=2.5, min_height_threshold=20):
        """강화된 가로방향 선 제거 함수"""
        try:
            filtered_mask = mask.copy()
            
            # 연결된 컴포넌트 분석
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            for i in range(1, num_labels):  # 0은 배경이므로 제외
                x, y, w, h, area = stats[i]
                
                # 더 엄격한 가로선 판단 기준
                should_remove = False
                
                # 1. 가로세로 비율이 임계값보다 큰 경우
                if h > 0 and (w / h) >= horizontal_ratio_threshold:
                    should_remove = True
                
                # 2. 높이가 너무 작은 경우 (얇은 가로선)
                elif h <= 5 and w > 50:
                    should_remove = True  
                
                # 3. 화면 하단의 가로로 긴 객체 (정지선 가능성 높음)
                elif y > mask.shape[0] * 0.7 and h > 0 and (w / h) >= 2.0:
                    should_remove = True
                
                if should_remove:
                    filtered_mask[labels == i] = 0
            
            return filtered_mask
                
        except Exception as e:
            rospy.logwarn(f"Error in horizontal line filtering: {e}")
            return mask
    
    def remove_horizontal_lines_morphology(self, mask):
        """강화된 모폴로지 연산으로 가로선 제거"""
        try:
            # 더 큰 커널로 확실하게 가로선 검출
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
            horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
            
            # 세로 방향 커널로 세로선만 보존
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 30))  
            vertical_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
            
            # 대각선도 보존 (좌상-우하)
            diagonal_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            diagonal_lines1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, diagonal_kernel1)
            
            # 원본에서 가로선 완전 제거
            filtered_mask = cv2.subtract(mask, horizontal_lines)
            
            # 세로선과 대각선은 보존
            result = cv2.bitwise_or(filtered_mask, vertical_lines)
            
            
            return result
                
        except Exception as e:
            rospy.logwarn(f"Error in morphology horizontal line filtering: {e}")
            return mask
    
    def thin_lanes(self, mask):
        """차선을 매우 얇게 만드는 함수"""
        try:
            # 모폴로지 연산으로 두꺼운 선을 얇게 만들기
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # 먼저 약간의 closing으로 끊어진 선 연결
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 스켈레톤화 (Zhang-Suen 알고리즘)
            skeleton = self.skeletonize(closed)
            
            # 스켈레톤을 아주 약간만 두껍게 만들기
            thicken_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # 작은 커널
            thickened = cv2.dilate(skeleton, thicken_kernel, iterations=1)  # 1번만 확장
            
            
            return thickened
            
        except Exception as e:
            rospy.logwarn(f"Error in lane thinning: {e}")
            return mask
    
    def skeletonize(self, img):
        """Zhang-Suen 스켈레톤화 알고리즘"""
        img = img.copy()
        img[img > 0] = 1  # 이진화
        
        skeleton = np.zeros(img.shape, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()
            
            if cv2.countNonZero(img) == 0:
                break
        
        skeleton[skeleton > 0] = 255
        return skeleton
        
    def image_callback(self, data):
        try:
            # 이미지 변환
            if self.compressed_input:
                cv_img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            else:
                cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # 원본 이미지 저장 (BEV용)
            original_for_bev = cv_img.copy()
            if original_for_bev.shape[1] != 1920 or original_for_bev.shape[0] != 1080:
                original_for_bev = cv2.resize(original_for_bev, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            
            # YOLO 추론용 이미지 크기 조정
            im0s = cv2.resize(cv_img, (1280, 720), interpolation=cv2.INTER_LINEAR)
            
            # Letterbox 처리
            img, ratio, (dw, dh) = letterbox(im0s, self.img_size, stride=32)
            
            # BGR -> RGB, HWC -> CHW
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            
            # 텐서 변환
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # 추론
            t1 = time_synchronized()
            [pred, anchor_grid], seg, ll = self.model(img)
            t2 = time_synchronized()
            
            # 세그멘테이션 마스크 추출
            ll_seg_mask = lane_line_mask(ll)
            
            # 마스크를 8비트 이미지로 변환
            ll_seg_mask_8u = np.zeros(ll_seg_mask.shape, dtype=np.uint8)
            ll_seg_mask_8u[ll_seg_mask > 0] = 255
            
            # BEV 변환 적용 (차선 세그멘테이션 마스크에 대해)
            ll_seg_mask_bev = None
            
            if self.bev_config_transform is not None:
                try:
                    # 차선 마스크를 원본 해상도로 복원
                    ll_seg_original_size = cv2.resize(ll_seg_mask_8u, (1920, 1080), interpolation=cv2.INTER_NEAREST)
                    
                    # 차선 마스크 전처리: 가로선 완전 제거 후 스켈레톤화
                    # 1단계: 강력한 가로선 제거 (두께 증가 전)
                    ll_step1 = self.aggressive_horizontal_removal(ll_seg_original_size)
                    
                    # 2단계: 모폴로지 연산으로 추가 가로선 제거
                    ll_step2 = self.remove_horizontal_lines_morphology(ll_step1)
                    
                    # 3단계: 컴포넌트 분석으로 남은 가로선 제거  
                    ll_filtered = self.filter_horizontal_lines(ll_step2, horizontal_ratio_threshold=1.8)
                    
                    # 4단계: 마지막에 차선을 얇게 만들기 (BEV 변환용)
                    ll_thinned = self.thin_lanes(ll_filtered)
                    
                    # BEV 변환 (전처리된 차선 마스크 사용)
                    ll_seg_mask_bev = self.bev_config_transform.transform_to_bev(ll_thinned, apply_undistort=True)
                    
                    # 원본 이미지도 BEV 변환
                    original_bev = self.bev_config_transform.transform_to_bev(original_for_bev, apply_undistort=True)
                    
                    # Lane fitting 수행
                    config_info = self.bev_config_transform.get_config_info()
                    real_scale = config_info['real_scale']
                    
                    # BEV 차선 마스크에서 실제 좌표 점들 추출
                    lane_pts = extract_lane_points_from_bev(ll_seg_mask_bev, real_scale)
                    rospy.loginfo(f"Lane points extracted: {lane_pts.shape}, non-zero pixels: {cv2.countNonZero(ll_seg_mask_bev)}")
                    
                    # 차선 피팅
                    x_pred, y_pred_l, y_pred_r = self.curve_fitter.fit_curve(lane_pts)
                    
                    if x_pred is not None:
                        rospy.loginfo(f"Lane fitting successful: {len(x_pred)} points")
                    else:
                        rospy.logwarn("Lane fitting failed")
                        # 실패 시 마커 지우기
                        clear_markers = self.lane_visualizer.clear_all_markers(data.header.stamp, frame_id='ego_car')
                        self.lane_markers_pub.publish(clear_markers)
                        self.lane_boundary_markers_pub.publish(MarkerArray())
                        self.lane_info_markers_pub.publish(MarkerArray())
                    
                    # Lane path 메시지 생성 및 발행
                    if x_pred is not None:
                        self.curve_fitter.write_path_msg(x_pred, y_pred_l, y_pred_r, frame_id='ego_car')
                        lane_path = self.curve_fitter.get_lane_path()
                        lane_path.header.stamp = data.header.stamp
                        self.lane_path_pub.publish(lane_path)
                        
                        # Lane markers 생성 및 발행 (기본 색상)
                        lane_markers = self.lane_visualizer.create_lane_markers(
                            x_pred, y_pred_l, y_pred_r, data.header.stamp, frame_id='ego_car'
                        )
                        self.lane_markers_pub.publish(lane_markers)
                        
                        # Lane boundary markers 생성 및 발행
                        boundary_markers = self.lane_visualizer.create_lane_boundary_markers(
                            x_pred, y_pred_l, y_pred_r, data.header.stamp, frame_id='ego_car'
                        )
                        self.lane_boundary_markers_pub.publish(boundary_markers)
                        
                        # Lane info text markers 생성 및 발행
                        info_markers = self.lane_visualizer.create_lane_info_text_marker(
                            x_pred, y_pred_l, y_pred_r, data.header.stamp, frame_id='ego_car'
                        )
                        self.lane_info_markers_pub.publish(info_markers)
                        
                        rospy.loginfo_throttle(2.0, f"Published lane markers: {len(x_pred)} points, "
                                             f"avg width: {np.mean(y_pred_l - y_pred_r):.2f}m")
                        
                        # Fitted lanes를 BEV 이미지에 그리기
                        lane_fitted_bev = draw_fitted_lanes_on_bev(ll_seg_mask_bev, x_pred, y_pred_l, y_pred_r, real_scale)
                        
                        # Fitted BEV 이미지 발행
                        try:
                            lane_fitted_bev_msg = self.bridge.cv2_to_imgmsg(lane_fitted_bev, "bgr8")
                            lane_fitted_bev_msg.header = data.header
                            self.lane_fitted_bev_pub.publish(lane_fitted_bev_msg)
                        except Exception as e:
                            rospy.logwarn(f"Failed to publish fitted BEV: {e}")
                    
                    # 전처리된 차선 마스크 발행 (원본 크기로 다시 축소)
                    ll_processed_resized = cv2.resize(ll_thinned, (ll_seg_mask_8u.shape[1], ll_seg_mask_8u.shape[0]), interpolation=cv2.INTER_NEAREST)
                    lane_seg_processed_msg = self.bridge.cv2_to_imgmsg(ll_processed_resized, "mono8")
                    lane_seg_processed_msg.header = data.header
                    self.lane_seg_processed_pub.publish(lane_seg_processed_msg)
                    
                except Exception as e:
                    rospy.logwarn(f"BEV transformation failed: {e}")
            else:
                rospy.logwarn_throttle(10.0, "BEV Config Transform not initialized")
            
            # 결과 발행
            try:
                # 원본 세그멘테이션 결과
                lane_seg_msg = self.bridge.cv2_to_imgmsg(ll_seg_mask_8u, "mono8")
                lane_seg_msg.header = data.header
                self.lane_seg_pub.publish(lane_seg_msg)
                
                
                # BEV 세그멘테이션 결과
                if ll_seg_mask_bev is not None:
                    lane_seg_bev_msg = self.bridge.cv2_to_imgmsg(ll_seg_mask_bev, "mono8")
                    lane_seg_bev_msg.header = data.header
                    self.lane_seg_bev_pub.publish(lane_seg_bev_msg)
                
                # 원본 BEV 이미지 발행
                if 'original_bev' in locals() and original_bev is not None:
                    original_bev_msg = self.bridge.cv2_to_imgmsg(original_bev, "bgr8")
                    original_bev_msg.header = data.header
                    self.original_bev_pub.publish(original_bev_msg)
                
                
            except CvBridgeError as e:
                rospy.logerr(f"Error converting image to ROS message: {e}")
            
            
        except CvBridgeError as e:
            rospy.logerr(e)
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

if __name__ == '__main__':
    try:
        node = BEVNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass