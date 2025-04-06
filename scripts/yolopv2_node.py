#!/usr/bin/env python3
import rospy
import torch
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import os
import sys

# Add the YOLOPv2 utils directory to the Python path
yolopv2_ros_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_dir = os.path.join(yolopv2_ros_dir, 'utils')
sys.path.append(utils_dir)

from utils import (
    time_synchronized, select_device, 
    lane_line_mask, driving_area_mask
)

class YOLOPv2Node:
    def __init__(self):
        rospy.init_node('yolopv2_node', anonymous=True)
        
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
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        
        if self.half:
            self.model.half()  # to FP16  
        self.model.eval()
        
        # 모델 초기화를 위한 더미 입력
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))
            
        # cv_bridge 초기화
        self.bridge = CvBridge()
        
        # 발행자 초기화
        self.lane_seg_pub = rospy.Publisher('yolo_lane_seg', Image, queue_size=1)
        self.drive_area_pub = rospy.Publisher('yolo_drive_area', Image, queue_size=1)
        self.result_pub = rospy.Publisher('yolo_result_image', Image, queue_size=1)
        
        # 구독자 초기화
        if self.compressed_input:
            self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        else:
            self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        
        rospy.loginfo("YOLOPv2 node initialized. Waiting for images...")
        
    def image_callback(self, data):
        try:
            # 이미지 변환
            if self.compressed_input:
                cv_img = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            else:
                cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # 원본 이미지 저장
            im0s = cv_img.copy()  # 원본 이미지 유지
            
            # 이미지 크기 조정 (원본 YOLOPv2 코드는 720p로 리사이즈)
            im0s = cv2.resize(im0s, (1280, 720), interpolation=cv2.INTER_LINEAR)
            orig_h, orig_w = im0s.shape[:2]
            
            # Letterbox 처리 (원본 YOLOPv2 처럼 처리)
            img, ratio, (dw, dh) = self.letterbox(im0s, self.img_size, stride=32)
            
            # BGR -> RGB, HWC -> CHW
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            
            # 텐서 변환
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # 추론
            t1 = time_synchronized()
            [pred, anchor_grid], seg, ll = self.model(img)
            t2 = time_synchronized()
            
            # 세그멘테이션 마스크 추출 (원본 YOLOPv2와 동일)
            da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)
            
            # 결과 시각화 (원본 YOLOPv2와 동일)
            result_img = im0s.copy()
            self.show_seg_result(result_img, (da_seg_mask, ll_seg_mask))
            
            # 마스크를 8비트 이미지로 변환
            da_seg_mask_8u = np.zeros(da_seg_mask.shape, dtype=np.uint8)
            ll_seg_mask_8u = np.zeros(ll_seg_mask.shape, dtype=np.uint8)
            da_seg_mask_8u[da_seg_mask > 0] = 255
            ll_seg_mask_8u[ll_seg_mask > 0] = 255
            
            # 결과 발행
            try:
                lane_seg_msg = self.bridge.cv2_to_imgmsg(ll_seg_mask_8u, "mono8")
                lane_seg_msg.header = data.header
                self.lane_seg_pub.publish(lane_seg_msg)
                
                drive_area_msg = self.bridge.cv2_to_imgmsg(da_seg_mask_8u, "mono8")
                drive_area_msg.header = data.header
                self.drive_area_pub.publish(drive_area_msg)
                
                result_msg = self.bridge.cv2_to_imgmsg(result_img, "bgr8")
                result_msg.header = data.header
                self.result_pub.publish(result_msg)
            except CvBridgeError as e:
                rospy.logerr(f"Error converting image to ROS message: {e}")
            
            # 로그 정보
            fps = 1.0 / (t2 - t1)
            rospy.logdebug(f"Inference time: {t2 - t1:.4f}s, FPS: {fps:.1f}")
            
        except CvBridgeError as e:
            rospy.logerr(e)
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
            
    def show_seg_result(self, img, result, palette=None, is_demo=False):
        """
        원본 YOLOPv2 show_seg_result 함수를 재구현
        """
        if palette is None:
            palette = np.random.randint(0, 255, size=(3, 3))
            palette[0] = [0, 0, 0]
            palette[1] = [0, 255, 0]
            palette[2] = [255, 0, 0]
        palette = np.array(palette)
        
        # 원본 YOLOPv2 구현과 동일하게 처리
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        color_area[result[0] == 1] = [0, 255, 0]  # 주행 영역 - 녹색
        color_area[result[1] == 1] = [255, 0, 0]  # 차선 - 빨간색
        
        # BGR로 변환 (OpenCV 형식)
        color_seg = color_area[..., ::-1]
        
        # 오버레이 적용
        color_mask = np.mean(color_seg, 2)
        img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
        
        return img
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)

if __name__ == '__main__':
    try:
        node = YOLOPv2Node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 