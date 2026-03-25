"""
人脸检测模块 - 支持 MTCNN、YOLOv8、MediaPipe 等多个引擎
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FaceDetector:
    """人脸检测器基类"""
    
    def __init__(self, model_type='mtcnn'):
        """
        初始化人脸检测器
        
        Args:
            model_type: 'mtcnn' | 'mediapipe' | 'yolov8'
        """
        self.model_type = model_type
        self.detector = None
        self._init_model()
    
    def _init_model(self):
        """初始化对应的检测模型"""
        if self.model_type == 'mtcnn':
            self._init_mtcnn()
        elif self.model_type == 'mediapipe':
            self._init_mediapipe()
        elif self.model_type == 'yolov8':
            self._init_yolov8()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _init_mtcnn(self):
        """使用 MTCNN 进行人脸检测"""
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            logger.info("[OK] MTCNN 模型加载成功")
        except ImportError:
            logger.error("MTCNN 未安装，请运行: pip install mtcnn")
            raise
    
    def _init_mediapipe(self):
        """使用 MediaPipe 进行人脸检测"""
        try:
            import mediapipe as mp
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,  # 0=short-range, 1=full-range
                min_detection_confidence=0.5
            )
            logger.info("[OK] MediaPipe 人脸检测器加载成功")
        except ImportError:
            logger.error("MediaPipe 未安装，请运行: pip install mediapipe")
            raise
    
    def _init_yolov8(self):
        """使用 YOLOv8 进行人脸检测"""
        try:
            from ultralytics import YOLO
            self.detector = YOLO('yolov8n-face.pt')  # nano 模型，速度快
            logger.info("[OK] YOLOv8 人脸检测器加载成功")
        except ImportError:
            logger.error("YOLOv8 未安装，请运行: pip install ultralytics")
            raise
    
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        检测图片中的所有人脸
        
        Args:
            image: 输入图片 (numpy array, BGR 格式)
            confidence_threshold: 置信度阈值
        
        Returns:
            人脸信息列表，每个元素包含:
            {
                'bbox': [x1, y1, x2, y2],      # 人脸边界框
                'landmarks': [...],             # 人脸关键点
                'confidence': float,            # 置信度
                'face_id': int                  # 人脸ID
            }
        """
        if self.model_type == 'mtcnn':
            return self._detect_mtcnn(image, confidence_threshold)
        elif self.model_type == 'mediapipe':
            return self._detect_mediapipe(image, confidence_threshold)
        elif self.model_type == 'yolov8':
            return self._detect_yolov8(image, confidence_threshold)
    
    def _detect_mtcnn(self, image: np.ndarray, confidence_threshold: float) -> List[Dict]:
        """MTCNN 检测"""
        faces = self.detector.detect_faces(image)
        
        results = []
        for i, face in enumerate(faces):
            if face['confidence'] >= confidence_threshold:
                bbox = face['box']
                # 转换为 [x1, y1, x2, y2] 格式
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': face['confidence'],
                    'landmarks': face['keypoints'],
                    'face_id': i
                })
        
        return results
    
    def _detect_mediapipe(self, image: np.ndarray, confidence_threshold: float) -> List[Dict]:
        """MediaPipe 检测"""
        import mediapipe as mp
        
        h, w, c = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)
        
        faces = []
        if results.detections:
            for i, detection in enumerate(results.detections):
                if detection.score[0] >= confidence_threshold:
                    bbox = detection.location_data.relative_bounding_box
                    
                    # 转换相对坐标到像素坐标
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    
                    faces.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(detection.score[0]),
                        'landmarks': None,  # MediaPipe 需要另外处理landmarks
                        'face_id': i
                    })
        
        return faces
    
    def _detect_yolov8(self, image: np.ndarray, confidence_threshold: float) -> List[Dict]:
        """YOLOv8 检测"""
        results = self.detector(image, conf=confidence_threshold, verbose=False)
        
        faces = []
        for i, result in enumerate(results):
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    faces.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'landmarks': None,
                        'face_id': i
                    })
        
        return faces
    
    def draw_faces(self, image: np.ndarray, faces: List[Dict], 
                   draw_landmarks: bool = True) -> np.ndarray:
        """
        在图片上绘制人脸检测结果
        
        Args:
            image: 输入图片
            faces: 人脸检测结果列表
            draw_landmarks: 是否绘制关键点
        
        Returns:
            标注后的图片
        """
        output = image.copy()
        
        for face in faces:
            bbox = face['bbox']
            x1, y1, x2, y2 = bbox
            
            # 绘制边界框
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制置信度
            confidence = face['confidence']
            cv2.putText(output, f"{confidence:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制关键点
            if draw_landmarks and face.get('landmarks'):
                landmarks = face['landmarks']
                for landmark in landmarks.values():
                    cv2.circle(output, tuple(map(int, landmark)), 3, (0, 0, 255), -1)
        
        return output


# 简化使用接口
def quick_detect(image_path: str, model_type: str = 'mtcnn') -> List[Dict]:
    """快速检测图片中的人脸"""
    image = cv2.imread(image_path)
    detector = FaceDetector(model_type=model_type)
    return detector.detect(image)


def quick_detect_video(video_path: str, model_type: str = 'mtcnn', 
                       sample_frames: int = 5) -> Dict:
    """
    快速检测视频中的人脸（每隔 sample_frames 帧检测一次）
    
    Returns:
        {
            'frame_index': [人脸列表],
            ...
        }
    """
    cap = cv2.VideoCapture(video_path)
    detector = FaceDetector(model_type=model_type)
    
    results = {}
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_frames == 0:
            faces = detector.detect(frame)
            if faces:
                results[frame_idx] = faces
        
        frame_idx += 1
    
    cap.release()
    return results


if __name__ == '__main__':
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        faces = quick_detect(image_path, model_type='mtcnn')
        
        print(f"[OK] 检测到 {len(faces)} 张人脸")
        for face in faces:
            print(f"  - ID: {face['face_id']}, 置信度: {face['confidence']:.2f}, "
                  f"位置: {face['bbox']}")
