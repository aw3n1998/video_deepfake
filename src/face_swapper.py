"""
人脸交换模块 - 基于 Deepfaceslive / InsightFace
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 模块级工作函数（必须在类外，multiprocessing 才能序列化）
# ─────────────────────────────────────────────────────────────────────────────
def _worker_swap_segment(args):
    """子进程工作函数：独立加载模型、处理一段视频"""
    source_image_path, input_path, output_path, worker_id, total_workers = args

    # 子进程独立配置日志
    _log = logging.getLogger(f'worker_{worker_id}')
    if not _log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter(
            f'[%(asctime)s][Worker {worker_id}/{total_workers}] %(message)s',
            datefmt='%H:%M:%S'
        ))
        _log.addHandler(_h)
        _log.setLevel(logging.INFO)

    try:
        swapper = FaceSwapper()          # 每个子进程独立加载模型
        ok = swapper.swap_male_faces_in_video(source_image_path, input_path, output_path)
        return worker_id, ok, output_path
    except Exception as exc:
        return worker_id, False, str(exc)


class FaceSwapper:
    """使用 InsightFace 进行人脸交换"""
    
    def __init__(self, model_name: str = 'inswapper_128.onnx'):
        """
        初始化人脸交换模型
        
        Args:
            model_name: 使用的模型名称
        """
        self.model_name = model_name
        self.face_swapper = None
        self.face_analyser = None
        self._init_models()
    
    def _init_models(self):
        """初始化 InsightFace 模型"""
        try:
            import insightface
            
            # 初始化人脸分析器（用于获取人脸特征）
            self.face_analyser = insightface.app.FaceAnalysis(
                name='buffalo_l',  # 高精度模型
                providers=['CPUExecutionProvider']  # 或 ['CUDAExecutionProvider'] 如果有 GPU
            )
            self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
            
            # 初始化人脸交换器
            self.face_swapper = insightface.model_zoo.get_model(
                f'models/{self.model_name}'
            )
            
            logger.info(f"[OK] InsightFace 模型加载成功 ({self.model_name})")
            
        except ImportError:
            logger.error("InsightFace 未安装，请运行: pip install insightface")
            raise
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def get_face_embedding(self, image: np.ndarray, face_index: int = 0):
        """
        获取图片中指定人脸的特征向量
        
        Args:
            image: 输入图片
            face_index: 人脸索引（如果有多个人脸）
        
        Returns:
            人脸特征对象
        """
        try:
            faces = self.face_analyser.get(image)
            if len(faces) == 0:
                logger.warning("图片中未检测到人脸")
                return None
            
            if face_index >= len(faces):
                logger.warning(f"指定的人脸索引 {face_index} 超出范围")
                return faces[0]
            
            return faces[face_index]
        
        except Exception as e:
            logger.error(f"获取人脸特征失败: {e}")
            return None
    
    def swap_faces(self, source_image: np.ndarray, target_image: np.ndarray,
                   source_face_index: int = 0, target_face_index: int = 0) -> Optional[np.ndarray]:
        """
        将源图片中的人脸转移到目标图片中
        
        Args:
            source_image: 源图片 (提供人脸特征的图片)
            target_image: 目标图片 (被替换人脸的图片)
            source_face_index: 源图片中的人脸索引
            target_face_index: 目标图片中的人脸索引
        
        Returns:
            交换后的图片，如果失败则返回 None
        """
        try:
            # 获取两个图片中的人脸特征
            source_face = self.get_face_embedding(source_image, source_face_index)
            if source_face is None:
                logger.error("源图片中未检测到有效人脸")
                return None
            
            target_faces = self.face_analyser.get(target_image)
            if len(target_faces) == 0:
                logger.error("目标图片中未检测到人脸")
                return None
            
            if target_face_index >= len(target_faces):
                logger.warning(f"目标人脸索引超出范围，使用第一张人脸")
                target_face_index = 0
            
            # 执行人脸交换
            result_image = target_image.copy()
            result_image = self.face_swapper.get(
                result_image, 
                target_faces[target_face_index], 
                source_face, 
                paste_back=True
            )
            
            logger.info("[OK] 人脸交换成功")
            return result_image
        
        except Exception as e:
            logger.error(f"人脸交换失败: {e}")
            return None
    
    def swap_faces_in_video(self, source_image_path: str, video_path: str,
                           output_path: str, progress_callback=None) -> bool:
        """
        在视频的所有帧中进行人脸交换

        Args:
            source_image_path: 源人脸图片路径
            video_path: 输入视频路径
            output_path: 输出视频路径
            progress_callback: 进度回调函数

        Returns:
            成功返回 True
        """
        try:
            # 加载源人脸图片
            source_image = cv2.imread(source_image_path)
            if source_image is None:
                logger.error(f"无法读取源图片: {source_image_path}")
                return False

            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return False

            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 初始化视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 进行人脸交换
                swapped_frame = self.swap_faces(source_image, frame)
                if swapped_frame is not None:
                    out.write(swapped_frame)
                else:
                    # 如果交换失败，写入原始帧
                    out.write(frame)

                frame_count += 1

                # 进度回调
                if progress_callback:
                    progress = frame_count / total_frames
                    progress_callback(progress)

                if frame_count % 30 == 0:
                    logger.info(f"处理进度: {frame_count}/{total_frames} 帧 "
                              f"({100*frame_count/total_frames:.1f}%)")

            cap.release()
            out.release()

            logger.info(f"[OK] 视频处理完成，输出: {output_path}")
            return True

        except Exception as e:
            logger.error(f"视频处理失败: {e}")
            return False

    def swap_male_faces_in_video(self, source_image_path: str, video_path: str,
                                 output_path: str, progress_callback=None) -> bool:
        """
        在视频所有帧中只替换男性人脸（gender==1），女性人脸保持不变。
        使用 InsightFace buffalo_l 模型内置的性别属性进行过滤。

        Args:
            source_image_path: 替换用的女性人脸图片路径
            video_path: 输入视频路径
            output_path: 输出视频路径（无音频）
            progress_callback: 进度回调函数 (0.0~1.0)

        Returns:
            成功返回 True
        """
        try:
            source_image = cv2.imread(source_image_path)
            if source_image is None:
                logger.error(f"无法读取源图片: {source_image_path}")
                return False

            # 获取替换用的源人脸特征
            source_face = self.get_face_embedding(source_image)
            if source_face is None:
                logger.error("源图片中未检测到有效人脸")
                return False

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return False

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            male_swap_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result_frame = frame.copy()
                target_faces = self.face_analyser.get(frame)

                for target_face in target_faces:
                    # InsightFace gender: 0=female, 1=male
                    if getattr(target_face, 'gender', -1) == 1:
                        result_frame = self.face_swapper.get(
                            result_frame, target_face, source_face, paste_back=True
                        )
                        male_swap_count += 1

                out.write(result_frame)
                frame_count += 1

                if progress_callback:
                    progress_callback(frame_count / total_frames)

                if frame_count % 30 == 0:
                    logger.info(
                        f"处理进度: {frame_count}/{total_frames} 帧 "
                        f"({100 * frame_count / total_frames:.1f}%) | "
                        f"累计替换男脸: {male_swap_count} 次"
                    )

            cap.release()
            out.release()

            logger.info(f"[OK] 男性人脸替换完成，共替换 {male_swap_count} 次，输出: {output_path}")
            return True

        except Exception as e:
            logger.error(f"男性人脸替换失败: {e}")
            return False

    def swap_male_faces_in_video_parallel(self, source_image_path: str, video_path: str,
                                          output_path: str, n_workers: int = None) -> bool:
        """
        并行版：把视频切成 n_workers 段，多进程同时换脸，最后合并。

        Args:
            source_image_path: 替换用女性人脸图片
            video_path: 输入视频
            output_path: 输出视频（无音频）
            n_workers: 并行进程数（默认 = CPU核心数 // 2，最少2，最多8）

        Returns:
            成功返回 True
        """
        import multiprocessing
        import subprocess
        import shutil

        # ── 确定进程数 ──────────────────────────────────────────────────────
        cpu_count = multiprocessing.cpu_count()
        if n_workers is None:
            n_workers = max(2, cpu_count // 2)
        n_workers = max(1, min(n_workers, 8))
        logger.info(f"并行换脸：{n_workers} 进程（CPU 核心数: {cpu_count}）")

        # ── 获取视频时长 ─────────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return False
        fps         = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        total_duration = total_frames / fps if fps > 0 else 0
        seg_duration   = total_duration / n_workers
        logger.info(f"视频时长: {total_duration:.1f}s，每段约 {seg_duration:.1f}s")

        # ── 检查 ffmpeg 是否可用 ─────────────────────────────────────────────
        import shutil as _shutil
        if not _shutil.which('ffmpeg'):
            logger.error("未找到 ffmpeg！请先安装：winget install ffmpeg，然后重新打开终端")
            return False

        # ── 临时目录 ─────────────────────────────────────────────────────────
        work_dir = Path(output_path).parent / '_parallel_tmp'
        work_dir.mkdir(parents=True, exist_ok=True)

        # ── FFmpeg 切片（stream copy，无损极快）───────────────────────────────
        input_segs = []
        for i in range(n_workers):
            start    = i * seg_duration
            seg_path = str(work_dir / f'seg_{i:03d}_in.mp4')
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', f'{start:.3f}',
                '-t',  f'{seg_duration:.3f}',
                '-c', 'copy',
                seg_path
            ]
            ret = subprocess.run(cmd, capture_output=True)
            if ret.returncode != 0 or not Path(seg_path).exists():
                logger.error(f"切割片段 {i} 失败: {ret.stderr.decode(errors='replace')}")
                shutil.rmtree(work_dir, ignore_errors=True)
                return False
            input_segs.append(seg_path)
            logger.info(f"切片 {i+1}/{n_workers} 完成 → {seg_path}")

        output_segs = [str(work_dir / f'seg_{i:03d}_out.mp4') for i in range(n_workers)]

        # ── 多进程并行换脸 ───────────────────────────────────────────────────
        worker_args = [
            (source_image_path, input_segs[i], output_segs[i], i + 1, n_workers)
            for i in range(n_workers)
        ]
        logger.info(f"启动 {n_workers} 个子进程开始处理...")

        # Windows 必须在 spawn 上下文中使用 Pool
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_worker_swap_segment, worker_args)

        # ── 检查子进程结果 ───────────────────────────────────────────────────
        failed = [r for r in results if not r[1]]
        if failed:
            for wid, _, msg in failed:
                logger.error(f"Worker {wid} 失败: {msg}")
            shutil.rmtree(work_dir, ignore_errors=True)
            return False
        logger.info(f"所有 {n_workers} 个片段处理完毕，开始合并...")

        # ── FFmpeg concat 合并 ───────────────────────────────────────────────
        concat_txt = str(work_dir / 'concat.txt')
        with open(concat_txt, 'w', encoding='utf-8') as f:
            for seg in output_segs:
                f.write(f"file '{os.path.abspath(seg)}'\n")

        merge_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_txt,
            '-c:v', 'libx264',   # 重新编码确保合并无缝
            '-preset', 'fast',
            '-crf', '18',
            output_path
        ]
        ret = subprocess.run(merge_cmd, capture_output=True, text=True)
        if ret.returncode != 0:
            logger.error(f"合并失败: {ret.stderr}")
            return False

        logger.info(f"[OK] 并行换脸完成 → {output_path}")
        shutil.rmtree(work_dir, ignore_errors=True)
        return True


class DeepfaceLiveSwapper:
    """
    DeepfaceLive 方案（本地部署）
    
    注意：需要先 clone DeepFaceLive 仓库
    git clone https://github.com/iperov/DeepFaceLive.git
    """
    
    def __init__(self, deepfaceslive_dir: str = None):
        """
        初始化 DeepfaceLive
        
        Args:
            deepfaceslive_dir: DeepFaceLive 项目目录
        """
        self.deepfaceslive_dir = deepfaceslive_dir or './DeepFaceLive'
        self._validate_installation()
    
    def _validate_installation(self):
        """验证 DeepfaceLive 是否已安装"""
        dfl_path = Path(self.deepfaceslive_dir)
        if not dfl_path.exists():
            logger.error(f"DeepfaceLive 目录不存在: {self.deepfaceslive_dir}")
            logger.info("请执行: git clone https://github.com/iperov/DeepFaceLive.git")
            raise RuntimeError("DeepfaceLive not installed")
        
        logger.info(f"[OK] DeepfaceLive 已找到: {self.deepfaceslive_dir}")
    
    def swap_faces_in_video_cli(self, source_image_path: str, video_path: str,
                                output_path: str) -> bool:
        """
        使用 DeepfaceLive CLI 进行人脸交换
        （这是一个包装函数，具体实现需要调用 DeepfaceLive 的 Python API）
        
        Args:
            source_image_path: 源人脸图片
            video_path: 输入视频
            output_path: 输出视频
        
        Returns:
            成功返回 True
        """
        try:
            import subprocess
            
            # 使用 DeepfaceLive 的 main_cli.py
            cmd = [
                'python',
                f'{self.deepfaceslive_dir}/main_cli.py',
                '--source-path', source_image_path,
                '--input-path', video_path,
                '--output-path', output_path,
                '--device', 'cpu'  # 或 'cuda' 如果有 GPU
            ]
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info(result.stdout)
            logger.info(f"[OK] 使用 DeepfaceLive 处理完成: {output_path}")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"DeepfaceLive 执行失败: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"处理失败: {e}")
            return False


# 推荐使用的快速接口
def quick_face_swap(source_image_path: str, target_image_path: str,
                    output_path: str = None) -> Optional[np.ndarray]:
    """
    快速人脸交换 (图片)
    
    Args:
        source_image_path: 源人脸图片路径
        target_image_path: 目标图片路径
        output_path: 输出图片路径（可选）
    
    Returns:
        交换后的图片
    """
    swapper = FaceSwapper()
    
    source = cv2.imread(source_image_path)
    target = cv2.imread(target_image_path)
    
    result = swapper.swap_faces(source, target)
    
    if result is not None and output_path:
        cv2.imwrite(output_path, result)
        logger.info(f"[OK] 结果已保存: {output_path}")
    
    return result


def quick_face_swap_video(source_image_path: str, video_path: str,
                          output_path: str) -> bool:
    """
    快速人脸交换 (视频)
    """
    swapper = FaceSwapper()
    return swapper.swap_faces_in_video(source_image_path, video_path, output_path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 2:
        source = sys.argv[1]
        target = sys.argv[2]
        output = sys.argv[3] if len(sys.argv) > 3 else 'output.jpg'
        
        result = quick_face_swap(source, target, output)
        if result is not None:
            print(f"[OK] 人脸交换成功，输出: {output}")
        else:
            print("[FAIL] 人脸交换失败")
