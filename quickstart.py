#!/usr/bin/env python3
"""
快速开始示例脚本
演示如何使用各个模块
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.face_detector import quick_detect
from src.face_swapper import quick_face_swap, quick_face_swap_video
from src.audio_processor import quick_tts, quick_extract_audio, quick_add_audio_to_video
from src.video_composer import quick_compose_video, quick_get_video_info
from src.pipeline import VideoDeepfakePipeline


def demo_face_detection():
    """演示人脸检测"""
    print("\n" + "="*60)
    print("演示: 人脸检测")
    print("="*60)
    
    image_path = 'test_image.jpg'
    print(f"检测图片: {image_path}")
    
    try:
        faces = quick_detect(image_path, model_type='mtcnn')
        print(f"✓ 检测到 {len(faces)} 张人脸")
        for i, face in enumerate(faces):
            print(f"  人脸 {i}: 位置 {face['bbox']}, 置信度 {face['confidence']:.2f}")
    except Exception as e:
        print(f"✗ 检测失败: {e}")


def demo_face_swap():
    """演示人脸交换"""
    print("\n" + "="*60)
    print("演示: 人脸交换")
    print("="*60)
    
    source_image = 'source_face.jpg'
    target_image = 'target_image.jpg'
    output_image = 'swapped_face.jpg'
    
    print(f"源图片: {source_image}")
    print(f"目标图片: {target_image}")
    print(f"输出: {output_image}")
    
    try:
        result = quick_face_swap(source_image, target_image, output_image)
        if result is not None:
            print(f"✓ 人脸交换成功: {output_image}")
        else:
            print("✗ 人脸交换失败")
    except Exception as e:
        print(f"✗ 交换失败: {e}")


def demo_face_swap_video():
    """演示视频人脸交换"""
    print("\n" + "="*60)
    print("演示: 视频人脸交换")
    print("="*60)
    
    source_image = 'source_face.jpg'
    input_video = 'input_video.mp4'
    output_video = 'swapped_video.mp4'
    
    print(f"源人脸: {source_image}")
    print(f"输入视频: {input_video}")
    print(f"输出视频: {output_video}")
    print("处理中... (这可能需要几分钟)")
    
    try:
        success = quick_face_swap_video(source_image, input_video, output_video)
        if success:
            print(f"✓ 视频人脸交换成功: {output_video}")
        else:
            print("✗ 视频交换失败")
    except Exception as e:
        print(f"✗ 失败: {e}")


def demo_tts():
    """演示 TTS 文字转语音"""
    print("\n" + "="*60)
    print("演示: TTS 文字转语音")
    print("="*60)
    
    text = "你好，这是合成的语音。这是一个演示。"
    output_audio = 'generated_audio.mp3'
    
    print(f"文本: {text}")
    print(f"输出: {output_audio}")
    print("生成中... (需要网络连接)")
    
    try:
        success = quick_tts(text, output_audio, voice='zh-CN-XiaoxiaoNeural')
        if success:
            print(f"✓ TTS 成功: {output_audio}")
        else:
            print("✗ TTS 失败")
    except Exception as e:
        print(f"✗ 失败: {e}")


def demo_extract_audio():
    """演示从视频提取音频"""
    print("\n" + "="*60)
    print("演示: 从视频提取音频")
    print("="*60)
    
    input_video = 'input_video.mp4'
    output_audio = 'extracted_audio.mp3'
    
    print(f"输入视频: {input_video}")
    print(f"输出音频: {output_audio}")
    
    try:
        success = quick_extract_audio(input_video, output_audio)
        if success:
            print(f"✓ 音频提取成功: {output_audio}")
        else:
            print("✗ 提取失败")
    except Exception as e:
        print(f"✗ 失败: {e}")


def demo_compose_video():
    """演示视频合成"""
    print("\n" + "="*60)
    print("演示: 视频和音频合成")
    print("="*60)
    
    video_path = 'video.mp4'
    audio_path = 'audio.mp3'
    output_path = 'final.mp4'
    
    print(f"视频: {video_path}")
    print(f"音频: {audio_path}")
    print(f"输出: {output_path}")
    
    try:
        success = quick_compose_video(video_path, audio_path, output_path)
        if success:
            print(f"✓ 视频合成成功: {output_path}")
        else:
            print("✗ 合成失败")
    except Exception as e:
        print(f"✗ 失败: {e}")


def demo_full_pipeline():
    """演示完整工作流"""
    print("\n" + "="*60)
    print("演示: 完整工作流程")
    print("="*60)
    
    source_image = 'source_face.jpg'
    input_video = 'input_video.mp4'
    output_text = '你好，这是合成的视频。欢迎观看！'
    output_video = 'final_output.mp4'
    
    print(f"源人脸: {source_image}")
    print(f"输入视频: {input_video}")
    print(f"输出文字: {output_text}")
    print(f"输出视频: {output_video}")
    print("\n处理中... (这可能需要 5-10 分钟)")
    
    try:
        pipeline = VideoDeepfakePipeline()
        success = pipeline.process_full_pipeline(
            source_image=source_image,
            input_video=input_video,
            output_text=output_text,
            output_path=output_video
        )
        
        if success:
            print(f"\n✓ 完整工作流成功！")
            print(f"输出文件: {output_video}")
        else:
            print("✗ 工作流失败")
    except Exception as e:
        print(f"✗ 失败: {e}")


def list_demos():
    """列出所有演示"""
    demos = {
        '1': ('人脸检测', demo_face_detection),
        '2': ('人脸交换（图片）', demo_face_swap),
        '3': ('人脸交换（视频）', demo_face_swap_video),
        '4': ('TTS 文字转语音', demo_tts),
        '5': ('从视频提取音频', demo_extract_audio),
        '6': ('视频和音频合成', demo_compose_video),
        '7': ('完整工作流程', demo_full_pipeline),
    }
    
    print("\n" + "="*60)
    print("可用的演示:")
    print("="*60)
    for key, (name, _) in demos.items():
        print(f"{key}. {name}")
    print("0. 退出")
    
    return demos


def main():
    parser = argparse.ArgumentParser(
        description='视频深度伪造处理器 - 快速开始演示',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python quickstart.py                 # 交互式选择
  python quickstart.py --demo 1        # 运行演示 1
  python quickstart.py --pipeline      # 运行完整工作流
        """
    )
    
    parser.add_argument('--demo', type=int, help='演示编号 (1-7)')
    parser.add_argument('--pipeline', action='store_true', help='运行完整工作流程')
    
    args = parser.parse_args()
    
    if args.pipeline:
        demo_full_pipeline()
    elif args.demo:
        demos = {
            1: demo_face_detection,
            2: demo_face_swap,
            3: demo_face_swap_video,
            4: demo_tts,
            5: demo_extract_audio,
            6: demo_compose_video,
            7: demo_full_pipeline,
        }
        
        if args.demo in demos:
            demos[args.demo]()
        else:
            print(f"✗ 无效的演示编号: {args.demo}")
            sys.exit(1)
    else:
        # 交互式模式
        while True:
            demos = list_demos()
            choice = input("\n请选择 (0-7): ").strip()
            
            if choice == '0':
                print("再见！")
                break
            elif choice in demos:
                _, demo_func = demos[choice]
                demo_func()
            else:
                print("✗ 无效选择")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        sys.exit(1)
