#!/usr/bin/env python3
"""
Real-time Object Detection Example using SSD

This example demonstrates real-time object detection on video streams,
including webcam and video files.
"""

import os
import sys
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import deque

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.ssd_detector import SSDDetector
from utils.visualization import draw_detections


class RealTimeDetector:
    """Real-time object detection with performance monitoring"""
    
    def __init__(self, config_path: str = 'configs/ssd300_config.json', 
                 device: str = 'auto'):
        """
        Initialize real-time detector
        
        Args:
            config_path: Path to model configuration
            device: Device to run inference on
        """
        self.detector = SSDDetector(config_path=config_path, device=device)
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)  # Last 30 frames
        self.detection_history = deque(maxlen=30)
        
        # Display settings
        self.show_fps = True
        self.show_detections = True
        self.confidence_threshold = 0.5
        
        print(f"Real-time detector initialized on {self.detector.device}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with detections
        """
        start_time = time.time()
        
        # Detect objects
        detections = self.detector._detect_frame(frame, self.confidence_threshold)
        
        # Calculate FPS
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time
        
        # Update history
        self.fps_history.append(fps)
        self.detection_history.append(len(detections))
        
        # Draw detections
        if self.show_detections:
            frame = draw_detections(frame, detections, confidence_threshold=self.confidence_threshold)
        
        # Draw performance info
        if self.show_fps:
            frame = self._draw_performance_info(frame)
        
        return frame
    
    def _draw_performance_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw performance information on frame"""
        # Calculate average FPS
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Calculate average detections
        avg_detections = np.mean(self.detection_history) if self.detection_history else 0
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0)
        
        # FPS info
        fps_text = f"FPS: {avg_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), font, font_scale, color, thickness)
        
        # Detection count
        detections_text = f"Detections: {avg_detections:.1f}"
        cv2.putText(frame, detections_text, (10, 60), font, font_scale, color, thickness)
        
        # Confidence threshold
        conf_text = f"Confidence: {self.confidence_threshold}"
        cv2.putText(frame, conf_text, (10, 90), font, font_scale, color, thickness)
        
        return frame
    
    def run_webcam(self, camera_id: int = 0, save_output: bool = False):
        """
        Run real-time detection on webcam
        
        Args:
            camera_id: Camera device ID
            save_output: Whether to save output video
        """
        print(f"Starting webcam detection (Camera ID: {camera_id})")
        print("Press 'q' to quit, 's' to save screenshot, '+'/'-' to adjust confidence")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera resolution: {width}x{height}, FPS: {fps}")
        
        # Initialize video writer if saving
        writer = None
        if save_output:
            output_path = f"outputs/webcam_detection_{int(time.time())}.mp4"
            os.makedirs("outputs", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Real-time Object Detection', processed_frame)
                
                # Save frame if requested
                if writer:
                    writer.write(processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f"outputs/screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"Screenshot saved: {screenshot_path}")
                elif key == ord('+') or key == ord('='):
                    # Increase confidence threshold
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                elif key == ord('-'):
                    # Decrease confidence threshold
                    self.confidence_threshold = max(0.05, self.confidence_threshold - 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                
                frame_count += 1
                
                # Print stats every 100 frames
                if frame_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    avg_fps = frame_count / elapsed_time
                    print(f"Processed {frame_count} frames, Avg FPS: {avg_fps:.2f}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            total_time = time.time() - start_time
            print(f"\nSession Statistics:")
            print(f"  Total frames processed: {frame_count}")
            print(f"  Total time: {total_time:.2f} seconds")
            print(f"  Average FPS: {frame_count / total_time:.2f}")
            print(f"  Average detections per frame: {np.mean(self.detection_history):.2f}")
    
    def run_video(self, video_path: str, save_output: bool = True):
        """
        Run detection on video file
        
        Args:
            video_path: Path to input video
            save_output: Whether to save output video
        """
        print(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        print(f"Video properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Total frames: {total_frames}")
        
        # Initialize video writer if saving
        writer = None
        if save_output:
            output_path = f"outputs/video_detection_{int(time.time())}.mp4"
            os.makedirs("outputs", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Video Object Detection', processed_frame)
                
                # Save frame if requested
                if writer:
                    writer.write(processed_frame)
                
                # Handle quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    progress = frame_count / total_frames * 100
                    elapsed_time = time.time() - start_time
                    avg_fps = frame_count / elapsed_time
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), "
                          f"Avg FPS: {avg_fps:.2f}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            total_time = time.time() - start_time
            print(f"\nVideo Processing Complete:")
            print(f"  Frames processed: {frame_count}")
            print(f"  Processing time: {total_time:.2f} seconds")
            print(f"  Average processing FPS: {frame_count / total_time:.2f}")
            print(f"  Average detections per frame: {np.mean(self.detection_history):.2f}")
    
    def run_image_sequence(self, image_dir: str, output_dir: str = "outputs"):
        """
        Run detection on a sequence of images
        
        Args:
            image_dir: Directory containing images
            output_dir: Output directory for processed images
        """
        print(f"Processing image sequence from: {image_dir}")
        
        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in {image_dir}")
            return
        
        image_files = sorted(image_files)
        print(f"Found {len(image_files)} images")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            # Load image
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Could not load image: {image_path}")
                continue
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Save processed image
            output_path = os.path.join(output_dir, f"processed_{image_path.name}")
            cv2.imwrite(output_path, processed_frame)
        
        total_time = time.time() - start_time
        print(f"\nImage sequence processing complete:")
        print(f"  Images processed: {len(image_files)}")
        print(f"  Processing time: {total_time:.2f} seconds")
        print(f"  Average time per image: {total_time / len(image_files):.3f} seconds")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Real-time Object Detection with SSD')
    parser.add_argument('--mode', choices=['webcam', 'video', 'images'], 
                       default='webcam', help='Detection mode')
    parser.add_argument('--source', type=str, default='0', 
                       help='Source (camera ID, video path, or image directory)')
    parser.add_argument('--config', type=str, default='configs/ssd300_config.json',
                       help='Model configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to run on (cpu, cuda, auto)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--save', action='store_true',
                       help='Save output video/images')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RealTimeDetector(config_path=args.config, device=args.device)
    detector.confidence_threshold = args.confidence
    
    print("Real-time Object Detection")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Source: {args.source}")
    print(f"Device: {detector.detector.device}")
    print(f"Confidence threshold: {detector.confidence_threshold}")
    
    try:
        if args.mode == 'webcam':
            camera_id = int(args.source) if args.source.isdigit() else 0
            detector.run_webcam(camera_id=camera_id, save_output=args.save)
        
        elif args.mode == 'video':
            detector.run_video(video_path=args.source, save_output=args.save)
        
        elif args.mode == 'images':
            detector.run_image_sequence(image_dir=args.source, output_dir="outputs")
    
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
