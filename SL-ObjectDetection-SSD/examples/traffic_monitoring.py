#!/usr/bin/env python3
"""
Traffic Monitoring System using SSD Object Detection

This is a comprehensive case study demonstrating how to use SSD for
real-time traffic monitoring, including vehicle detection, counting,
and speed estimation.
"""

import os
import sys
import cv2
import numpy as np
import time
import json
from pathlib import Path
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.ssd_detector import SSDDetector
from utils.visualization import draw_detections


class TrafficMonitor:
    """
    Traffic monitoring system using SSD object detection
    """
    
    def __init__(self, config_path: str = 'configs/ssd300_config.json',
                 device: str = 'auto'):
        """
        Initialize traffic monitoring system
        
        Args:
            config_path: Path to model configuration
            device: Device to run inference on
        """
        self.detector = SSDDetector(config_path=config_path, device=device)
        
        # Traffic analysis settings
        self.vehicle_classes = [3, 6, 8]  # car, bus, truck in COCO
        self.person_class = 1  # person in COCO
        
        # Lane detection settings
        self.lanes = []
        self.lane_counters = defaultdict(int)
        
        # Vehicle tracking
        self.tracked_vehicles = {}
        self.next_track_id = 0
        self.tracking_history = defaultdict(list)
        
        # Speed estimation
        self.speed_calibration = 1.0  # pixels per meter
        self.frame_rate = 30.0  # assumed frame rate
        
        # Statistics
        self.total_vehicles = 0
        self.vehicle_speeds = []
        self.detection_history = deque(maxlen=100)
        
        # Output settings
        self.save_analytics = True
        self.analytics_file = "outputs/traffic_analytics.json"
        
        print("Traffic monitoring system initialized")
    
    def setup_lanes(self, frame_shape: tuple, num_lanes: int = 2):
        """
        Setup traffic lanes for counting
        
        Args:
            frame_shape: Shape of input frame (height, width)
            num_lanes: Number of traffic lanes
        """
        height, width = frame_shape[:2]
        
        # Create horizontal lanes
        lane_height = height // (num_lanes + 1)
        self.lanes = []
        
        for i in range(num_lanes):
            y = (i + 1) * lane_height
            self.lanes.append({
                'y': y,
                'count': 0,
                'vehicles': set(),
                'speed_limit': 50.0  # km/h
            })
        
        print(f"Setup {num_lanes} traffic lanes")
    
    def detect_vehicles(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> list:
        """
        Detect vehicles in frame
        
        Args:
            frame: Input frame
            confidence_threshold: Detection confidence threshold
            
        Returns:
            List of vehicle detections
        """
        # Get all detections
        detections = self.detector._detect_frame(frame, confidence_threshold)
        
        # Filter for vehicles only
        vehicle_detections = []
        for detection in detections:
            if detection['label'] in self.vehicle_classes:
                vehicle_detections.append(detection)
        
        return vehicle_detections
    
    def track_vehicles(self, detections: list, frame_number: int):
        """
        Track vehicles across frames
        
        Args:
            detections: Current frame detections
            frame_number: Current frame number
        """
        current_vehicles = {}
        
        for detection in detections:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Find closest tracked vehicle
            min_distance = float('inf')
            best_match = None
            
            for track_id, vehicle in self.tracked_vehicles.items():
                if vehicle['active']:
                    last_pos = vehicle['positions'][-1]
                    distance = np.sqrt((center_x - last_pos[0])**2 + (center_y - last_pos[1])**2)
                    
                    if distance < min_distance and distance < 100:  # max tracking distance
                        min_distance = distance
                        best_match = track_id
            
            if best_match is not None:
                # Update existing track
                self.tracked_vehicles[best_match]['positions'].append((center_x, center_y))
                self.tracked_vehicles[best_match]['last_seen'] = frame_number
                current_vehicles[best_match] = detection
            else:
                # Create new track
                self.tracked_vehicles[self.next_track_id] = {
                    'positions': [(center_x, center_y)],
                    'first_seen': frame_number,
                    'last_seen': frame_number,
                    'active': True,
                    'detection': detection
                }
                current_vehicles[self.next_track_id] = detection
                self.next_track_id += 1
        
        # Mark inactive tracks
        for track_id, vehicle in self.tracked_vehicles.items():
            if vehicle['active'] and track_id not in current_vehicles:
                if frame_number - vehicle['last_seen'] > 10:  # 10 frames timeout
                    vehicle['active'] = False
                    self._analyze_vehicle_track(track_id, vehicle)
    
    def _analyze_vehicle_track(self, track_id: int, vehicle: dict):
        """
        Analyze completed vehicle track
        
        Args:
            track_id: Vehicle track ID
            vehicle: Vehicle tracking data
        """
        if len(vehicle['positions']) < 2:
            return
        
        positions = vehicle['positions']
        
        # Calculate speed
        if len(positions) >= 2:
            # Calculate total distance traveled
            total_distance = 0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                total_distance += np.sqrt(dx**2 + dy**2)
            
            # Calculate time taken
            frames_taken = vehicle['last_seen'] - vehicle['first_seen']
            time_taken = frames_taken / self.frame_rate
            
            if time_taken > 0:
                # Convert to real-world speed (km/h)
                speed_pixels_per_sec = total_distance / time_taken
                speed_m_per_sec = speed_pixels_per_sec / self.speed_calibration
                speed_km_per_h = speed_m_per_sec * 3.6
                
                self.vehicle_speeds.append(speed_km_per_h)
                
                # Check for speed violations
                for lane in self.lanes:
                    if self._vehicle_in_lane(positions[-1], lane):
                        if speed_km_per_h > lane['speed_limit']:
                            print(f"Speed violation detected: {speed_km_per_h:.1f} km/h")
        
        # Count vehicles crossing lanes
        self._count_lane_crossings(track_id, vehicle)
    
    def _vehicle_in_lane(self, position: tuple, lane: dict) -> bool:
        """Check if vehicle is in a specific lane"""
        y = position[1]
        lane_y = lane['y']
        return abs(y - lane_y) < 50  # 50 pixel tolerance
    
    def _count_lane_crossings(self, track_id: int, vehicle: dict):
        """Count vehicles crossing each lane"""
        positions = vehicle['positions']
        
        for lane in self.lanes:
            lane_y = lane['y']
            crossed = False
            
            for i in range(1, len(positions)):
                y1 = positions[i-1][1]
                y2 = positions[i][1]
                
                # Check if vehicle crossed the lane
                if (y1 < lane_y and y2 > lane_y) or (y1 > lane_y and y2 < lane_y):
                    if track_id not in lane['vehicles']:
                        lane['vehicles'].add(track_id)
                        lane['count'] += 1
                        self.total_vehicles += 1
                        crossed = True
                        break
            
            if crossed:
                print(f"Vehicle {track_id} crossed lane at y={lane_y}, total: {lane['count']}")
    
    def estimate_traffic_density(self, detections: list, frame_shape: tuple) -> float:
        """
        Estimate traffic density
        
        Args:
            detections: Current detections
            frame_shape: Frame shape
            
        Returns:
            Traffic density (vehicles per pixel)
        """
        if not detections:
            return 0.0
        
        # Calculate total area covered by vehicles
        total_vehicle_area = 0
        for detection in detections:
            bbox = detection['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            total_vehicle_area += area
        
        # Calculate frame area
        frame_area = frame_shape[0] * frame_shape[1]
        
        # Return density as percentage
        density = (total_vehicle_area / frame_area) * 100
        return density
    
    def analyze_traffic_patterns(self) -> dict:
        """
        Analyze traffic patterns
        
        Returns:
            Dictionary containing traffic analysis
        """
        analysis = {
            'total_vehicles': self.total_vehicles,
            'lane_counts': {i: lane['count'] for i, lane in enumerate(self.lanes)},
            'average_speed': np.mean(self.vehicle_speeds) if self.vehicle_speeds else 0,
            'speed_violations': len([s for s in self.vehicle_speeds if s > 50]),
            'traffic_density': np.mean(self.detection_history) if self.detection_history else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    def process_frame(self, frame: np.ndarray, frame_number: int = 0) -> np.ndarray:
        """
        Process a single frame for traffic monitoring
        
        Args:
            frame: Input frame
            frame_number: Frame number for tracking
            
        Returns:
            Processed frame with traffic information
        """
        # Detect vehicles
        vehicle_detections = self.detect_vehicles(frame, confidence_threshold=0.5)
        
        # Track vehicles
        self.track_vehicles(vehicle_detections, frame_number)
        
        # Calculate traffic density
        density = self.estimate_traffic_density(vehicle_detections, frame.shape)
        self.detection_history.append(len(vehicle_detections))
        
        # Draw detections
        annotated_frame = draw_detections(frame, vehicle_detections)
        
        # Draw traffic information
        annotated_frame = self._draw_traffic_info(annotated_frame, vehicle_detections, density)
        
        # Draw lanes
        annotated_frame = self._draw_lanes(annotated_frame)
        
        return annotated_frame
    
    def _draw_traffic_info(self, frame: np.ndarray, detections: list, density: float) -> np.ndarray:
        """Draw traffic information on frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (0, 255, 0)
        
        # Vehicle count
        cv2.putText(frame, f"Vehicles: {len(detections)}", (10, 30), 
                   font, font_scale, color, thickness)
        
        # Traffic density
        cv2.putText(frame, f"Density: {density:.1f}%", (10, 60), 
                   font, font_scale, color, thickness)
        
        # Total vehicles counted
        cv2.putText(frame, f"Total: {self.total_vehicles}", (10, 90), 
                   font, font_scale, color, thickness)
        
        # Average speed
        if self.vehicle_speeds:
            avg_speed = np.mean(self.vehicle_speeds)
            cv2.putText(frame, f"Avg Speed: {avg_speed:.1f} km/h", (10, 120), 
                       font, font_scale, color, thickness)
        
        return frame
    
    def _draw_lanes(self, frame: np.ndarray) -> np.ndarray:
        """Draw traffic lanes on frame"""
        height, width = frame.shape[:2]
        
        for i, lane in enumerate(self.lanes):
            y = lane['y']
            color = (255, 0, 0)  # Blue for lanes
            
            # Draw lane line
            cv2.line(frame, (0, y), (width, y), color, 2)
            
            # Draw lane count
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            cv2.putText(frame, f"Lane {i+1}: {lane['count']}", (10, y-10), 
                       font, font_scale, color, thickness)
        
        return frame
    
    def run_monitoring(self, source: str = 0, duration: int = None):
        """
        Run traffic monitoring
        
        Args:
            source: Video source (camera ID or video file)
            duration: Monitoring duration in seconds (None for unlimited)
        """
        print(f"Starting traffic monitoring from source: {source}")
        print("Press 'q' to quit, 's' to save analytics")
        
        # Setup video capture
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open source {source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Video properties: {width}x{height}, {fps} FPS")
        
        # Setup lanes
        self.setup_lanes((height, width), num_lanes=2)
        
        # Setup output
        os.makedirs("outputs", exist_ok=True)
        output_path = f"outputs/traffic_monitoring_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame, frame_count)
                
                # Display frame
                cv2.imshow('Traffic Monitoring', processed_frame)
                
                # Save frame
                writer.write(processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_analytics()
                
                frame_count += 1
                
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    avg_fps = frame_count / elapsed_time
                    print(f"Processed {frame_count} frames, Avg FPS: {avg_fps:.2f}")
        
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            
            # Save final analytics
            self._save_analytics()
            
            # Print final statistics
            self._print_final_statistics()
    
    def _save_analytics(self):
        """Save traffic analytics to file"""
        if self.save_analytics:
            analytics = self.analyze_traffic_patterns()
            
            with open(self.analytics_file, 'w') as f:
                json.dump(analytics, f, indent=2)
            
            print(f"Analytics saved to: {self.analytics_file}")
    
    def _print_final_statistics(self):
        """Print final monitoring statistics"""
        print("\n" + "=" * 50)
        print("TRAFFIC MONITORING STATISTICS")
        print("=" * 50)
        
        analytics = self.analyze_traffic_patterns()
        
        print(f"Total Vehicles Counted: {analytics['total_vehicles']}")
        print(f"Lane Counts:")
        for lane_id, count in analytics['lane_counts'].items():
            print(f"  Lane {lane_id + 1}: {count} vehicles")
        
        print(f"Average Speed: {analytics['average_speed']:.1f} km/h")
        print(f"Speed Violations: {analytics['speed_violations']}")
        print(f"Average Traffic Density: {analytics['traffic_density']:.1f}%")
        
        # Create summary plot
        self._create_summary_plot()
    
    def _create_summary_plot(self):
        """Create summary plot of traffic data"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Lane counts
        lane_ids = list(self.analyze_traffic_patterns()['lane_counts'].keys())
        lane_counts = list(self.analyze_traffic_patterns()['lane_counts'].values())
        
        axes[0, 0].bar(lane_ids, lane_counts)
        axes[0, 0].set_title('Vehicle Count by Lane')
        axes[0, 0].set_xlabel('Lane')
        axes[0, 0].set_ylabel('Count')
        
        # Speed distribution
        if self.vehicle_speeds:
            axes[0, 1].hist(self.vehicle_speeds, bins=20, alpha=0.7)
            axes[0, 1].set_title('Speed Distribution')
            axes[0, 1].set_xlabel('Speed (km/h)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Traffic density over time
        if self.detection_history:
            axes[1, 0].plot(list(self.detection_history))
            axes[1, 0].set_title('Traffic Density Over Time')
            axes[1, 0].set_xlabel('Frame')
            axes[1, 0].set_ylabel('Vehicle Count')
        
        # Vehicle tracking
        active_tracks = [v for v in self.tracked_vehicles.values() if v['active']]
        axes[1, 1].text(0.1, 0.5, f'Active Tracks: {len(active_tracks)}\n'
                                 f'Total Tracks: {len(self.tracked_vehicles)}',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Tracking Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('outputs/traffic_summary.png')
        plt.close()
        
        print("Summary plot saved to: outputs/traffic_summary.png")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Traffic Monitoring with SSD')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (camera ID or video file)')
    parser.add_argument('--config', type=str, default='configs/ssd300_config.json',
                       help='Model configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to run on (cpu, cuda, auto)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Monitoring duration in seconds')
    
    args = parser.parse_args()
    
    # Initialize traffic monitor
    monitor = TrafficMonitor(config_path=args.config, device=args.device)
    
    print("Traffic Monitoring System")
    print("=" * 50)
    print(f"Source: {args.source}")
    print(f"Device: {monitor.detector.device}")
    print(f"Duration: {args.duration if args.duration else 'Unlimited'}")
    
    try:
        # Run monitoring
        monitor.run_monitoring(source=args.source, duration=args.duration)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        monitor._save_analytics()
        monitor._print_final_statistics()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
