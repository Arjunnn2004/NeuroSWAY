import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class AdvancedSkeletonDetector:
    """
    Advanced skeleton detection with pose analysis, gesture recognition,
    and data logging capabilities
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the advanced skeleton detector"""
        self.config = self.load_config(config_file)
        self.setup_mediapipe()
        self.setup_tracking()
        self.setup_logging()
    
    def load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "detection_confidence": 0.5,
            "tracking_confidence": 0.5,
            "model_complexity": 1,
            "camera_width": 1280,
            "camera_height": 720,
            "camera_fps": 30,
            "save_poses": True,
            "pose_history_length": 100,
            "gesture_detection": True,
            "angle_analysis": True
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def setup_mediapipe(self):
        """Setup MediaPipe components"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.config["detection_confidence"],
            min_tracking_confidence=self.config["tracking_confidence"],
            model_complexity=self.config["model_complexity"]
        )
    
    def setup_tracking(self):
        """Setup tracking variables"""
        self.pose_history = []
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.gestures = []
    
    def setup_logging(self):
        """Setup logging directory"""
        self.log_dir = "pose_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def calculate_angle(self, point1: Tuple[float, float], 
                       point2: Tuple[float, float], 
                       point3: Tuple[float, float]) -> float:
        """Calculate angle between three points"""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def analyze_pose(self, landmarks: List[Tuple[float, float]]) -> Dict:
        """Analyze pose for various metrics"""
        if len(landmarks) < 33:
            return {}
        
        analysis = {}
        
        # Calculate key angles
        if self.config["angle_analysis"]:
            # Left arm angle (shoulder-elbow-wrist)
            if all(landmarks[i] != (0, 0) for i in [11, 13, 15]):
                left_arm_angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
                analysis["left_arm_angle"] = left_arm_angle
            
            # Right arm angle (shoulder-elbow-wrist)
            if all(landmarks[i] != (0, 0) for i in [12, 14, 16]):
                right_arm_angle = self.calculate_angle(landmarks[12], landmarks[14], landmarks[16])
                analysis["right_arm_angle"] = right_arm_angle
            
            # Left leg angle (hip-knee-ankle)
            if all(landmarks[i] != (0, 0) for i in [23, 25, 27]):
                left_leg_angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
                analysis["left_leg_angle"] = left_leg_angle
            
            # Right leg angle (hip-knee-ankle)
            if all(landmarks[i] != (0, 0) for i in [24, 26, 28]):
                right_leg_angle = self.calculate_angle(landmarks[24], landmarks[26], landmarks[28])
                analysis["right_leg_angle"] = right_leg_angle
        
        # Pose classification
        analysis["pose_type"] = self.classify_pose(landmarks)
        
        return analysis
    
    def classify_pose(self, landmarks: List[Tuple[float, float]]) -> str:
        """Simple pose classification"""
        if len(landmarks) < 33:
            return "unknown"
        
        # Check if person is standing (hips above knees)
        if landmarks[23][1] < landmarks[25][1] and landmarks[24][1] < landmarks[26][1]:
            # Check arm positions for gestures
            if landmarks[15][1] < landmarks[11][1] and landmarks[16][1] < landmarks[12][1]:
                return "arms_up"
            elif landmarks[15][1] > landmarks[13][1] and landmarks[16][1] > landmarks[14][1]:
                return "arms_down"
            else:
                return "standing"
        else:
            return "sitting_or_lying"
    
    def detect_gestures(self, landmarks: List[Tuple[float, float]]) -> List[str]:
        """Detect simple gestures"""
        gestures = []
        
        if len(landmarks) < 33:
            return gestures
        
        # Wave detection (hand above head)
        if landmarks[15][1] < landmarks[0][1] or landmarks[16][1] < landmarks[0][1]:
            gestures.append("wave")
        
        # T-pose detection (arms horizontal)
        if (abs(landmarks[11][1] - landmarks[15][1]) < 50 and 
            abs(landmarks[12][1] - landmarks[16][1]) < 50):
            gestures.append("t_pose")
        
        return gestures
    
    def save_pose_data(self, landmarks: List[Tuple[float, float]], analysis: Dict):
        """Save pose data to file"""
        pose_data = {
            "timestamp": datetime.now().isoformat(),
            "frame": self.frame_count,
            "landmarks": landmarks,
            "analysis": analysis
        }
        
        filename = os.path.join(self.log_dir, f"pose_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Append to existing file or create new one
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(pose_data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def draw_advanced_info(self, image: np.ndarray, results: object, analysis: Dict) -> np.ndarray:
        """Draw advanced information overlay"""
        height, width = image.shape[:2]
        
        # FPS and basic info
        cv2.putText(image, f'FPS: {self.current_fps:.1f}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status = "Pose Detected" if results.pose_landmarks else "No Pose"
        status_color = (0, 255, 0) if results.pose_landmarks else (0, 0, 255)
        cv2.putText(image, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Pose analysis
        y_offset = 90
        if analysis:
            for key, value in analysis.items():
                if isinstance(value, float):
                    text = f'{key}: {value:.1f}°'
                else:
                    text = f'{key}: {value}'
                cv2.putText(image, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 25
        
        # Gestures
        if hasattr(self, 'current_gestures') and self.current_gestures:
            cv2.putText(image, f'Gestures: {", ".join(self.current_gestures)}', 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Controls
        controls = [
            "Advanced Skeleton Detection",
            "Q - Quit | S - Save | R - Reset",
            "P - Save Pose Data | C - Clear History"
        ]
        
        for i, control in enumerate(controls):
            y_pos = height - 80 + (i * 25)
            cv2.putText(image, control, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def process_advanced_realtime(self, camera_index: int = 0):
        """Advanced real-time processing with pose analysis"""
        print("Starting Advanced Skeleton Detection")
        print("Features: Pose Analysis, Gesture Detection, Data Logging")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera_height"])
        cap.set(cv2.CAP_PROP_FPS, self.config["camera_fps"])
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect pose
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = self.pose.process(rgb_frame)
                rgb_frame.flags.writeable = True
                processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        processed_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Get landmarks and analyze
                    landmarks = []
                    if results.pose_landmarks:
                        height, width = processed_frame.shape[:2]
                        for landmark in results.pose_landmarks.landmark:
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            landmarks.append((x, y))
                    
                    # Analyze pose
                    analysis = self.analyze_pose(landmarks)
                    
                    # Detect gestures
                    if self.config["gesture_detection"]:
                        self.current_gestures = self.detect_gestures(landmarks)
                    
                    # Store pose history
                    if self.config["save_poses"]:
                        self.pose_history.append(landmarks)
                        if len(self.pose_history) > self.config["pose_history_length"]:
                            self.pose_history.pop(0)
                
                else:
                    analysis = {}
                    landmarks = []
                    self.current_gestures = []
                
                # Draw advanced info
                processed_frame = self.draw_advanced_info(processed_frame, results, analysis)
                
                # Update FPS
                self.fps_counter += 1
                if self.fps_counter % 30 == 0:
                    end_time = time.time()
                    self.current_fps = 30 / (end_time - self.fps_start_time)
                    self.fps_start_time = end_time
                
                # Display frame
                cv2.imshow('Advanced Skeleton Detection', processed_frame)
                self.frame_count += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f'advanced_frame_{self.frame_count}_{int(time.time())}.jpg'
                    cv2.imwrite(filename, processed_frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('p') and landmarks:
                    self.save_pose_data(landmarks, analysis)
                    print("Pose data saved")
                elif key == ord('c'):
                    self.pose_history.clear()
                    print("Pose history cleared")
                elif key == ord('r'):
                    self.setup_mediapipe()
                    print("Detector reset")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"Processed {self.frame_count} frames")
            print(f"Pose history length: {len(self.pose_history)}")


def main():
    """Main function for advanced skeleton detection"""
    print("Advanced Skeleton Detection System")
    print("Features:")
    print("- Real-time pose analysis")
    print("- Gesture detection")
    print("- Angle calculations")
    print("- Data logging")
    
    detector = AdvancedSkeletonDetector()
    
    camera_index = input("Enter camera index (default: 0): ").strip()
    camera_index = int(camera_index) if camera_index.isdigit() else 0
    
    detector.process_advanced_realtime(camera_index)


if __name__ == "__main__":
    main()