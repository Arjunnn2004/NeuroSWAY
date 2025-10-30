import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Optional, Tuple, List
import os


class SkeletonDetector:
    """
    A comprehensive skeleton detection class using MediaPipe Pose
    Supports both real-time camera feed and video file processing
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1):
        """
        Initialize the skeleton detector
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            model_complexity: Model complexity (0, 1, or 2)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            static_image_mode=False,
            smooth_landmarks=True,
            enable_segmentation=False
        )
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def detect_pose(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[object]]:
        """
        Detect pose landmarks in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (processed_image, pose_landmarks)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        # Perform pose detection
        results = self.pose.process(rgb_image)
        
        # Convert back to BGR
        rgb_image.flags.writeable = True
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        return bgr_image, results
    
    def draw_landmarks(self, image: np.ndarray, results: object) -> np.ndarray:
        """
        Draw pose landmarks and connections on the image
        
        Args:
            image: Input image
            results: MediaPipe pose results
            
        Returns:
            Image with drawn landmarks
        """
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return image
    
    def get_landmark_coordinates(self, results: object, image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        """
        Extract landmark coordinates from results
        
        Args:
            results: MediaPipe pose results
            image_shape: Shape of the image (height, width)
            
        Returns:
            List of (x, y) coordinates for each landmark
        """
        landmarks = []
        if results.pose_landmarks:
            height, width = image_shape[:2]
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmarks.append((x, y))
        return landmarks
    
    def draw_custom_skeleton(self, image: np.ndarray, landmarks: List[Tuple[float, float]]) -> np.ndarray:
        """
        Draw a custom skeleton with enhanced visualization
        
        Args:
            image: Input image
            landmarks: List of landmark coordinates
            
        Returns:
            Image with custom skeleton drawn
        """
        if len(landmarks) < 33:  # MediaPipe has 33 pose landmarks
            return image
        
        # Define colors for different body parts
        colors = {
            'head': (0, 255, 255),      # Yellow
            'torso': (0, 255, 0),       # Green
            'arms': (255, 0, 0),        # Blue
            'legs': (0, 0, 255)         # Red
        }
        
        # Define connections for different body parts
        connections = {
            'head': [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8)],
            'torso': [(11, 12), (11, 23), (12, 24), (23, 24)],
            'arms': [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)],
            'legs': [(23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)]
        }
        
        # Draw connections
        for body_part, part_connections in connections.items():
            color = colors[body_part]
            for start_idx, end_idx in part_connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    if start_point != (0, 0) and end_point != (0, 0):
                        cv2.line(image, start_point, end_point, color, 2)
        
        # Draw landmark points
        for i, (x, y) in enumerate(landmarks):
            if x != 0 and y != 0:
                cv2.circle(image, (x, y), 3, (255, 255, 255), -1)
                cv2.circle(image, (x, y), 2, (0, 0, 0), -1)
        
        return image
    
    def update_fps(self) -> None:
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            end_time = time.time()
            self.current_fps = 30 / (end_time - self.fps_start_time)
            self.fps_start_time = end_time
    
    def draw_info(self, image: np.ndarray, results: object) -> np.ndarray:
        """
        Draw information overlay on the image
        
        Args:
            image: Input image
            results: MediaPipe pose results
            
        Returns:
            Image with information overlay
        """
        height, width = image.shape[:2]
        
        # Draw FPS
        cv2.putText(image, f'FPS: {self.current_fps:.1f}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw pose detection status
        status = "Pose Detected" if results.pose_landmarks else "No Pose"
        status_color = (0, 255, 0) if results.pose_landmarks else (0, 0, 255)
        cv2.putText(image, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Draw landmark count
        if results.pose_landmarks:
            landmark_count = len(results.pose_landmarks.landmark)
            cv2.putText(image, f'Landmarks: {landmark_count}', 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Draw controls
        controls = [
            "Controls:",
            "Q - Quit",
            "S - Save frame",
            "R - Reset",
            "SPACE - Pause/Resume"
        ]
        
        for i, control in enumerate(controls):
            y_pos = height - 120 + (i * 25)
            cv2.putText(image, control, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return image
    
    def process_realtime(self, camera_index: int = 0) -> None:
        """
        Process real-time camera feed for skeleton detection
        
        Args:
            camera_index: Camera index (default: 0 for default camera)
        """
        print(f"Starting real-time skeleton detection with camera {camera_index}")
        print("Press 'q' to quit, 's' to save frame, 'r' to reset, SPACE to pause/resume")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        paused = False
        frame_count = 0
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Could not read frame from camera")
                        break
                    
                    # Process frame
                    processed_frame, results = self.detect_pose(frame)
                    
                    # Draw skeleton
                    processed_frame = self.draw_landmarks(processed_frame, results)
                    
                    # Get landmarks for custom drawing
                    landmarks = self.get_landmark_coordinates(results, processed_frame.shape)
                    if landmarks:
                        processed_frame = self.draw_custom_skeleton(processed_frame, landmarks)
                    
                    # Draw information overlay
                    processed_frame = self.draw_info(processed_frame, results)
                    
                    # Update FPS
                    self.update_fps()
                    
                    # Display frame
                    cv2.imshow('Real-time Skeleton Detection', processed_frame)
                    frame_count += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f'skeleton_frame_{frame_count}_{int(time.time())}.jpg'
                    cv2.imwrite(filename, processed_frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('r'):
                    # Reset (reinitialize pose detector)
                    self.pose.close()
                    self.pose = self.mp_pose.Pose(
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                        model_complexity=1
                    )
                    print("Pose detector reset")
                elif key == ord(' '):
                    # Pause/Resume
                    paused = not paused
                    status = "Paused" if paused else "Resumed"
                    print(f"Processing {status}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"Processed {frame_count} frames")
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Process a video file for skeleton detection
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
        """
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found")
            return
        
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path is provided
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, results = self.detect_pose(frame)
                
                # Draw skeleton
                processed_frame = self.draw_landmarks(processed_frame, results)
                
                # Get landmarks for custom drawing
                landmarks = self.get_landmark_coordinates(results, processed_frame.shape)
                if landmarks:
                    processed_frame = self.draw_custom_skeleton(processed_frame, landmarks)
                
                # Draw information overlay
                processed_frame = self.draw_info(processed_frame, results)
                
                # Save frame if output writer is available
                if video_writer:
                    video_writer.write(processed_frame)
                
                # Display frame
                cv2.imshow('Video Skeleton Detection', processed_frame)
                
                frame_count += 1
                
                # Update progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed_time = time.time() - start_time
                    estimated_total_time = (elapsed_time / frame_count) * total_frames
                    remaining_time = estimated_total_time - elapsed_time
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), "
                          f"ETA: {remaining_time:.1f}s")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Processing interrupted by user")
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f'video_frame_{frame_count}_{int(time.time())}.jpg'
                    cv2.imwrite(filename, processed_frame)
                    print(f"Frame saved as {filename}")
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            processing_time = time.time() - start_time
            print(f"Processed {frame_count} frames in {processing_time:.2f} seconds")
            print(f"Average FPS: {frame_count / processing_time:.2f}")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'pose'):
            self.pose.close()


def main():
    """Main function to demonstrate the skeleton detector"""
    detector = SkeletonDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    print("Skeleton Detection System")
    print("1. Real-time camera detection")
    print("2. Video file processing")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        camera_index = input("Enter camera index (default: 0): ").strip()
        camera_index = int(camera_index) if camera_index.isdigit() else 0
        detector.process_realtime(camera_index)
    
    elif choice == "2":
        video_path = input("Enter video file path: ").strip()
        save_output = input("Save processed video? (y/n): ").strip().lower()
        
        output_path = None
        if save_output == 'y':
            output_path = input("Enter output file path (default: output_skeleton.mp4): ").strip()
            if not output_path:
                output_path = "output_skeleton.mp4"
        
        detector.process_video(video_path, output_path)
    
    else:
        print("Invalid choice. Please run the program again.")


if __name__ == "__main__":
    main()