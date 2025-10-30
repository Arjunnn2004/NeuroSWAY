import cv2
import mediapipe as mp
import numpy as np
import time


def simple_skeleton_detection():
    """
    Simple real-time skeleton detection using MediaPipe
    Perfect for quick testing and demonstration
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Initialize pose detection
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Simple Skeleton Detection Started")
    print("Press 'q' to quit")
    
    # FPS calculation
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process the frame
        results = pose.process(rgb_frame)
        
        # Convert back to BGR
        rgb_frame.flags.writeable = True
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                bgr_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Calculate and display FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = end_time
            
        # Display FPS and status
        status = "Pose Detected" if results.pose_landmarks else "No Pose"
        cv2.putText(bgr_frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(bgr_frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Simple Skeleton Detection', bgr_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    simple_skeleton_detection()