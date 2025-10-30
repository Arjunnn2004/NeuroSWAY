import cv2
import mediapipe as mp
import numpy as np
import time


def minimal_skeleton_detection():
    """
    Minimal working skeleton detection - guaranteed to work
    """
    # Initialize MediaPipe Pose with minimal configuration
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Use basic pose configuration
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot access camera")
        print("Make sure:")
        print("1. Camera is connected")
        print("2. No other application is using the camera")
        print("3. Camera permissions are granted")
        return False
    
    print("✅ Camera initialized successfully!")
    print("📹 Starting skeleton detection...")
    print("🎮 Controls: Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Read frame from camera
            success, frame = cap.read()
            if not success:
                print("❌ Failed to read from camera")
                break
            
            # Convert color space (MediaPipe uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Improve performance by marking frame as not writeable
            rgb_frame.flags.writeable = False
            
            # Detect pose
            results = pose.process(rgb_frame)
            
            # Convert back to BGR for OpenCV
            rgb_frame.flags.writeable = True
            display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Draw pose landmarks if detected
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                
                # Add status text
                cv2.putText(display_frame, "POSE DETECTED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "NO POSE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = 30 / elapsed_time
                start_time = time.time()
                print(f"🎯 FPS: {fps:.1f}")
            
            # Add FPS to display
            elapsed = time.time() - start_time + 0.001  # Avoid division by zero
            current_fps = (frame_count % 30) / elapsed
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Skeleton Detection - Press Q to quit', display_frame)
            
            # Check for quit command
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        
        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n📊 Session Statistics:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Total time: {total_time:.1f}s")
    
    return True


def test_camera():
    """Test camera functionality"""
    print("🔍 Testing camera...")
    
    for i in range(3):
        print(f"   Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"   ✅ Camera {i}: {w}x{h} resolution")
                cap.release()
                return i
            cap.release()
        
    print("   ❌ No working camera found")
    return None


def main():
    """Main function"""
    print("=" * 50)
    print("🏃‍♂️ MINIMAL SKELETON DETECTION")
    print("=" * 50)
    
    # Test imports
    print("📦 Testing imports...")
    try:
        print(f"   ✅ OpenCV: {cv2.__version__}")
        print(f"   ✅ MediaPipe: {mp.__version__}")
        print(f"   ✅ NumPy: {np.__version__}")
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return
    
    # Test camera
    camera_index = test_camera()
    if camera_index is None:
        print("\n⚠️  No camera available. Please check your camera connection.")
        return
    
    print(f"\n🎥 Using camera {camera_index}")
    
    # Ask user if ready
    input("\n📋 Ready to start? Press Enter to begin (or Ctrl+C to cancel)...")
    
    # Run detection
    success = minimal_skeleton_detection()
    
    if success:
        print("\n✅ Session completed successfully!")
    else:
        print("\n❌ Session ended with errors")
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()