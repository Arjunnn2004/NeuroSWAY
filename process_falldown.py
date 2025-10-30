import cv2
import mediapipe as mp
import numpy as np
import time
import os


def process_falldown_video():
    """
    Process the falldown.mp4 video with skeleton detection
    Specialized for fall detection analysis
    """
    video_path = "falldown.mp4"
    output_path = "falldown_skeleton_output.mp4"
    
    print("🎬 Processing falldown.mp4 with skeleton detection")
    print("=" * 50)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Error: {video_path} not found in current directory")
        return False
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📹 Video Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {duration:.2f} seconds")
    print()
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"💾 Output will be saved as: {output_path}")
    print("🔄 Processing frames...")
    print()
    
    frame_count = 0
    pose_detected_frames = 0
    start_time = time.time()
    
    # For fall detection analysis
    pose_data = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process frame
            results = pose.process(rgb_frame)
            
            # Convert back to BGR
            rgb_frame.flags.writeable = True
            processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Draw skeleton if pose detected
            pose_detected = False
            if results.pose_landmarks:
                pose_detected = True
                pose_detected_frames += 1
                
                # Draw landmarks with custom style for fall detection
                mp_drawing.draw_landmarks(
                    processed_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 255), thickness=3, circle_radius=3
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 0, 255), thickness=2
                    )
                )
                
                # Analyze pose for fall detection
                landmarks = results.pose_landmarks.landmark
                
                # Get key points for fall analysis
                if len(landmarks) >= 24:  # Ensure we have hip landmarks
                    left_hip = landmarks[23]
                    right_hip = landmarks[24]
                    nose = landmarks[0]
                    
                    # Calculate hip center position
                    hip_center_y = (left_hip.y + right_hip.y) / 2
                    
                    # Calculate if person might be falling (hip lower than usual)
                    # This is a simple heuristic - you can improve this
                    if hip_center_y > 0.7:  # Hip is in lower 30% of frame
                        cv2.putText(processed_frame, "POTENTIAL FALL DETECTED", 
                                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # Store pose data for analysis
                    pose_data.append({
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'hip_center_y': hip_center_y,
                        'nose_y': nose.y,
                        'pose_detected': True
                    })
            else:
                pose_data.append({
                    'frame': frame_count,
                    'time': frame_count / fps,
                    'pose_detected': False
                })
            
            # Add frame information
            timestamp = frame_count / fps
            cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Time: {timestamp:.2f}s", 
                       (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Pose: {'YES' if pose_detected else 'NO'}", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if pose_detected else (0, 0, 255), 2)
            
            # Write frame to output video
            out.write(processed_frame)
            
            # Display frame (optional - comment out for faster processing)
            cv2.imshow('Fall Detection - Press Q to quit', processed_frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % (fps * 2) == 0:  # Every 2 seconds
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / frame_count) * (total_frames - frame_count)
                print(f"📊 Progress: {progress:.1f}% ({frame_count}/{total_frames}) - ETA: {eta:.1f}s")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Processing interrupted by user")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        pose.close()
    
    # Final statistics
    processing_time = time.time() - start_time
    detection_rate = (pose_detected_frames / frame_count) * 100 if frame_count > 0 else 0
    
    print("\n" + "=" * 50)
    print("📊 PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"✅ Total frames processed: {frame_count}")
    print(f"🏃 Frames with pose detected: {pose_detected_frames}")
    print(f"📈 Detection rate: {detection_rate:.1f}%")
    print(f"⏱️ Processing time: {processing_time:.2f} seconds")
    print(f"🚀 Processing speed: {frame_count/processing_time:.1f} FPS")
    print(f"💾 Output saved as: {output_path}")
    
    # Analyze pose data for fall patterns
    analyze_fall_patterns(pose_data)
    
    return True


def analyze_fall_patterns(pose_data):
    """Analyze the pose data for fall patterns"""
    print("\n🔍 FALL PATTERN ANALYSIS:")
    print("-" * 30)
    
    detected_frames = [data for data in pose_data if data.get('pose_detected', False)]
    
    if not detected_frames:
        print("❌ No pose data available for analysis")
        return
    
    # Analyze hip position changes
    hip_positions = [data['hip_center_y'] for data in detected_frames if 'hip_center_y' in data]
    
    if hip_positions:
        avg_hip_y = np.mean(hip_positions)
        max_hip_y = np.max(hip_positions)
        min_hip_y = np.min(hip_positions)
        
        print(f"📍 Hip Position Analysis:")
        print(f"   Average hip Y position: {avg_hip_y:.3f}")
        print(f"   Highest hip position: {min_hip_y:.3f} (top of frame)")
        print(f"   Lowest hip position: {max_hip_y:.3f} (bottom of frame)")
        
        # Detect potential falls (rapid hip position changes)
        rapid_changes = 0
        for i in range(1, len(detected_frames)):
            if 'hip_center_y' in detected_frames[i] and 'hip_center_y' in detected_frames[i-1]:
                change = abs(detected_frames[i]['hip_center_y'] - detected_frames[i-1]['hip_center_y'])
                if change > 0.1:  # Significant position change
                    rapid_changes += 1
        
        print(f"⚡ Rapid position changes detected: {rapid_changes}")
        
        # Check for sustained low positions (potential ground contact)
        low_position_frames = sum(1 for pos in hip_positions if pos > 0.7)
        low_position_percentage = (low_position_frames / len(hip_positions)) * 100
        
        print(f"⬇️ Frames with low hip position: {low_position_frames} ({low_position_percentage:.1f}%)")
        
        if low_position_percentage > 50:
            print("🚨 HIGH PROBABILITY: Person appears to be on ground for significant duration")
        elif low_position_percentage > 20:
            print("⚠️ MEDIUM PROBABILITY: Possible fall or ground contact detected")
        else:
            print("✅ LOW PROBABILITY: Person appears to remain upright")


def main():
    """Main function"""
    print("🎯 FALL DETECTION VIDEO PROCESSOR")
    print("Using falldown.mp4 with skeleton detection")
    print()
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # List video files in directory
    video_files = [f for f in os.listdir('.') if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    print(f"🎬 Video files found: {video_files}")
    print()
    
    # Process the video
    success = process_falldown_video()
    
    if success:
        print("\n🎉 Video processing completed successfully!")
        print("👀 Check the output file: falldown_skeleton_output.mp4")
    else:
        print("\n❌ Video processing failed!")
    
    print("\n👋 Done!")


if __name__ == "__main__":
    main()