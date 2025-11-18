"""
NeuroSWAY - Real-time Fall Detection System for Parkinson's Patients
Live camera monitoring with threshold + injury prediction based fall detection
"""

import cv2
import sys
import time
from fall_detector import FallDetector
from datetime import datetime


def realtime_fall_detection(
    camera_index: int = 0,
    config_path: str = "fall_detection_config.json",
    weight_kg: float = 70.0,
):
    """
    Run real-time fall detection from camera feed

    Args:
        camera_index: Camera device index (default: 0)
        config_path: Path to configuration file
        weight_kg: Weight of the person in kilograms (used for impact energy)
    """
    print("=" * 70)
    print("🏥 NeuroSWAY - FALL DETECTION SYSTEM FOR PARKINSON'S PATIENTS")
    print("=" * 70)
    print(f"👤 Patient weight: {weight_kg:.1f} kg")
    print(f"📹 Initializing camera {camera_index}...")

    # Initialize fall detector
    try:
        detector = FallDetector(config_path)
        # attach weight to detector instance so impact-energy model can use it
        detector.weight_kg = float(weight_kg)
        print("✅ Fall detector initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing fall detector: {e}")
        return False

    # Open camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_index}")
        print("💡 Tip: Check if camera is connected and not used by another application")
        return False

    # Set camera properties from config if available
    camera_settings = detector.config.get("camera_settings", {})
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_settings.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_settings.get("height", 720))
    cap.set(cv2.CAP_PROP_FPS, camera_settings.get("fps", 30))

    # Get actual camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"📹 Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
    print()
    print("=" * 70)
    print("FALL DETECTION THRESHOLDS:")
    print("=" * 70)
    thresholds = detector.config["fall_detection_thresholds"]
    for key, value in thresholds.items():
        print(f"  • {key}: {value}")
    print()
    print("=" * 70)
    print("CONTROLS:")
    print("=" * 70)
    print("  Q or ESC  - Quit")
    print("  S         - Save current frame")
    print("  R         - Reset fall counter")
    print("  SPACE     - Pause/Resume")
    print("  T         - Show thresholds")
    print("  H         - Show/Hide help overlay")
    print("=" * 70)
    print()
    print("🟢 Fall detection started - Monitoring in progress...")
    print()

    paused = False
    frame_count = 0
    start_time = time.time()
    show_help = True
    processed_frame = None  # to avoid reference before assignment when paused

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break

                # Process frame for fall detection
                processed_frame, status, fall_result = detector.process_frame(frame)

                # Read fall-type & injury info from fall_result (computed inside FallDetector)
                fall_type = fall_result.get("fall_type", "unknown")
                injury_info = fall_result.get("injury_info", {}) or {}
                injury_risk = injury_info.get("injury_risk", "")
                expected_injury = injury_info.get("expected_injury", "")
                recommendation = injury_info.get("recommendation", "")

                # Overlay a compact summary line if a fall is detected
                if status == "FALL_DETECTED":
                    text = f"{fall_type} | risk: {injury_risk}"
                    cv2.putText(
                        processed_frame,
                        text,
                        (20, 80),  # just below the status bar drawn by FallDetector
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                # Add help overlay if enabled
                if show_help:
                    processed_frame = draw_help_overlay(processed_frame)

                # Display frame
                cv2.imshow(
                    "NeuroSWAY - Parkinson's Patient Monitoring", processed_frame
                )

                # Print status updates for significant events
                if status == "FALL_DETECTED" and fall_result.get("reasons"):
                    print(
                        f"\n🚨 FALL DETECTED at {datetime.now().strftime('%H:%M:%S')}!"
                    )
                    print(f"   Reasons: {', '.join(fall_result['reasons'])}")
                    if fall_type:
                        print(f"   Fall type: {fall_type}")
                    if injury_risk:
                        print(f"   Injury risk: {injury_risk}")
                    if expected_injury:
                        print(f"   Expected injury: {expected_injury}")
                    if recommendation:
                        print(f"   Recommendation: {recommendation}")
                    print(f"   Total falls today: {detector.fall_count}")
                    print()
                elif status == "WARNING" and frame_count % 30 == 0:  # every ~1s
                    if fall_result.get("reasons"):
                        print(f"⚠️  Warning: {', '.join(fall_result['reasons'][:2])}")

                frame_count += 1

            else:
                # Show paused message
                if processed_frame is not None:
                    paused_frame = processed_frame.copy()
                else:
                    # In rare case paused before first frame
                    ret, frame = cap.read()
                    if not ret:
                        break
                    paused_frame = frame.copy()

                h, w = paused_frame.shape[:2]
                cv2.rectangle(
                    paused_frame,
                    (w // 2 - 150, h // 2 - 50),
                    (w // 2 + 150, h // 2 + 50),
                    (0, 0, 0),
                    -1,
                )
                cv2.rectangle(
                    paused_frame,
                    (w // 2 - 150, h // 2 - 50),
                    (w // 2 + 150, h // 2 + 50),
                    (255, 255, 255),
                    3,
                )
                cv2.putText(
                    paused_frame,
                    "PAUSED",
                    (w // 2 - 80, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    3,
                )
                cv2.imshow(
                    "NeuroSWAY - Parkinson's Patient Monitoring", paused_frame
                )

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # Q or ESC
                print("\n🛑 Shutting down NeuroSWAY...")
                break

            elif key == ord("s"):
                # Save current frame
                if processed_frame is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"manual_save_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"📸 Frame saved: {filename}")
                else:
                    print("⚠️  No frame available to save yet.")

            elif key == ord("r"):
                # Reset fall counter
                old_count = detector.fall_count
                detector.fall_count = 0
                detector.fall_events = []
                print(f"🔄 Fall counter reset (was {old_count}, now 0)")

            elif key == ord(" "):
                # Pause/Resume
                paused = not paused
                status_msg = "⏸️  PAUSED" if paused else "▶️  RESUMED"
                print(status_msg)

            elif key == ord("h"):
                # Toggle help overlay
                show_help = not show_help
                help_status = "shown" if show_help else "hidden"
                print(f"ℹ️  Help overlay {help_status}")

            elif key == ord("t"):
                # Show threshold adjustment info
                print("\n" + "=" * 70)
                print("CURRENT THRESHOLDS:")
                print("=" * 70)
                for k, v in detector.config["fall_detection_thresholds"].items():
                    print(f"  {k}: {v}")
                print("=" * 70)
                print(
                    "💡 To adjust thresholds, edit fall_detection_config.json and restart"
                )
                print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")

    except Exception as e:
        print(f"\n❌ Error during fall detection: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\n📊 Generating session statistics...")

        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print(f"  Duration: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"  Frames processed: {frame_count}")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Total falls detected: {detector.fall_count}")
        print(f"  Patient weight (kg): {getattr(detector, 'weight_kg', 'N/A')}")
        print("=" * 70)

        # Save statistics
        detector.save_statistics()

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        print("✅ NeuroSWAY shut down successfully")
        print()

        return True


def draw_help_overlay(image):
    """Draw help overlay on image"""
    h, w = image.shape[:2]

    # Semi-transparent background for help text
    overlay = image.copy()
    help_x = w - 320
    help_y = 80
    help_height = 180

    cv2.rectangle(
        overlay,
        (help_x - 10, help_y - 10),
        (w - 10, help_y + help_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # Draw help text
    cv2.putText(
        image,
        "QUICK HELP",
        (help_x, help_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    help_text = [
        "Q/ESC - Quit",
        "S - Save frame",
        "R - Reset counter",
        "SPACE - Pause",
        "T - Show thresholds",
        "H - Hide help",
    ]

    y_offset = help_y + 45
    for text in help_text:
        cv2.putText(
            image,
            text,
            (help_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_offset += 22

    return image


def video_fall_detection(
    video_path: str,
    config_path: str = "fall_detection_config.json",
    output_path: str = None,
    weight_kg: float = 70.0,
):
    """
    Process video file for fall detection

    Args:
        video_path: Path to input video file
        config_path: Path to configuration file
        output_path: Path to save output video (optional)
        weight_kg: Patient weight in kg (for impact energy estimation)
    """
    print("=" * 70)
    print("🎬 VIDEO FALL DETECTION ANALYSIS")
    print("=" * 70)
    print(f"📹 Input: {video_path}")
    if output_path:
        print(f"💾 Output: {output_path}")
    print()

    # Initialize fall detector
    try:
        detector = FallDetector(config_path)
        detector.weight_kg = float(weight_kg)
        print("✅ Fall detector initialized")
    except Exception as e:
        print(f"❌ Error initializing fall detector: {e}")
        return False

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return False

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"📹 Video: {width}x{height}, {fps} FPS, {total_frames} frames, {duration:.1f}s")
    print()

    # Setup video writer if output path provided
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print("💾 Output video writer initialized")

    print("🔄 Processing video...")
    print()

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame, status, fall_result = detector.process_frame(frame)

            fall_type = fall_result.get("fall_type", "unknown")
            injury_info = fall_result.get("injury_info", {}) or {}
            injury_risk = injury_info.get("injury_risk", "")

            # Overlay short summary if fall detected
            if status == "FALL_DETECTED":
                cv2.putText(
                    processed_frame,
                    f"{fall_type} | risk: {injury_risk}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Save to output video
            if video_writer:
                video_writer.write(processed_frame)

            # Display frame
            cv2.imshow(
                "Video Fall Detection Analysis - Press Q to quit", processed_frame
            )

            frame_count += 1

            # Progress update
            if frame_count % (max(fps, 1) * 2) == 0 or frame_count == 1:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                elapsed = time.time() - start_time
                eta = (
                    (elapsed / frame_count) * (total_frames - frame_count)
                    if frame_count > 0 and total_frames > 0
                    else 0
                )
                print(
                    f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                    f"ETA: {eta:.1f}s - Falls: {detector.fall_count}"
                )

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n⚠️  Processing interrupted by user")
                break

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        processing_time = time.time() - start_time
        processing_fps = frame_count / processing_time if processing_time > 0 else 0

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"  Frames processed: {frame_count}/{total_frames}")
        print(f"  Processing time: {processing_time:.1f}s")
        print(f"  Processing speed: {processing_fps:.1f} FPS")
        print(f"  Total falls detected: {detector.fall_count}")
        print("=" * 70)

        # Save statistics
        detector.save_statistics()

        # Release resources
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        print("✅ Video analysis complete")
        return True


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("🏥 NeuroSWAY - FALL DETECTION SYSTEM FOR PARKINSON'S PATIENTS")
    print("   Threshold-Based Real-Time Monitoring + Injury Prediction")
    print("=" * 70)
    print()
    print("Select mode:")
    print("  1. Real-time camera monitoring (recommended)")
    print("  2. Video file analysis")
    print("  3. Exit")
    print()

    choice = input("Enter your choice (1-3): ").strip()

    if choice == "1":
        print()
        # Ask for patient weight
        weight_str = input("Enter patient weight in kg (e.g., 70): ").strip()
        try:
            weight_kg = float(weight_str)
            if weight_kg <= 0:
                raise ValueError
        except ValueError:
            print("⚠️  Invalid weight entered. Using default 70 kg.")
            weight_kg = 70.0

        camera_idx = input("Enter camera index (default: 0): ").strip()
        camera_idx = int(camera_idx) if camera_idx.isdigit() else 0

        config = input(
            "Config file (default: fall_detection_config.json): "
        ).strip()
        config = config if config else "fall_detection_config.json"

        print()
        realtime_fall_detection(camera_idx, config, weight_kg)

    elif choice == "2":
        print()
        video_path = input("Enter video file path: ").strip()

        # Ask for weight even in video mode so energy estimates are realistic
        weight_str = input("Enter patient weight in kg (e.g., 70): ").strip()
        try:
            weight_kg = float(weight_str)
            if weight_kg <= 0:
                raise ValueError
        except ValueError:
            print("⚠️  Invalid weight entered. Using default 70 kg.")
            weight_kg = 70.0

        save = input("Save processed video? (y/n): ").strip().lower()
        output_path = None
        if save == "y":
            output_path = input(
                "Output path (default: fall_detection_output.mp4): "
            ).strip()
            output_path = output_path if output_path else "fall_detection_output.mp4"

        config = input(
            "Config file (default: fall_detection_config.json): "
        ).strip()
        config = config if config else "fall_detection_config.json"

        print()
        video_fall_detection(video_path, config, output_path, weight_kg)

    elif choice == "3":
        print("👋 Goodbye!")

    else:
        print("❌ Invalid choice. Please run again.")


if __name__ == "__main__":
    main()
