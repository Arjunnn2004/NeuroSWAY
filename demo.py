#!/usr/bin/env python3
"""
Demo script for skeleton detection system
Demonstrates all features and provides easy testing
"""

import cv2
import sys
import os
import time


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_modules = ['cv2', 'mediapipe', 'numpy']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} - OK")
        except ImportError:
            print(f"❌ {module} - MISSING")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  Missing modules: {', '.join(missing_modules)}")
        print("Please install them using:")
        print("pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True


def check_camera():
    """Check if camera is available"""
    print("\n🎥 Checking camera availability...")
    
    for i in range(3):  # Check first 3 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera {i} - Available ({frame.shape[1]}x{frame.shape[0]})")
                cap.release()
                return i
            cap.release()
        else:
            print(f"❌ Camera {i} - Not available")
    
    print("⚠️  No camera found!")
    return None


def demo_simple():
    """Demo the simple skeleton detection"""
    print("\n🏃 Starting Simple Skeleton Detection Demo...")
    print("This will run for 30 seconds or until you press 'q'")
    
    try:
        from simple_skeleton import simple_skeleton_detection
        
        # Run simple detection with timeout
        import threading
        
        def run_detection():
            simple_skeleton_detection()
        
        thread = threading.Thread(target=run_detection)
        thread.daemon = True
        thread.start()
        
        # Wait for 30 seconds or until thread completes
        thread.join(timeout=30)
        
        print("✅ Simple demo completed!")
        
    except ImportError as e:
        print(f"❌ Error importing simple_skeleton: {e}")
    except Exception as e:
        print(f"❌ Error running simple demo: {e}")


def demo_full():
    """Demo the full skeleton detection system"""
    print("\n🤖 Starting Full Skeleton Detection Demo...")
    
    try:
        from skeleton_detector import SkeletonDetector
        
        detector = SkeletonDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        print("Demo will run for 60 seconds...")
        print("Controls: Q-quit, S-save frame, R-reset, SPACE-pause")
        
        detector.process_realtime(camera_index=0)
        
        print("✅ Full demo completed!")
        
    except ImportError as e:
        print(f"❌ Error importing skeleton_detector: {e}")
    except Exception as e:
        print(f"❌ Error running full demo: {e}")


def demo_advanced():
    """Demo the advanced skeleton detection with pose analysis"""
    print("\n🧠 Starting Advanced Skeleton Detection Demo...")
    
    try:
        from advanced_skeleton import AdvancedSkeletonDetector
        
        detector = AdvancedSkeletonDetector()
        
        print("Advanced demo with pose analysis and gesture detection...")
        print("Controls: Q-quit, S-save, P-save pose data, C-clear history")
        
        detector.process_advanced_realtime(camera_index=0)
        
        print("✅ Advanced demo completed!")
        
    except ImportError as e:
        print(f"❌ Error importing advanced_skeleton: {e}")
    except Exception as e:
        print(f"❌ Error running advanced demo: {e}")


def benchmark_performance():
    """Benchmark the performance of different model complexities"""
    print("\n⚡ Running Performance Benchmark...")
    
    try:
        import mediapipe as mp
        import numpy as np
        
        # Test different model complexities
        complexities = [0, 1, 2]
        results = {}
        
        for complexity in complexities:
            print(f"\nTesting model complexity {complexity}...")
            
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=complexity
            )
            
            # Create dummy image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Benchmark processing time
            start_time = time.time()
            iterations = 100
            
            for _ in range(iterations):
                rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                results_mp = pose.process(rgb_image)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / iterations
            fps = 1.0 / avg_time
            
            results[complexity] = {
                'avg_time': avg_time * 1000,  # Convert to milliseconds
                'fps': fps
            }
            
            print(f"  Average processing time: {avg_time * 1000:.2f}ms")
            print(f"  Estimated FPS: {fps:.1f}")
            
            pose.close()
        
        print("\n📊 Benchmark Results:")
        print("Model Complexity | Avg Time (ms) | Est. FPS")
        print("-" * 42)
        for complexity, result in results.items():
            print(f"       {complexity}         |    {result['avg_time']:6.2f}    |  {result['fps']:6.1f}")
        
        print("\nRecommendations:")
        print("- Complexity 0: Best for real-time applications (fastest)")
        print("- Complexity 1: Good balance of speed and accuracy")
        print("- Complexity 2: Best accuracy but slower (may cause lag)")
        
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")


def main():
    """Main demo function"""
    print("=" * 60)
    print("🏃‍♂️ SKELETON DETECTION SYSTEM DEMO")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check camera
    camera_index = check_camera()
    if camera_index is None:
        print("⚠️  Camera required for real-time demos. Some demos may not work.")
    
    while True:
        print("\n" + "=" * 40)
        print("Choose a demo option:")
        print("1. 🏃 Simple Skeleton Detection (Quick test)")
        print("2. 🤖 Full Skeleton Detection (Complete system)")
        print("3. 🧠 Advanced Detection (With pose analysis)")
        print("4. ⚡ Performance Benchmark")
        print("5. 📋 System Information")
        print("6. ❌ Exit")
        print("=" * 40)
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == "1":
            if camera_index is not None:
                demo_simple()
            else:
                print("❌ Camera required for this demo")
        
        elif choice == "2":
            if camera_index is not None:
                demo_full()
            else:
                print("❌ Camera required for this demo")
        
        elif choice == "3":
            if camera_index is not None:
                demo_advanced()
            else:
                print("❌ Camera required for this demo")
        
        elif choice == "4":
            benchmark_performance()
        
        elif choice == "5":
            print_system_info()
        
        elif choice == "6":
            print("👋 Thanks for trying the Skeleton Detection System!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-6.")
        
        input("\nPress Enter to continue...")


def print_system_info():
    """Print system information"""
    print("\n💻 System Information:")
    print("-" * 30)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # OpenCV version
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("OpenCV: Not installed")
    
    # MediaPipe version
    try:
        import mediapipe as mp
        print(f"MediaPipe version: {mp.__version__}")
    except ImportError:
        print("MediaPipe: Not installed")
    
    # NumPy version
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("NumPy: Not installed")
    
    # Available files
    print(f"\nProject files:")
    files = [
        "skeleton_detector.py",
        "simple_skeleton.py", 
        "advanced_skeleton.py",
        "requirements.txt",
        "config.json",
        "README.md"
    ]
    
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  ✅ {file} ({size} bytes)")
        else:
            print(f"  ❌ {file} (missing)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your installation and try again.")