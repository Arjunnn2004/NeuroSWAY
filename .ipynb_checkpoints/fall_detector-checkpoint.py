"""
NeuroSWAY - Fall Detection System for Parkinson's Patients
Threshold-based fall detection using pose estimation
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
from typing import Optional, Tuple, List, Dict
from collections import deque
from datetime import datetime
import math


class FallDetector:
    """
    Fall detection system using threshold-based algorithms
    Specifically designed for monitoring Parkinson's patients
    """

    def __init__(self, config_path: str = "fall_detection_config.json"):
        """
        Initialize the fall detector

        Args:
            config_path: Path to configuration JSON file
        """
        # Load configuration
        self.config = self.load_config(config_path)

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.config["detection_confidence"],
            min_tracking_confidence=self.config["tracking_confidence"],
            model_complexity=self.config["model_complexity"],
            static_image_mode=False,
            smooth_landmarks=True,
            enable_segmentation=False,
        )

        # Patient weight (can be overridden from outside, e.g., realtime_fall_detection)
        self.weight_kg = self.config.get("patient_weight_kg", 70.0)

        # Fall detection state
        self.fall_detected = False
        self.fall_warning = False
        self.fall_start_time = None
        self.fall_count = 0
        self.consecutive_fall_frames = 0
        self.consecutive_normal_frames = 0
        self.cooldown_counter = 0

        # History tracking
        self.position_history = deque(
            maxlen=self.config["fall_detection_settings"]["history_window_size"]
        )
        self.velocity_history = deque(maxlen=30)
        self.angle_history = deque(maxlen=30)

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Fall event log
        self.fall_events = []

        # Create output directories
        self.setup_directories()

    def load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return self.get_default_config()

        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config

    def get_default_config(self) -> dict:
        """Return default configuration"""
        return {
            "detection_confidence": 0.5,
            "tracking_confidence": 0.5,
            "model_complexity": 1,
            "patient_weight_kg": 70.0,
            "fall_detection_thresholds": {
                "hip_height_threshold": 0.65,
                "hip_height_critical": 0.75,
                "torso_angle_threshold": 45,
                "torso_angle_critical": 70,
                "head_velocity_threshold": 0.15,
                "head_velocity_critical": 0.30,
                "hip_velocity_threshold": 0.12,
                "hip_velocity_critical": 0.25,
                "aspect_ratio_threshold": 1.8,
                "ground_contact_threshold": 0.85,
                "shoulder_hip_ratio_threshold": 0.3,
            },
            "fall_detection_settings": {
                "consecutive_frames_for_fall": 3,
                "consecutive_frames_for_recovery": 10,
                "history_window_size": 30,
                "cooldown_frames": 60,
                "enable_angle_check": True,
                "enable_velocity_check": True,
                "enable_aspect_ratio_check": True,
            },
            "alert_settings": {
                "enable_visual_alert": True,
                "enable_log_file": True,
                "auto_save_fall_frames": True,
            },
            "visualization": {
                "draw_landmarks": True,
                "draw_skeleton": True,
                "show_angles": True,
                "show_status_panel": True,
            },
            "colors": {
                "normal": [0, 255, 0],
                "warning": [0, 165, 255],
                "critical": [0, 0, 255],
            },
            "output_settings": {
                "save_logs": True,
                "log_directory": "fall_logs",
                "fall_frames_directory": "fall_detections",
                "save_statistics": True,
                "save_fall_frames": True,
            },
        }

    def setup_directories(self):
        """Create necessary output directories"""
        if self.config.get("output_settings", {}).get("save_logs", True):
            log_dir = self.config["output_settings"]["log_directory"]
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                print(f"📁 Created log directory: {log_dir}")

        if self.config.get("output_settings", {}).get("save_fall_frames", True):
            frames_dir = self.config["output_settings"]["fall_frames_directory"]
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)
                print(f"📁 Created fall frames directory: {frames_dir}")

    def calculate_distance(self, point1, point2) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def calculate_angle(self, point1, point2, point3) -> float:
        """
        Calculate angle at point2 formed by three points
        Returns angle in degrees
        """
        # Vector from point2 to point1
        v1 = np.array([point1.x - point2.x, point1.y - point2.y])
        # Vector from point2 to point3
        v2 = np.array([point3.x - point2.x, point3.y - point2.y])

        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def calculate_torso_angle(self, landmarks) -> float:
        """
        Calculate torso angle from vertical
        Returns angle in degrees (0 = vertical, 90 = horizontal)
        """
        # Use shoulder midpoint and hip midpoint
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2

        # Calculate angle from vertical (90 degrees - angle from horizontal)
        dx = hip_mid_x - shoulder_mid_x
        dy = hip_mid_y - shoulder_mid_y

        angle_rad = math.atan2(abs(dx), abs(dy))
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def detect_fall(self, landmarks, image_shape: Tuple[int, int]) -> Dict:
        """
        Detect fall using multiple threshold-based criteria

        Args:
            landmarks: MediaPipe pose landmarks
            image_shape: (height, width) of the image

        Returns:
            Dictionary with fall detection results and metrics
        """
        if not landmarks or len(landmarks) < 33:
            return {
                "fall_detected": False,
                "fall_warning": False,
                "metrics": {},
                "reasons": ["No landmarks detected"],
                "impact_point": None,
                "impact_energy": 0.0,
            }

        height, width = image_shape[:2]
        thresholds = self.config["fall_detection_thresholds"]
        settings = self.config["fall_detection_settings"]

        # Extract key landmarks
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Calculate centers
        hip_center_y = (left_hip.y + right_hip.y) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        knee_center_y = (left_knee.y + right_knee.y) / 2
        ankle_center_y = (left_ankle.y + right_ankle.y) / 2

        # Init result dict
        results = {
            "fall_detected": False,
            "fall_warning": False,
            "metrics": {},
            "reasons": [],
            "impact_point": None,
            "impact_energy": 0.0,
        }

        # 1. Hip height check (primary indicator)
        results["metrics"]["hip_height"] = hip_center_y

        if hip_center_y > thresholds.get("ground_contact_threshold", 0.85):
            results["fall_detected"] = True
            results["reasons"].append(f"Ground contact: Hip at {hip_center_y:.2f}")
        elif hip_center_y > thresholds["hip_height_critical"]:
            results["fall_warning"] = True
            results["reasons"].append(f"Critical hip height: {hip_center_y:.2f}")
        elif hip_center_y > thresholds["hip_height_threshold"]:
            results["fall_warning"] = True
            results["reasons"].append(f"Low hip height: {hip_center_y:.2f}")

        # 2. Torso angle check
        if settings.get("enable_angle_check", True):
            torso_angle = self.calculate_torso_angle(landmarks)
            results["metrics"]["torso_angle"] = torso_angle
            self.angle_history.append(torso_angle)

            if torso_angle > thresholds["torso_angle_critical"]:
                results["fall_detected"] = True
                results["reasons"].append(
                    f"Critical torso angle: {torso_angle:.1f}°"
                )
            elif torso_angle > thresholds["torso_angle_threshold"]:
                results["fall_warning"] = True
                results["reasons"].append(f"High torso angle: {torso_angle:.1f}°")

        # 3. Vertical velocity check (if history available)
        head_velocity = 0.0
        hip_velocity = 0.0
        shoulder_velocity = 0.0
        knee_velocity = 0.0
        ankle_velocity = 0.0

        if settings.get("enable_velocity_check", True) and len(self.position_history) > 5:
            current_head_y = nose.y
            prev = self.position_history[-5]

            prev_head_y = prev["nose_y"]
            prev_hip_y = prev["hip_y"]
            prev_shoulder_y = prev["shoulder_y"]
            prev_knee_y = prev.get("knee_y", knee_center_y)
            prev_ankle_y = prev.get("ankle_y", ankle_center_y)

            head_velocity = abs(current_head_y - prev_head_y)
            hip_velocity = abs(hip_center_y - prev_hip_y)
            shoulder_velocity = abs(shoulder_mid_y - prev_shoulder_y)
            knee_velocity = abs(knee_center_y - prev_knee_y)
            ankle_velocity = abs(ankle_center_y - prev_ankle_y)

            results["metrics"]["head_velocity"] = head_velocity
            results["metrics"]["hip_velocity"] = hip_velocity
            results["metrics"]["shoulder_velocity"] = shoulder_velocity
            results["metrics"]["knee_velocity"] = knee_velocity
            results["metrics"]["ankle_velocity"] = ankle_velocity

            if (
                head_velocity > thresholds["head_velocity_critical"]
                or hip_velocity > thresholds["hip_velocity_critical"]
            ):
                results["fall_detected"] = True
                results["reasons"].append(
                    f"Critical velocity: head={head_velocity:.3f}, hip={hip_velocity:.3f}"
                )
            elif (
                head_velocity > thresholds["head_velocity_threshold"]
                or hip_velocity > thresholds["hip_velocity_threshold"]
            ):
                results["fall_warning"] = True
                results["reasons"].append(
                    f"High velocity: head={head_velocity:.3f}, hip={hip_velocity:.3f}"
                )

        # 4. Aspect ratio check (body orientation)
        if settings.get("enable_aspect_ratio_check", True):
            all_x = [lm.x for lm in landmarks if lm.visibility > 0.5]
            all_y = [lm.y for lm in landmarks if lm.visibility > 0.5]

            if all_x and all_y:
                body_width = max(all_x) - min(all_x)
                body_height = max(all_y) - min(all_y)
                aspect_ratio = body_width / (body_height + 1e-6)
                results["metrics"]["aspect_ratio"] = aspect_ratio

                if aspect_ratio > thresholds["aspect_ratio_threshold"]:
                    results["fall_warning"] = True
                    results["reasons"].append(
                        f"Horizontal orientation: AR={aspect_ratio:.2f}"
                    )

        # 5. Shoulder-hip vertical distance check
        shoulder_hip_distance = abs(hip_center_y - shoulder_mid_y)
        results["metrics"]["shoulder_hip_distance"] = shoulder_hip_distance

        if shoulder_hip_distance < thresholds.get("shoulder_hip_ratio_threshold", 0.3):
            results["fall_warning"] = True
            results["reasons"].append(f"Compressed torso: {shoulder_hip_distance:.2f}")

        # ---- Impact point & energy estimation ----
        # Only attempt if we have some velocity history
        if len(self.position_history) > 5:
            # Current positions (y: 0 = top, 1 = bottom)
            positions = {
                "head": nose.y,
                "shoulder": shoulder_mid_y,
                "hip": hip_center_y,
                "knee": knee_center_y,
                "ankle": ankle_center_y,
            }

            velocities = {
                "head": head_velocity,
                "shoulder": shoulder_velocity,
                "hip": hip_velocity,
                "knee": knee_velocity,
                "ankle": ankle_velocity,
            }

            vulnerability = {
                "head": 1.3,
                "shoulder": 1.0,
                "hip": 1.0,
                "knee": 0.7,
                "ankle": 0.5,
            }

            impact_scores = {}
            for part in positions.keys():
                y = positions[part]
                v = velocities.get(part, 0.0)

                # ground_proximity: 0 at y<=0.5, 1 at y>=1.0
                ground_factor = max(0.0, min(1.0, (y - 0.5) / 0.5))
                score = ground_factor * (v ** 2) * vulnerability[part]
                impact_scores[part] = score

            # Choose part with highest score
            best_part = max(impact_scores, key=impact_scores.get)
            best_score = impact_scores[best_part]
            best_velocity = velocities.get(best_part, 0.0)

            if best_score > 0.0:
                mass = float(getattr(self, "weight_kg", 70.0))
                impact_energy = 0.5 * mass * (best_velocity ** 2)

                results["impact_point"] = best_part
                results["impact_energy"] = impact_energy
                results["metrics"]["impact_velocity"] = best_velocity

        # Store position history (for next frame velocities)
        self.position_history.append(
            {
                "timestamp": time.time(),
                "nose_y": nose.y,
                "hip_y": hip_center_y,
                "shoulder_y": shoulder_mid_y,
                "knee_y": knee_center_y,
                "ankle_y": ankle_center_y,
            }
        )

        return results

    def update_fall_state(self, fall_result: Dict) -> str:
        """
        Update fall detection state based on detection results
        Uses consecutive frame logic to avoid false positives

        Returns:
            Status string: 'FALL_DETECTED', 'WARNING', 'NORMAL', 'RECOVERING'
        """
        settings = self.config["fall_detection_settings"]

        # Cooldown logic (prevent repeated triggers)
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.fall_detected:
                return "RECOVERING"

        # Update consecutive frame counters
        if fall_result["fall_detected"]:
            self.consecutive_fall_frames += 1
            self.consecutive_normal_frames = 0
        elif fall_result["fall_warning"]:
            self.consecutive_normal_frames = 0
        else:
            self.consecutive_fall_frames = 0
            self.consecutive_normal_frames += 1

        # Fall detection logic
        if self.consecutive_fall_frames >= settings["consecutive_frames_for_fall"]:
            if not self.fall_detected:
                # New fall detected
                self.fall_detected = True
                self.fall_start_time = datetime.now()
                self.fall_count += 1
                self.log_fall_event(fall_result)
                self.cooldown_counter = settings["cooldown_frames"]
            return "FALL_DETECTED"

        # Recovery logic
        if (
            self.fall_detected
            and self.consecutive_normal_frames >= settings["consecutive_frames_for_recovery"]
        ):
            self.fall_detected = False
            return "RECOVERING"

        # Warning state
        if fall_result["fall_warning"] or self.consecutive_fall_frames > 0:
            self.fall_warning = True
            return "WARNING"

        # Normal state
        self.fall_warning = False
        return "NORMAL"

    def log_fall_event(self, fall_result: Dict):
        """Log fall event details"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "fall_number": self.fall_count,
            "metrics": fall_result["metrics"],
            "reasons": fall_result["reasons"],
            "impact_point": fall_result.get("impact_point"),
            "impact_energy": fall_result.get("impact_energy"),
            "weight_kg": float(getattr(self, "weight_kg", 70.0)),
        }
        self.fall_events.append(event)

        # Save to log file
        if self.config["output_settings"]["save_logs"]:
            log_path = os.path.join(
                self.config["output_settings"]["log_directory"],
                f"fall_log_{datetime.now().strftime('%Y%m%d')}.json",
            )

            try:
                # Load existing log or create new
                if os.path.exists(log_path):
                    with open(log_path, "r") as f:
                        log_data = json.load(f)
                else:
                    log_data = {"falls": []}

                log_data["falls"].append(event)

                with open(log_path, "w") as f:
                    json.dump(log_data, f, indent=2)

                print(f"📝 Fall event logged to {log_path}")
            except Exception as e:
                print(f"⚠️  Error logging fall event: {e}")

    def draw_skeleton(self, image: np.ndarray, results) -> np.ndarray:
        """Draw pose landmarks on image"""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        return image

    def draw_status_panel(self, image: np.ndarray, status: str, fall_result: Dict) -> np.ndarray:
        """Draw status panel with fall detection information"""
        height, width = image.shape[:2]

        # Define colors
        colors = self.config["colors"]
        if status == "FALL_DETECTED":
            panel_color = tuple(colors["critical"])
            status_text = "⚠️ FALL DETECTED ⚠️"
        elif status == "WARNING":
            panel_color = tuple(colors["warning"])
            status_text = "⚠ WARNING: Possible Fall"
        elif status == "RECOVERING":
            panel_color = tuple(colors["warning"])
            status_text = "🔄 Recovering..."
        else:
            panel_color = tuple(colors["normal"])
            status_text = "✓ Normal"

        # Draw status bar at top
        cv2.rectangle(image, (0, 0), (width, 60), panel_color, -1)
        cv2.putText(
            image,
            status_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
        )

        # Draw fall count
        cv2.putText(
            image,
            f"Falls Today: {self.fall_count}",
            (width - 250, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Draw metrics panel on left side
        panel_x = 10
        panel_y = 80
        line_height = 30

        cv2.rectangle(
            image,
            (panel_x - 5, panel_y - 5),
            (350, panel_y + line_height * 10),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            image,
            (panel_x - 5, panel_y - 5),
            (350, panel_y + line_height * 10),
            (255, 255, 255),
            2,
        )

        # Display metrics
        metrics = fall_result.get("metrics", {})
        y = panel_y + 20

        cv2.putText(
            image,
            "METRICS:",
            (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        y += line_height

        # Hip height
        if "hip_height" in metrics:
            threshold = self.config["fall_detection_thresholds"]["hip_height_threshold"]
            value = metrics["hip_height"]
            color = (0, 0, 255) if value > threshold else (0, 255, 0)
            cv2.putText(
                image,
                f"Hip Height: {value:.3f} (T:{threshold:.2f})",
                (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            y += line_height

        # Torso angle
        if "torso_angle" in metrics:
            threshold = self.config["fall_detection_thresholds"]["torso_angle_threshold"]
            value = metrics["torso_angle"]
            color = (0, 0, 255) if value > threshold else (0, 255, 0)
            cv2.putText(
                image,
                f"Torso Angle: {value:.1f}deg (T:{threshold}deg)",
                (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            y += line_height

        # Velocities
        if "head_velocity" in metrics:
            threshold = self.config["fall_detection_thresholds"]["head_velocity_threshold"]
            value = metrics["head_velocity"]
            color = (0, 0, 255) if value > threshold else (0, 255, 0)
            cv2.putText(
                image,
                f"Head Velocity: {value:.3f} (T:{threshold:.2f})",
                (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            y += line_height

        if "hip_velocity" in metrics:
            threshold = self.config["fall_detection_thresholds"]["hip_velocity_threshold"]
            value = metrics["hip_velocity"]
            color = (0, 0, 255) if value > threshold else (0, 255, 0)
            cv2.putText(
                image,
                f"Hip Velocity: {value:.3f} (T:{threshold:.2f})",
                (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            y += line_height

        # Aspect ratio
        if "aspect_ratio" in metrics:
            threshold = self.config["fall_detection_thresholds"]["aspect_ratio_threshold"]
            value = metrics["aspect_ratio"]
            color = (0, 0, 255) if value > threshold else (0, 255, 0)
            cv2.putText(
                image,
                f"Aspect Ratio: {value:.2f} (T:{threshold:.1f})",
                (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            y += line_height

        # Impact info
        impact_point = fall_result.get("impact_point")
        impact_energy = fall_result.get("impact_energy", 0.0)
        if impact_point is not None:
            cv2.putText(
                image,
                f"Impact Point: {impact_point}",
                (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y += line_height
            cv2.putText(
                image,
                f"Impact Energy (rel): {impact_energy:.3f}",
                (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y += line_height

        # FPS
        cv2.putText(
            image,
            f"FPS: {self.current_fps:.1f}",
            (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Draw reasons if warning or fall
        if fall_result.get("reasons"):
            reason_y = height - 150
            cv2.putText(
                image,
                "ALERTS:",
                (panel_x, reason_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            reason_y += 25
            for reason in fall_result["reasons"][:3]:  # Show max 3 reasons
                cv2.putText(
                    image,
                    f"• {reason}",
                    (panel_x, reason_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                reason_y += 20

        # Draw controls at bottom
        controls_y = height - 80
        cv2.putText(
            image,
            "Controls: Q-Quit | S-Save | R-Reset | SPACE-Pause",
            (panel_x, controls_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        return image

    def save_fall_frame(self, image: np.ndarray):
        """Save frame when fall is detected"""
        if self.config["alert_settings"]["auto_save_fall_frames"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fall_{self.fall_count}_{timestamp}.jpg"
            filepath = os.path.join(
                self.config["output_settings"]["fall_frames_directory"], filename
            )
            cv2.imwrite(filepath, image)
            print(f"📸 Fall frame saved: {filepath}")

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            end_time = time.time()
            self.current_fps = 30 / (end_time - self.fps_start_time)
            self.fps_start_time = end_time

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, Dict]:
        """
        Process a single frame for fall detection

        Returns:
            Tuple of (processed_frame, status, fall_result)
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Detect pose
        results = self.pose.process(rgb_frame)

        # Convert back to BGR
        rgb_frame.flags.writeable = True
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Detect fall
        fall_result = {
            "fall_detected": False,
            "fall_warning": False,
            "metrics": {},
            "reasons": [],
            "impact_point": None,
            "impact_energy": 0.0,
        }
        if results.pose_landmarks:
            fall_result = self.detect_fall(results.pose_landmarks.landmark, frame.shape)

            # Draw skeleton
            if self.config["visualization"]["draw_skeleton"]:
                processed_frame = self.draw_skeleton(processed_frame, results)

        # Update fall state
        status = self.update_fall_state(fall_result)

        # Draw status panel
        if self.config["visualization"]["show_status_panel"]:
            processed_frame = self.draw_status_panel(
                processed_frame, status, fall_result
            )

        # Save frame if fall detected
        if status == "FALL_DETECTED" and self.fall_start_time:
            time_since_fall = (datetime.now() - self.fall_start_time).total_seconds()
            if time_since_fall < 1:  # Save only once per fall (within first second)
                self.save_fall_frame(processed_frame)

        # Update FPS
        self.update_fps()

        return processed_frame, status, fall_result

    def get_statistics(self) -> Dict:
        """Get fall detection statistics"""
        return {
            "total_falls": self.fall_count,
            "fall_events": self.fall_events,
            "current_fps": self.current_fps,
            "fall_detected": self.fall_detected,
            "fall_warning": self.fall_warning,
            "weight_kg": float(getattr(self, "weight_kg", 70.0)),
        }

    def save_statistics(self):
        """Save statistics to file"""
        if self.config["output_settings"].get("save_statistics", True):
            stats = self.get_statistics()
            filename = (
                f"fall_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            filepath = os.path.join(
                self.config["output_settings"]["log_directory"], filename
            )
            with open(filepath, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"📊 Statistics saved: {filepath}")

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, "pose"):
            self.pose.close()
