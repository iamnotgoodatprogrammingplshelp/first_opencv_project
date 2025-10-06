"""Motion and action detection using OpenCV and MediaPipe."""

import logging
from typing import Dict, List, Tuple, Optional
from collections import deque

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available. Advanced pose detection disabled.")

logger = logging.getLogger(__name__)


class MotionDetector:
    """Detect motion and classify actions."""

    ACTIONS = ["standing", "sitting", "walking", "running", "jumping", "waving", "unknown"]

    def __init__(
        self,
        use_pose_detection: bool = True,
        motion_threshold: int = 25,
        history_size: int = 30
    ):
        """Initialize motion detector.
        
        Args:
            use_pose_detection: Use MediaPipe for pose detection.
            motion_threshold: Threshold for motion detection (0-255).
            history_size: Number of frames to keep in history.
        """
        self.use_pose_detection = use_pose_detection and MEDIAPIPE_AVAILABLE
        self.motion_threshold = motion_threshold
        self.history_size = history_size
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        # Motion history
        self.motion_history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=history_size)
        
        # MediaPipe pose detection
        self.pose_detector = None
        if self.use_pose_detection:
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe pose detection initialized")
        
        self.prev_frame = None
        self.frame_count = 0

    def detect_motion_simple(self, frame: np.ndarray) -> Dict[str, any]:
        """Detect motion using background subtraction.
        
        Args:
            frame: Input frame (BGR).
            
        Returns:
            Dictionary with motion detection results.
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (value 127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Calculate motion intensity
        motion_pixels = cv2.countNonZero(fg_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        motion_intensity = motion_pixels / total_pixels
        
        # Find largest contour (main moving object)
        bounding_box = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(largest_contour)
                bounding_box = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        
        return {
            "motion_detected": motion_intensity > 0.01,
            "motion_intensity": float(motion_intensity),
            "bounding_box": bounding_box,
            "fg_mask": fg_mask,
            "contour_count": len(contours),
        }

    def calculate_velocity(
        self, prev_bbox: Optional[Dict], curr_bbox: Optional[Dict], fps: float = 30.0
    ) -> float:
        """Calculate velocity between two bounding boxes.
        
        Args:
            prev_bbox: Previous bounding box.
            curr_bbox: Current bounding box.
            fps: Frames per second.
            
        Returns:
            Velocity in pixels per second.
        """
        if prev_bbox is None or curr_bbox is None:
            return 0.0
        
        # Calculate center points
        prev_center_x = prev_bbox["x"] + prev_bbox["w"] / 2
        prev_center_y = prev_bbox["y"] + prev_bbox["h"] / 2
        
        curr_center_x = curr_bbox["x"] + curr_bbox["w"] / 2
        curr_center_y = curr_bbox["y"] + curr_bbox["h"] / 2
        
        # Calculate distance
        distance = np.sqrt(
            (curr_center_x - prev_center_x) ** 2 +
            (curr_center_y - prev_center_y) ** 2
        )
        
        # Convert to velocity (pixels per second)
        velocity = distance * fps
        
        return float(velocity)

    def detect_pose(self, frame: np.ndarray) -> Optional[Dict[str, any]]:
        """Detect human pose using MediaPipe.
        
        Args:
            frame: Input frame (BGR).
            
        Returns:
            Dictionary with pose landmarks or None.
        """
        if not self.use_pose_detection or self.pose_detector is None:
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose_detector.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility,
            })
        
        return {
            "landmarks": landmarks,
            "pose_world_landmarks": results.pose_world_landmarks,
        }

    def classify_action(
        self,
        motion_intensity: float,
        velocity: float,
        pose_data: Optional[Dict] = None
    ) -> Tuple[str, float]:
        """Classify action based on motion data.
        
        Args:
            motion_intensity: Motion intensity (0.0 to 1.0).
            velocity: Movement velocity.
            pose_data: Pose landmarks data.
            
        Returns:
            Tuple of (action, confidence).
        """
        # Simple rule-based classification
        # TODO: Replace with trained classifier for better accuracy
        
        if motion_intensity < 0.01:
            return "standing", 0.9
        
        if velocity < 50:
            return "standing", 0.7
        elif velocity < 150:
            return "walking", 0.7
        elif velocity < 300:
            return "running", 0.7
        else:
            return "running", 0.8
        
        # If pose data available, use it for better classification
        if pose_data and pose_data.get("landmarks"):
            landmarks = pose_data["landmarks"]
            
            # Check vertical movement of key points
            # This is a simplified example
            if len(landmarks) >= 16:  # Ensure we have enough landmarks
                left_ankle = landmarks[27]
                right_ankle = landmarks[28]
                
                # If ankles are high, might be jumping
                avg_ankle_y = (left_ankle["y"] + right_ankle["y"]) / 2
                if avg_ankle_y < 0.7:  # Normalized coordinates
                    return "jumping", 0.6
        
        return "unknown", 0.5

    def detect(self, frame: np.ndarray) -> Dict[str, any]:
        """Perform complete motion and action detection.
        
        Args:
            frame: Input frame (BGR).
            
        Returns:
            Dictionary with all detection results.
        """
        self.frame_count += 1
        
        # Detect motion
        motion_data = self.detect_motion_simple(frame)
        
        # Calculate velocity
        prev_bbox = self.motion_history[-1]["bounding_box"] if self.motion_history else None
        curr_bbox = motion_data["bounding_box"]
        velocity = self.calculate_velocity(prev_bbox, curr_bbox)
        
        # Detect pose
        pose_data = self.detect_pose(frame)
        
        # Classify action
        action, confidence = self.classify_action(
            motion_data["motion_intensity"],
            velocity,
            pose_data
        )
        
        # Update history
        self.motion_history.append(motion_data)
        self.velocity_history.append(velocity)
        
        # Compile results
        result = {
            "frame_number": self.frame_count,
            "motion_detected": motion_data["motion_detected"],
            "motion_intensity": motion_data["motion_intensity"],
            "velocity": velocity,
            "bounding_box": motion_data["bounding_box"],
            "action": action,
            "action_confidence": confidence,
            "pose_landmarks": pose_data["landmarks"] if pose_data else None,
            "fg_mask": motion_data["fg_mask"],
        }
        
        return result

    def draw_motion(
        self,
        frame: np.ndarray,
        detection_result: Dict[str, any],
        show_mask: bool = False,
        show_pose: bool = True,
    ) -> np.ndarray:
        """Draw motion detection results on frame.
        
        Args:
            frame: Input frame.
            detection_result: Detection results from detect().
            show_mask: Show foreground mask overlay.
            show_pose: Show pose landmarks.
            
        Returns:
            Annotated frame.
        """
        output = frame.copy()
        
        # Draw bounding box
        bbox = detection_result.get("bounding_box")
        if bbox:
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw action label
            action = detection_result.get("action", "unknown")
            confidence = detection_result.get("action_confidence", 0.0)
            label = f"{action}: {confidence:.2f}"
            
            cv2.putText(
                output,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Draw motion info
        motion_intensity = detection_result.get("motion_intensity", 0.0)
        velocity = detection_result.get("velocity", 0.0)
        
        info_text = f"Intensity: {motion_intensity:.3f} | Velocity: {velocity:.1f} px/s"
        cv2.putText(
            output,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Show foreground mask
        if show_mask and "fg_mask" in detection_result:
            fg_mask = detection_result["fg_mask"]
            mask_overlay = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            mask_overlay = cv2.resize(mask_overlay, (200, 150))
            output[10:160, output.shape[1]-210:output.shape[1]-10] = mask_overlay
        
        # Draw pose landmarks
        if show_pose and detection_result.get("pose_landmarks") and self.use_pose_detection:
            self._draw_pose_landmarks(output, detection_result["pose_landmarks"])
        
        return output

    def _draw_pose_landmarks(self, frame: np.ndarray, landmarks: List[Dict]) -> None:
        """Draw pose landmarks on frame.
        
        Args:
            frame: Frame to draw on.
            landmarks: List of landmark dictionaries.
        """
        h, w = frame.shape[:2]
        
        # Draw landmarks as circles
        for landmark in landmarks:
            if landmark["visibility"] > 0.5:
                x = int(landmark["x"] * w)
                y = int(landmark["y"] * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        
        # Draw connections (simplified)
        connections = [
            (11, 12), (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),             # Right arm
            (11, 23), (12, 24),             # Torso
            (23, 24), (23, 25), (25, 27),   # Left leg
            (24, 26), (26, 28),             # Right leg
        ]
        
        for conn in connections:
            if conn[0] < len(landmarks) and conn[1] < len(landmarks):
                lm1 = landmarks[conn[0]]
                lm2 = landmarks[conn[1]]
                
                if lm1["visibility"] > 0.5 and lm2["visibility"] > 0.5:
                    x1 = int(lm1["x"] * w)
                    y1 = int(lm1["y"] * h)
                    x2 = int(lm2["x"] * w)
                    y2 = int(lm2["y"] * h)
                    
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    def reset(self):
        """Reset detector state."""
        self.motion_history.clear()
        self.velocity_history.clear()
        self.prev_frame = None
        self.frame_count = 0
        logger.info("Motion detector reset")