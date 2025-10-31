import cv2
import mediapipe as mp
import numpy as np
import time
import requests  

def set_desk_height(position):
    hass_url = "http://fake-homeassistant.local:8123/api/services/cover/set_cover_position"
    token = "FAKE_LONG_LIVED_TOKEN"
    entity_id = "cover.fake_desk"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {
        "entity_id": entity_id,
        "position": position
    }
    print("\n[FAKE API CALL]")
    print("POST", hass_url)
    print("Headers:", headers)
    print("Json:", data)
    print(f"→ Would send command to move desk to {position}%\n")
    # If real: requests.post(hass_url, json=data, headers=headers)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def analyze_posture(landmarks):
    left_shoulder  = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_hip  = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
    shoulder_mid = [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2]
    hip_mid = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
    spine_angle = calculate_angle(left_shoulder, shoulder_mid, hip_mid)
    if shoulder_diff > 0.05:
        posture_status = "Uneven Shoulders"
    elif spine_angle < 160 or spine_angle > 200:
        posture_status = "Slouching"
    else:
        posture_status = "Good"
    return posture_status, shoulder_diff, spine_angle

def main():
    cap = cv2.VideoCapture(0)
    last_status = None
    last_update_time = 0
    adjustment_cooldown = 10
    last_suggestion = ""
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            suggestion = ""
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                )
                posture_status, shoulder_diff, spine_angle = analyze_posture(results.pose_landmarks.landmark)
                cv2.putText(image, f"Posture: {posture_status}", (10, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"Shoulder Diff: {shoulder_diff:.3f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Spine Angle: {spine_angle:.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                current_time = time.time()
                if posture_status == "Slouching":
                    suggestion = "Bad posture: Try standing up."
                    desired_position = 100
                elif posture_status == "Uneven Shoulders":
                    suggestion = "Keep your shoulders level."
                    desired_position = 0
                else:
                    suggestion = "Good posture. Keep it up!"
                    desired_position = 0
                if (posture_status != last_status or current_time - last_update_time > adjustment_cooldown):
                    set_desk_height(desired_position)
                    last_update_time = current_time
                    last_status = posture_status
                    last_suggestion = suggestion
            else:
                suggestion = "Cannot detect full body."
            cv2.putText(image, f"Suggestion: {suggestion or last_suggestion}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow('Posture Detection (API Sim Demo)', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
if __name__ == "__main__":
    main()
