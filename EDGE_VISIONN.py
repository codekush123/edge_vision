import cv2
import mediapipe as mp
import numpy as np
import time
import requests



def set_desk_height(position: int):
    hass_url = "http://fake-homeassistant.local:8123/api/services/cover/set_cover_position"
    token = "FAKE_LONG_LIVED_TOKEN"
    entity_id = "cover.fake_desk"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    data = {"entity_id": entity_id, "position": position}
    print("\n[Desk Height API CALL]")
    print(f"POST {hass_url}")
    print("Headers:", headers)
    print("Json:", data)
    print(f"→ Would move desk to {position}% height\n")
    # requests.post(hass_url, json=data, headers=headers)



def set_smart_bulb_brightness(brightness: int):
    hass_url = "http://fake-homeassistant.local:8123/api/services/light/turn_on"
    token = "FAKE_LONG_LIVED_TOKEN"
    entity_id = "light.fake_smart_bulb"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {"entity_id": entity_id, "brightness_pct": brightness}
    print("\n[Smart Bulb API CALL]")
    print(f"POST {hass_url}")
    print("Headers:", headers)
    print("Json:", data)
    print(f"→ Would set bulb brightness to {brightness}%\n")
    # requests.post(hass_url, json=data, headers=headers)



mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)



def calculate_angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle



def analyze_posture(landmarks):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    

    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    

    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    

    shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])

    shoulder_mid = np.mean([left_shoulder, right_shoulder], axis=0)

    hip_mid = np.mean([left_hip, right_hip], axis=0)

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
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    last_status = None
    last_update_time = 0
    last_brightness = None
    last_brightness_update = 0
    last_suggestion = ""
    adjustment_cooldown = 10
    brightness_cooldown = 10



    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image_rgb.flags.writeable = False

            results = pose.process(image_rgb)

            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            suggestion = ""


            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
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
                    suggestion, desired_position = "Bad posture: Try standing up.", 100
                elif posture_status == "Uneven Shoulders":
                    suggestion, desired_position = "Keep your shoulders level.", 0
                else:
                    suggestion, desired_position = "Good posture. Keep it up!", 0
                if posture_status != last_status or (current_time - last_update_time > adjustment_cooldown):
                    set_desk_height(desired_position)
                    last_status = posture_status
                    last_update_time = current_time
                    last_suggestion = suggestion
            else:
                suggestion = "Cannot detect full body."
            face_results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


            brightness_suggestion = ""
            current_time = time.time()


            if face_results.detections:
                detection = face_results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x_min, y_min = max(int(bbox.xmin * w), 0), max(int(bbox.ymin * h), 0)
                box_width, box_height = int(bbox.width * w), int(bbox.height * h)
                face_roi = frame[y_min:y_min + box_height, x_min:x_min + box_width]
                if face_roi.size > 0:
                    brightness = np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
                    brightness_threshold_low, brightness_threshold_high = 80, 180
                    if brightness < brightness_threshold_low:
                        bulb_brightness = 100
                        brightness_suggestion = "Face is dark, increasing light brightness."
                    elif brightness > brightness_threshold_high:
                        bulb_brightness = 20
                        brightness_suggestion = "Face is too bright, decreasing light brightness."
                    else:
                        bulb_brightness = 60
                        brightness_suggestion = "Face brightness is normal."
                    if bulb_brightness != last_brightness or (current_time - last_brightness_update > brightness_cooldown):
                        set_smart_bulb_brightness(bulb_brightness)
                        last_brightness = bulb_brightness
                        last_brightness_update = current_time
            else:
                brightness_suggestion = "No face detected to adjust brightness."
            cv2.putText(image, f"Suggestion: {suggestion or last_suggestion}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, brightness_suggestion, (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('Posture & Face Brightness (Demo)', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        face_detection.close()


if __name__ == "__main__":
    main()