import cv2
import mediapipe as mp

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Detect both hands
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and get the hand landmarks
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_positions = []
        gesture_detected = False  # Flag to ensure only one gesture is displayed at a time

        # Iterate through each hand detected
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the wrist position of the hand
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            hand_positions.append(wrist)

            # Get image dimensions for denormalization
            img_height, img_width, _ = img.shape

            # Visualize wrist positions (for debugging)
            wrist_x, wrist_y = int(wrist.x * img_width), int(wrist.y * img_height)
            cv2.circle(img, (wrist_x, wrist_y), 10, (255, 0, 0), -1)  # Draw wrist circle for debugging

        # Detect the "Shenka" gesture (both hands raised near the face)
        if len(hand_positions) == 2:  # Check if two hands are detected
            wrist_1_y = hand_positions[0].y
            wrist_2_y = hand_positions[1].y

            # Debugging: Print wrist positions to the console
            print(f"Wrist 1 Y: {wrist_1_y}, Wrist 2 Y: {wrist_2_y}")

            # Check if both hands are above a threshold (0.6 in Y-axis for flexibility)
            if wrist_1_y < 0.6 and wrist_2_y < 0.6:  
                cv2.putText(img, 'Shenka Gesture Detected', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                gesture_detected = True  # Set the flag to True once the Shenka gesture is detected

        # Detect other gestures if "Shenka" was not detected
        if not gesture_detected:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the coordinates for the thumb tip and index finger tip
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Get image dimensions for denormalization
                thumb_tip_x, thumb_tip_y = int(thumb_tip.x * img_width), int(thumb_tip.y * img_height)
                index_tip_x, index_tip_y = int(index_tip.x * img_width), int(index_tip.y * img_height)

                # Visual Debugging: Draw circles on the thumb and index finger tips
                cv2.circle(img, (thumb_tip_x, thumb_tip_y), 8, (0, 255, 0), -1)
                cv2.circle(img, (index_tip_x, index_tip_y), 8, (0, 0, 255), -1)

                # Detect "OK" gesture (thumb and index finger tips are close)
                distance = ((thumb_tip_x - index_tip_x) ** 2 + (thumb_tip_y - index_tip_y) ** 2) ** 0.5
                if distance < 30:  # OK gesture: Adjust threshold as needed
                    cv2.putText(img, 'OK Gesture Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    gesture_detected = True  # Set the flag to True once a gesture is detected
                    break  # Exit the loop after detecting the gesture

                # Detect thumbs-up/thumbs-down gestures based on the y-coordinate
                if thumb_tip_y < index_tip_y:  # Thumb is above the index finger (Thumbs up)
                    cv2.putText(img, "Thumbs Up Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    gesture_detected = True  # Set the flag to True
                    break  # Exit the loop after detecting the gesture

                elif thumb_tip_y > index_tip_y:  # Thumb is below the index finger (Thumbs Down)
                    # Adjust the threshold for detecting thumbs down
                    thumb_angle = thumb_tip_y - index_tip_y  # Ensure thumb is lower
                    thumb_position = thumb_tip_x - index_tip_x  # Check relative x position

                    # Relax the constraints for thumb position and angle
                    if thumb_angle > 15:  # Reduced from 20
                        cv2.putText(img, "Thumbs Down Detected", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        gesture_detected = True  # Set the flag to True
                        break  # Exit the loop after detecting the gesture

    # Display the image with the annotations
    cv2.imshow('Hand Gesture Recognition', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
