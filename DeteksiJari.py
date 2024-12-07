import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands  #Modul hands dari MediaPipe.
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) 
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

#inisialisasi titik titik jari
def count_raised_fingers(hand_landmarks, handedness):
    finger_tips = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]
    finger_count = 0

    # Cek jempol
    if handedness.classification[0].label == "Right":
        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
            finger_count += 1
    else:  # Left hand
        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
            finger_count += 1

    # Cek jari lainnya
    for tip in finger_tips[1:]:  # Mulai dari telunjuk
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_count += 1

    return finger_count

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Gagal membaca frame dari webcam.")
        continue

    # Flip gambar horizontal untuk tampilan mirror
    image = cv2.flip(image, 1)

    # Konversi gambar ke RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Proses deteksi tangan
    results = hands.process(image_rgb)

    total_fingers = 0
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Gambar landmark tangan
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Hitung jumlah jari yang terangkat
            finger_count = count_raised_fingers(hand_landmarks, handedness)
            total_fingers += finger_count
            
            # Tampilkan jumlah jari untuk setiap tangan
            hand_label = "Kanan" if handedness.classification[0].label == "Right" else "Kiri"
            cv2.putText(image, f'Tangan {hand_label}: {finger_count}', 
                        (10, 30 if hand_label == "Kanan" else 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan total jari
    cv2.putText(image, f'Total Jari: {total_fingers}', (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Tampilkan hasil
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Tekan 'ESC' untuk keluar
        break

cap.release()
cv2.destroyAllWindows()