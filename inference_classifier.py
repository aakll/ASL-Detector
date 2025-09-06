import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 8: 'I', 11: 'L', 27: '_', 28: '<DEL>', 
    29: 'Hi ', 30: 'My ', 31: 'Name ', 32: 'Your ', 33: 'What '
}

current_prediction = None
prediction_count = 0
stable_threshold = 15  # frames required for confirmation
current_text = ""
display_text = ""
display_translation = ""
translation_duration = 3  # seconds to display English
translation_time = None

def translate_sentence(sl_sentence):
    # Simple keyword-based translation
    sl_sentence = sl_sentence.replace('_', '').strip()
    words = sl_sentence.split()
    english = []

    if 'Hi' in words:
        english.append('Hi,')
    if 'My' in words and 'Name' in words:
        name_index = words.index('Name') + 1
        if name_index < len(words):
            english.append(f'my name is {words[name_index]}')
        else:
            english.append('my name is ...')
    if 'Your' in words and 'Name' in words and 'What' in words:
        english.append('What is your name?')
    return ' '.join(english)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1) # Flip the frame horizontally
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            if predicted_character == current_prediction:
                prediction_count += 1
            else:
                current_prediction = predicted_character
                prediction_count = 1

            if prediction_count == stable_threshold:
                if predicted_character == '<DEL>':
                    current_text = current_text[:-1]
                else:
                    current_text += predicted_character
                prediction_count = 0

                if predicted_character == '_':
                    display_text = current_text
                    display_translation = translate_sentence(current_text)
                    translation_time = time.time()
                    current_text = ""  # Reset

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)

    # Current typing line
    cv2.putText(frame, current_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 0), 3, cv2.LINE_AA)

    # Show SL sentence + English directly below it
    if display_text:
        cv2.putText(frame, display_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"English: {display_translation}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 3, cv2.LINE_AA)

        if time.time() - translation_time > translation_duration:
            display_text = ""
            display_translation = ""
            translation_time = None

   
    cv2.imshow('Sign Language Translator', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
