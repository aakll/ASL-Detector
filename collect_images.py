import cv2
import os

# Change this to the letter/gesture you want to capture
label = "33"
data_dir = "data"
label_dir = os.path.join(data_dir, label)
os.makedirs(label_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0   # image counter
frame_count = 0   # to control skipping

print("Press 'q' to stop capturing")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture - " + label, frame)

    # Save only 1 out of every 5 frames
    if frame_count % 5 == 0:
        img_path = os.path.join(label_dir, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

    frame_count += 1

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} images to {label_dir}")
