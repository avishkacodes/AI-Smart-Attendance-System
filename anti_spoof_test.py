import cv2
import numpy as np

from anti_spoofing.src.anti_spoof_predict import AntiSpoofPredict
from anti_spoofing.src.generate_patches import CropImage

model = AntiSpoofPredict(0)
image_cropper = CropImage()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    try:
        bbox = model.get_bbox(frame)

        x, y, w, h = bbox

        face = image_cropper.crop(
            org_img=frame,
            bbox=bbox,
            scale=2.7,
            out_w=80,
            out_h=80
        )

        result1 = model.predict(
            face,
            "anti_spoofing/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
        )

        face2 = image_cropper.crop(
            org_img=frame,
            bbox=bbox,
            scale=4.0,
            out_w=80,
            out_h=80
        )

        result2 = model.predict(
            face2,
            "anti_spoofing/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth"
        )

        prediction = result1 + result2

        label = np.argmax(prediction)
        confidence = np.max(prediction)

        print("Confidence:", confidence)

        if label == 1 and confidence > 0.90:
            text = "REAL FACE"
            color = (0, 255, 0)
        else:
            text = "FAKE FACE"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # cv2.putText(
        #     frame,
        #     f"{confidence:.2f}",
        #     (x, y + h + 25),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.6,
        #     color,
        #     2
        # )

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    except Exception as e:
        print("ERROR:", e)

    cv2.imshow("Anti Spoof Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()