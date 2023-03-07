import cv2
import numpy as np
from tensorflow.keras.models import load_model


cv2.namedWindow("PaintList")

img = np.zeros((512,512), dtype="uint8")

draw_permission = False

def handle_click(event, x, y, flags, params):
    global draw_permission
    if event == cv2.EVENT_MOUSEMOVE:
        if draw_permission:
            cv2.circle(img,(x,y),15,255,-1)
    elif event == cv2.EVENT_LBUTTONDOWN:
        draw_permission = True
    elif event == cv2.EVENT_LBUTTONUP:
        draw_permission = False

cv2.setMouseCallback("PaintList",handle_click)

while True:
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    if key == ord('r'):
        model = load_model("model.h5")
        gaus = cv2.GaussianBlur(img, (5, 5), 0)
        gaus_resized = cv2.resize(gaus/255, (28,28))
        gaus_resized = gaus_resized.reshape(1,28,28,1)
        predictions = model.predict(gaus_resized)
        print(f"Number on paintList: {int(np.argmax(predictions,1))}")
    if key == ord("c"):
        img[:] = 0
        
    cv2.imshow("PaintList", img)

cv2.destroyAllWindows()