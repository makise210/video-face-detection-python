import dlib
import cv2
import imutils
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
# 開啟影片檔案 open file
cap = cv2.VideoCapture(file_path)
# 取得畫面尺寸
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 使用 XVID 編碼 use XVID
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# build VideoWriter 
out = cv2.VideoWriter('output.avi', fourcc, 25.0, (width, height))

# Dlib face detection
detector = dlib.get_frontal_face_detector()


while(cap.isOpened()):
  ret, frame = cap.read()

  #detect face 
  face_rects, scores, idx = detector.run(frame, 0)

  # get the result of detection
  for i, d in enumerate(face_rects):
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    text = "%2.2f(%d)" % (scores[i], idx[i])

    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

   
    cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
            0.7, (255, 255, 255), 1, cv2.LINE_AA)


  out.write(frame)

  # show the result
  cv2.imshow("Face Detection", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
out.release()
cv2.destroyAllWindows()