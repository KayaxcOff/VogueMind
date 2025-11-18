import cv2
import os
import dotenv
import numpy as np
from tensorflow.keras import models

dotenv.load_dotenv()

model_path = os.getenv('MODEL_PATH')

class_names = [
    'Blazer', 'Clana_Panjang', 'Clana_Pendek',
    'Gaun', 'Hoodie', 'Jaket',
    'Jaket_Denim', 'jaket_Olahraga', 'Jeans',
    'Kaos', 'Kemeja', 'Mantel',
    'Polo', 'Rok', 'Sweter'
]

model = models.load_model(model_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]

    cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Clothes Recognition', frame)

cap.release()
cv2.destroyAllWindows()