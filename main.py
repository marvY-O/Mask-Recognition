import numpy as np
import cv2
import os

def credits():
    print("Created by: Vyom Verma\n"
          "Thank You for using the application!")

os.system("clear")
print("Welcome to Mask Recognition Application")

# load model

print("Loading Model")
names = {0: "Mask", 1: "No Mask"}

with_mask = np.load('Mask.npy')
without_mask = np.load('NoMask.npy')

with_mask = with_mask.reshape(1250, 50 * 50 * 3)
without_mask = without_mask.reshape(1250, 50 * 50 * 3)

X = np.r_[with_mask, without_mask]

labels = np.zeros(X.shape[0])
labels[1250:] = 1.0

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

svm = SVC()

svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)
acc = accuracy_score(y_test, y_pred)

haar_data = cv2.CascadeClassifier('frontalRecognition.xml')

# Loading completed
while True:
    os.system("clear")
    print("Welcome to Mask Recognition Application")
    print("Model Accuracy: ", acc * 100, "%")
    print("1. Webcam")
    print("2. Video")
    print("3. Picture")
    print("4. Exit")

    choice = input("\n>>")
    try:
        int(choice)
    except ValueError:
        continue
    if int(choice) == 1:
        os.system("clear")
        capture = cv2.VideoCapture(0)
        print("Press Esc to exit")
        while True:
            flag, img = capture.read()
            if flag:
                faces = haar_data.detectMultiScale(img)
                for x, y, w, h in faces:

                    face = img[y:y + h, x:x + w, :]
                    face = cv2.resize(face, (50, 50))
                    face = face.reshape(1, -1)
                    # face = pca.transform(face)
                    pred = svm.predict(face)
                    n = names[int(pred)]
                    # print(n)

                    if (n == "Mask"):
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Draw a label with a name below the face
                        cv2.rectangle(img, (x - 1, y + h), (x + w + 1, y + h + 35), (0, 255, 0), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img, n, (x, y + h + 20), font, 1.0, (0, 0, 0), 1)
                    else:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        # Draw a label with a name below the face
                        cv2.rectangle(img, (x - 1, y + h), (x + w + 1, y + h + 35), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(img, n, (x, y + h + 20), font, 1.0, (0, 0, 0), 1)

                cv2.imshow("result", img)
                if cv2.waitKey(2) == 27:
                    break

        capture.release()
        cv2.destroyAllWindows()
        credits()

    elif int(choice) == 2:
        os.system("clear")
        pathVid = input("Enter path of the video: ")
        if (os.path.exists(pathVid)==False):
            print("No such file exists!")
        else:
            credits()
            exit()
        capture = cv2.VideoCapture(pathVid)
        print("Press Esc to exit")
        while True:
            flag, img = capture.read()
            if flag:
                faces = haar_data.detectMultiScale(img)
                for x, y, w, h in faces:

                    face = img[y:y + h, x:x + w, :]
                    face = cv2.resize(face, (50, 50))
                    face = face.reshape(1, -1)
                    # face = pca.transform(face)
                    pred = svm.predict(face)
                    n = names[int(pred)]
                    # print(n)

                    if (n == "Mask"):
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Draw a label with a name below the face
                        cv2.rectangle(img, (x - 1, y + h), (x + w + 1, y + h + 35), (0, 255, 0), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img, n, (x, y + h + 20), font, 1.0, (0, 0, 0), 1)
                    else:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        # Draw a label with a name below the face
                        cv2.rectangle(img, (x - 1, y + h), (x + w + 1, y + h + 35), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(img, n, (x, y + h + 20), font, 1.0, (0, 0, 0), 1)

                cv2.imshow("result", img)
                if cv2.waitKey(2) == 27:
                    break

        capture.release()
        cv2.destroyAllWindows()

    elif int(choice) == 3:
        os.system("clear")
        pathImg = input("Enter path of the image: ")
        if ( os.path.exists(pathImg) == False ):
            print("No such file exists!")
        else:
            credits()
            exit()
        print("Press Esc to exit")

        img = cv2.imread(pathImg)
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            face = img[y:y + h, x:x + w, :]
            face = cv2.resize(face, (50, 50))
            face = face.reshape(1, -1)
            # face = pca.transform(face)
            pred = svm.predict(face)
            n = names[int(pred)]
            # print(n)

            if (n == "Mask"):
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(img, (x - 1, y + h), (x + w + 1, y + h + 35), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, n, (x, y + h + 20), font, 0.5, (0, 0, 0), 1)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(img, (x - 1, y + h), (x + w + 1, y + h + 35), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, n, (x, y + h + 20), font, 0.5, (0, 0, 0), 1)

        while True:
            cv2.imshow("result", img)
            if cv2.waitKey(2) == 27:
                break
    elif int(choice) == 4:
        credits()
        exit()
    else:
        print("Invalid Option!")