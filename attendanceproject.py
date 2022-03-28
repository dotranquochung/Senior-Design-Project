##Libraries
#Library for attendance project
import cv2
from cv2 import FONT_HERSHEY_COMPLEX
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date
#Library for speech to text
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence

path = 'images'
images = []
classNames = []
myList = os.listdir(path)

#print for testing
#print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

#print for testing
print('----------------------------------------------------------------------------')
print('List contains: ')
print(classNames)

#------------Encoding images--------------#
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#------------Attendance checking--------------#
def markAttendance(name):
    #Open Attendance.csv (like Excel) with read and write
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            
        #If the name is not the list, then mark
        if name not in nameList:
            now = datetime.now()
            today = date.today()
            dtString = now.strftime('%H:%M:%S')
            dtString_date = today.strftime("%d/%m/%Y")
            f.writelines(f'\n{name},{dtString},{dtString_date}')

#Create a list of encoding images
encodeListKnown = findEncodings(images)

#print for testing: Encoding complete
print('----------------------------------------------------------------------------')
print('Encoding Complete')

cap = cv2.VideoCapture(0)   #turn on webcam

while True:
    success, img = cap.read()
    #Resize the current image to smaller image to speed up the encoding process
    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    facesCurFrame = face_recognition.face_locations(imgSmall)#Find face in current frame
    encodesCurFrame = face_recognition.face_encodings(imgSmall,facesCurFrame)#Encode face in current frame
        
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        #Based on encoded img in current frame, find in the encoded list
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            
        #These print lines are created for debugging and improving the method
        #Comment out to see the matching index and matching distance
        #print(faceDis)
        faceDisMinimum = np.min(faceDis) #The min_face_distance (closest face)
        #print(faceDisMinimum)
        matchIndex = np.argmin(faceDis) #MATCH: when face dis is the minimum

        if (matches[matchIndex] and faceDisMinimum <0.4):#if match, print the name
            name = classNames[matchIndex].upper()
             #print(name) #print out detected person  
            y1,x2,y2,x1 = faceLoc #face location, draw a rectangle around the face
            cv2.rectangle(img,(x1,y1),(x2,y2),(3,252,240),2)    #draw rectangle
            cv2.putText(img,name,(x1-6,y2-6),FONT_HERSHEY_COMPLEX,1,(255,255,255),2)#write name
            markAttendance(name)
        else:
            y1,x2,y2,x1 = faceLoc #face location, draw a rectangle around the face
            name = 'Unknown'
            cv2.rectangle(img,(x1,y1),(x2,y2),(3,252,240),2)    #draw rectangle
            cv2.putText(img,name,(x1-6,y2-6),FONT_HERSHEY_COMPLEX,1,(255,255,255),2)#write name
            
    cv2.imshow('Webcam', img)
    cv2.waitKey(1);
    k = cv2.waitKey(1) & 0xFF
    # press 'q' to exit
    if k == ord('q'):
        break
        
 #       c = cv2.waitKey(7) % 0x100 #Press ESC or Enter to exit
 #       if c == 27:
 #           break
    
cap.release()
cv2.destroyAllWindows()

#Read file excel: people has been checked in 
print('--------------------------------------Attendance.csv--------------------------------------')
print('People has checkin and time: ')
f = open("Attendance.csv", "r")
print(f.read())

#Intention adding features:
# if date is different than current date in Attendance sheet
#   -> create a new sheet (Attendance (date) )
# 
# Comment: if the person is not found -> "Unknown"
# (done)

#feature:
#encoding multiple persons on window screen
# (done)

#voice -> transcript (movie)
#loading


###-----------------------------------Speech to text function------------------------------------
# create a speech recognition object
r = sr.Recognizer()
path = 'audio/sample1.wav'

# a function that splits the audio file into chunks
# and applies speech recognition
def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text
    
#print("\nFull text of the audio file: \n", get_large_audio_transcription(path))\

#Adding feature:
#Person speak on webcam -> output text on webcam