import cv2
import os

def getImages(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            if filename.find('right_hand') != -1:
                imagesPath.append(os.path.join(folder, filename))
                images.append(img)
                print(folder+'/'+filename)
        else:
            getImages(os.path.join(folder, filename))
    return images, imagesPath

images = []
imagesPath = []
image_folder = 'C:/Users/LEGION/PycharmProjects/pythonProject/InstrumentProject/data/erhu'
video_name = 'rightHand2.avi'
images, imagesPath = getImages(image_folder)
print(len(images))
frame = images[0]
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, 0, 15, (width, height))
imageTotal = len(images)

for image in images:
    # imageTotal-=1
    # mess = str(imageTotal)+' images remaining....'
    # print(mess)
    video.write(image)

print('Sucessfull created video from '+str(len(images))+' images!')

cv2.destroyAllWindows()
video.release()