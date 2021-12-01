# importing libraries
import os
import cv2
from PIL import Image

# Checking the current directory path
# print(os.getcwd())

# Folder which contains all the images
# from which video is to be generated
# os.chdir("C:\\Python\\Geekfolder2")
# path = "C:\\Python\\Geekfolder2"

def getImages(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            if filename.find('erhuTop') != -1:
                imagesPath.append(os.path.join(folder, filename))
                images.append(img)
                # print(folder+'/'+filename)
        else:
            getImages(os.path.join(folder, filename))
    return images, imagesPath

images = []
imagesPath = []
image_folder = 'C:/Users/LEGION/PycharmProjects/pythonProject/InstrumentProject/data/erhu'
# video_name = 'topErhu2.avi'
images, imagesPath = getImages(image_folder)
mean_height = 0
mean_width = 0
num_of_images = len(images)
print(num_of_images)

for image in images:
    # im = Image.open(os.path.join(path, file))
    # print(image.shape)
    height, width, _ = image.shape
    mean_width += width
    mean_height += height
# im.show() # uncomment this for displaying the image

# Finding the mean height and width of all images.
# This is required because the video frame needs
# to be set with same width and height. Otherwise
# images not equal to that width height will not get
# embedded into the video
mean_width = int(mean_width / num_of_images)
mean_height = int(mean_height / num_of_images)
# print(mean_width, mean_width)
# print(mean_height)
# print(mean_width)

# Resizing of the images to give
# them same width and height
for (image, imagePath) in zip(images, imagesPath):
    # if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
    # opening image using PIL Image
    im = Image.open(imagePath)

    # im.size includes the height and width of image
    # height, width, _ = image.shape
    width, height = im.size
    # print(width, height)

    # resizing
    # print(mean_height, mean_width)
    # print(image, imagePath)
    imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)
    # print(imagePath)
    imResize.save(imagePath, 'JPEG', quality=95)  # setting quality
    # printing each resized image name
    # print(imagePath, " is resized")


# Video Generating function
def generate_video(images, imagesPath):
    # image_folder = '.'  # make sure to use your folder
    video_name = 'topErhu.avi'
    # os.chdir("C:\\Python\\Geekfolder2")

    # images = [img for img in os.listdir(image_folder)
    #           if img.endswith(".jpg") or
    #           img.endswith(".jpeg") or
    #           img.endswith("png")]

    # Array images should only consider
    # the image files ignoring others if any
    # print(images)

    frame = images[0]

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 15, (width, height))
    counter = 0
    # Appending the images to the video one by one
    for (image, imagePath) in zip(images, imagesPath):
        print(imagePath)
        counter+=1
        video.write(image)
    # Deallocating memories taken for window creation
    print('Image Counter : '+str(counter))
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated


# Calling the generate_video function
generate_video(images, imagesPath)
