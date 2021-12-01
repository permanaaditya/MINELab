import os
import sys
import shutil
import cv2
import math
import time
import numpy as np
import argparse
import util.Util as Util
import pyaudio
import wave
import threading
import time
import subprocess
import pyrealsense2 as rs

class AudioRecorder():
    def __init__(self, audioPath):
        self.open = True
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 2
        self.format = pyaudio.paInt16
        self.audio_filename = audioPath
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []


    # Audio starts being recorded
    def record(self):
        self.stream.start_stream()
        while(self.open == True):
            data = self.stream.read(self.frames_per_buffer)
            self.audio_frames.append(data)
            if self.open==False:
                break


    # Finishes the audio recording therefore the thread too
    def stop(self):
        if self.open==True:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()

        pass

    # Launches the audio recording function using a thread
    def start(self):
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()





def start_AVrecording(filename):
    global video_thread
    global audio_thread

    video_thread = VideoRecorder()
    audio_thread = AudioRecorder()

    audio_thread.start()
    video_thread.start()
    return filename




def start_video_recording(filename):
    global video_thread
    video_thread = VideoRecorder()
    video_thread.start()
    return filename


def start_audio_recording(filename):
    global audio_thread
    audio_thread = AudioRecorder()
    audio_thread.start()

    return filename




def stop_AVrecording(filename):
    audio_thread.stop()
    frame_counts = video_thread.frame_counts
    elapsed_time = time.time() - video_thread.start_time
    recorded_fps = frame_counts / elapsed_time
    print("total frames " + str(frame_counts))
    print("elapsed time " + str(elapsed_time))
    print("recorded fps " + str(recorded_fps))
    video_thread.stop()

    # Makes sure the threads have finished
    while threading.active_count() > 1:
        time.sleep(1)

#    Merging audio and video signal
    if abs(recorded_fps - 6) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected
        print("Re-encoding")
        cmd = "ffmpeg -r " + str(recorded_fps) + " -i temp_video.avi -pix_fmt yuv420p -r 6 temp_video2.avi"
        subprocess.call(cmd, shell=True)

        print("Muxing")
        cmd = "ffmpeg -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video2.avi -pix_fmt yuv420p " + filename + ".avi"
        subprocess.call(cmd, shell=True)
    else:
        print("Normal recording\nMuxing")
        cmd = "ffmpeg -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video.avi -pix_fmt yuv420p " + filename + ".avi"
        subprocess.call(cmd, shell=True)
        print("..")


class VideoRecorderClass:

    def __init__(self, args):
        self.DATA_BASE_DIRECTORY = Util.DATA_BASE_DIRECTORY
        self.DATA_NEW_VIDEO_DIRECTORY = Util.DATA_NEW_VIDEO_DIRECTORY
        self.CAMERA = []
        self.RECORDING = False
        self.audio_thread = None
        if hasattr(args, 'dir') and args.dir is not None and args.dir != "":
            self.DATA_BASE_DIRECTORY = args.dir

        if hasattr(args, 'video') and args.video is not None and args.video != "":
            self.DATA_NEW_VIDEO_DIRECTORY = args.video

        for i in range(len(Util.CAMERA)) :
            if 'DEPTH' in Util.CAMERA[i] and Util.CAMERA[i]['DEPTH']:
                '''
                self.CAMERA.append(cv2.VideoCapture(Util.CAMERA[i]['ID'], cv2.CAP_DSHOW))
                self.CAMERA[i].set(cv2.CAP_PROP_FPS, Util.CAMERA[i]['FPS'])
                self.CAMERA[i].set(cv2.CAP_PROP_FRAME_WIDTH, Util.CAMERA[i]['WIDTH'])
                self.CAMERA[i].set(cv2.CAP_PROP_FRAME_HEIGHT, Util.CAMERA[i]['HEIGHT'])
                '''
                pipeline = rs.pipeline()
                config = rs.config()
                ctx = rs.context()
                devices = ctx.query_devices()

                config.enable_stream(rs.stream.color, Util.CAMERA[i]['WIDTH'], Util.CAMERA[i]['HEIGHT'], rs.format.rgb8, Util.CAMERA[i]['FPS'])
                config.enable_stream(rs.stream.depth, Util.CAMERA[i]['WIDTH'], Util.CAMERA[i]['HEIGHT'], rs.format.z16, Util.CAMERA[i]['FPS'])

                profile = pipeline.start(config)
                for device in devices:
                    if device.get_info(rs.camera_info.name).lower() in ["intel realsense d400", "intel realsense d435"]:
                        profile.get_device().query_sensors()[0].set_option(rs.option.emitter_enabled, 0)

                self.depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = self.depth_sensor.get_depth_scale()

                align_to = rs.stream.color
                self.align = rs.align(align_to)

                self.CAMERA.append(pipeline)

            else :
                self.CAMERA.append(cv2.VideoCapture(Util.CAMERA[i]['ID'], cv2.CAP_DSHOW))
                self.CAMERA[i].set(cv2.CAP_PROP_FPS, Util.CAMERA[i]['FPS'])
                self.CAMERA[i].set(cv2.CAP_PROP_FRAME_WIDTH, Util.CAMERA[i]['WIDTH'])
                self.CAMERA[i].set(cv2.CAP_PROP_FRAME_HEIGHT, Util.CAMERA[i]['HEIGHT'])

        if len(self.CAMERA) == 0:
            return

        self.build()
        self.run()

    def build(self):
        os.makedirs(self.DATA_BASE_DIRECTORY, exist_ok=True)
        os.makedirs(os.path.join(self.DATA_BASE_DIRECTORY, self.DATA_NEW_VIDEO_DIRECTORY), exist_ok=True)

        folder = os.path.join(self.DATA_BASE_DIRECTORY, self.DATA_NEW_VIDEO_DIRECTORY)

        if False :
            for filename in os.listdir(os.path.join(self.DATA_BASE_DIRECTORY, self.DATA_NEW_VIDEO_DIRECTORY)):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    def run(self):
            FPS = 0
            FPS_counter = 0
            start_time = time.time()

            videoOutObj = []
            videoFilePath = []

            DEPTH_ARRAY = []
            depthOutObj = []
            depthFilePath = []

            while True :
                FRAMES = []
                DEPTHS = []
                FRAME_RESULT = True

                FPS_counter +=1
                if (time.time() - start_time) > 1.0 :
                    FPS = math.ceil(FPS_counter / (time.time() - start_time))
                    FPS_counter = 0
                    start_time = time.time()

                for CAMERA_INDEX in range(len(self.CAMERA)) :
                    if type(self.CAMERA[CAMERA_INDEX]) == rs.pyrealsense2.pipeline :
                        try:
                            frames = self.CAMERA[CAMERA_INDEX].wait_for_frames()
                            aligned_frames = self.align.process(frames)
                            aligned_depth_frame = aligned_frames.get_depth_frame()

                            depthFrame = np.asanyarray(aligned_depth_frame.get_data())
                            frames = np.asanyarray(frames.get_color_frame().get_data())
                            frames = cv2.cvtColor(frames, cv2.COLOR_RGB2BGR)

                            FRAMES.append(frames)
                            DEPTHS.append(depthFrame)
                        except Exception as e:
                            print(e)
                            FRAME_RESULT = False
                            break

                    else :
                        res, frames = self.CAMERA[CAMERA_INDEX].read()

                        if not res:
                            FRAME_RESULT = False
                            break

                        FRAMES.append(frames)

                if not FRAME_RESULT:
                    print("Camera error!")
                    if len(videoOutObj) > 0 :
                        for INDEX in range(len(videoOutObj)) :
                            videoOutObj[INDEX].release()
                        videoOutObj = []

                    if len(videoOutObj) > 0 :
                        for INDEX in range(len(depthOutObj)) :
                            depthOutObj[INDEX].release()
                        depthOutObj = []
                    break


                for FRAME_INDEX in range(len(FRAMES)) :
                    if self.RECORDING and len(videoOutObj) > 0 :
                        videoOutObj[FRAME_INDEX].write(FRAMES[FRAME_INDEX])

                    cv2.putText(FRAMES[FRAME_INDEX],"FPS - %d" % (FPS), (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(FRAMES[FRAME_INDEX], "Record-R, Stop-C", (5, FRAMES[FRAME_INDEX].shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255 , 0, 0), 1)

                    if self.RECORDING :
                        cv2.putText(FRAMES[FRAME_INDEX], "Recording", (FRAMES[FRAME_INDEX].shape[1] - 85, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else :
                        cv2.putText(FRAMES[FRAME_INDEX], "Stopped", (FRAMES[FRAME_INDEX].shape[1] - 85, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    cv2.imshow("Camera {}".format(FRAME_INDEX + 1), FRAMES[FRAME_INDEX])


                for DEPTH_INDEX in range(len(DEPTHS)) :
                    if self.RECORDING and len(depthOutObj) > 0 :
                        DEPTH_ARRAY[DEPTH_INDEX].append(DEPTHS[DEPTH_INDEX].copy())

                    DEPTHS[DEPTH_INDEX] = cv2.applyColorMap(cv2.convertScaleAbs(DEPTHS[DEPTH_INDEX], alpha=0.03), cv2.COLORMAP_JET)

                    if self.RECORDING and len(depthOutObj) > 0 :
                        depthOutObj[DEPTH_INDEX].write(DEPTHS[DEPTH_INDEX])

                    cv2.putText(DEPTHS[DEPTH_INDEX],"FPS - %d" % (FPS), (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(DEPTHS[DEPTH_INDEX], "Record-R, Stop-C", (5, DEPTHS[DEPTH_INDEX].shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255 , 0, 0), 1)

                    if self.RECORDING :
                        cv2.putText(DEPTHS[DEPTH_INDEX], "Recording", (DEPTHS[DEPTH_INDEX].shape[1] - 85, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else :
                        cv2.putText(DEPTHS[DEPTH_INDEX], "Stopped", (DEPTHS[DEPTH_INDEX].shape[1] - 85, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    cv2.imshow("Depth camera {}".format(DEPTH_INDEX + 1), DEPTHS[DEPTH_INDEX])


                key = cv2.waitKey(1)
                if key == 27 :
                    if self.RECORDING is False :
                        break
                elif key == 114 :
                    if self.RECORDING is not True and len(videoOutObj) == 0 :
                        self.RECORDING = True
                        audioPath = os.path.join(self.DATA_BASE_DIRECTORY, self.DATA_NEW_VIDEO_DIRECTORY, "{}_audio.wav".format(start_time))
                        self.audio_thread = AudioRecorder(audioPath)
                        self.audio_thread.start()

                        for CAMERA_INDEX in range(len(self.CAMERA)) :
                            videoFilePath.append(os.path.join(self.DATA_BASE_DIRECTORY, self.DATA_NEW_VIDEO_DIRECTORY, "{}_cam_{}.mp4".format(start_time, CAMERA_INDEX + 1)))
                            if type(self.CAMERA[CAMERA_INDEX]) == rs.pyrealsense2.pipeline :
                                videoOutObj.append(cv2.VideoWriter(videoFilePath[CAMERA_INDEX], cv2.VideoWriter_fourcc(*'MP4V'), Util.CAMERA[CAMERA_INDEX]['FPS'], (Util.CAMERA[CAMERA_INDEX]['WIDTH'], Util.CAMERA[CAMERA_INDEX]['HEIGHT'])))
                            else :
                                videoOutObj.append(cv2.VideoWriter(videoFilePath[CAMERA_INDEX], cv2.VideoWriter_fourcc(*'MP4V'), self.CAMERA[CAMERA_INDEX].get(cv2.CAP_PROP_FPS), (int(self.CAMERA[CAMERA_INDEX].get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.CAMERA[CAMERA_INDEX].get(cv2.CAP_PROP_FRAME_HEIGHT)))))

                        for CAMERA_INDEX in range(len(self.CAMERA)) :
                            if type(self.CAMERA[CAMERA_INDEX]) == rs.pyrealsense2.pipeline :
                                DEPTH_ARRAY.append([])

                                depth_path = os.path.join(self.DATA_BASE_DIRECTORY, self.DATA_NEW_VIDEO_DIRECTORY, "{}_depth_{}.mp4".format(start_time, CAMERA_INDEX + 1))
                                depthFilePath.append(depth_path)
                                depthOutObj.append(cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'MP4V'), Util.CAMERA[CAMERA_INDEX]['FPS'], (Util.CAMERA[CAMERA_INDEX]['WIDTH'], Util.CAMERA[CAMERA_INDEX]['HEIGHT'])))

                elif key == 99 :
                    if self.RECORDING and len(videoOutObj) > 0 :
                        self.RECORDING = False
                        self.audio_thread.stop()

                        for INDEX in range(len(videoOutObj)) :
                            videoOutObj[INDEX].release()
                            #cmd = "C:/ffmpeg-20200603-b6d7c4c-win64-static/bin/ffmpeg -ac 2 -channel_layout stereo -i {} -i {} -pix_fmt yuv420p {}_mixed.mp4".format(self.audio_thread.audio_filename, videoFilePath[CAMERA_INDEX], videoFilePath[CAMERA_INDEX])
                            #print(cmd)
                            #subprocess.call(cmd, shell=True)

                        for INDEX in range(len(depthOutObj)) :
                            depthOutObj[INDEX].release()
                            np.savez(depthFilePath[INDEX], DEPTH_ARRAY[INDEX])

                        videoOutObj = []
                        videoFilePath = []

                        DEPTH_ARRAY = []
                        depthOutObj = []
                        depthFilePath = []

            for CAMERA_INDEX in range(len(self.CAMERA)) :
                if type(self.CAMERA[CAMERA_INDEX]) == rs.pyrealsense2.pipeline :
                    self.CAMERA[CAMERA_INDEX].stop()
                else :
                    self.CAMERA[CAMERA_INDEX].release()

            if len(videoOutObj) > 0 :
                for INDEX in range(len(videoOutObj)) :
                    videoOutObj[INDEX].release()

            if len(videoOutObj) > 0 :
                for INDEX in range(len(depthOutObj)) :
                    depthOutObj[INDEX].release()

            if self.RECORDING :
                self.audio_thread.stop()

            cv2.destroyAllWindows()

if __name__ == '__main__':
    print("# Recording tool is starting...\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", default="data", type=str, help="Data folder")
    parser.add_argument("--video", "-v", default="new_video", type=str, help="New video directory")
    args = parser.parse_args()

    VideoRecorderClass(args)

    print("\n# Recording tool is finished.")
