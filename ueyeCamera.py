from pyueye import ueye
import numpy as np
import sys
from ctypes import c_wchar_p
from time import time_ns



class Camera(object):

    def __init__(self):
        # Variables
        self.hCam = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID
        self.sInfo = ueye.SENSORINFO()
        self.cInfo = ueye.CAMINFO()
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        self.rectAOI = ueye.IS_RECT()
        self.pitch = ueye.INT()
        self.nBitsPerPixel = ueye.INT(8)  # 24: bits per pixel for color mode; take 8 bits per pixel for monochrome
        self.channels = 1  # 3 channels for color mode (RGB); take 1 channel for monochrome
        self.m_nColorMode = ueye.INT()  # Y8/RGB16/RGB24/REG32
        self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
        self.image_file_params = ueye.IMAGE_FILE_PARAMS()


    def configure(self):#TODO
        self.connect()
        self.get_data()
        self.sensor_info()

        self.reset_camera()
        self.set_display_to_DIB()
        self.set_color_mode()#TODO which one
        self.set_camera_settings()
        # self.set_full_auto()#TODO change camera settings


    def connect(self):
        # Starts the driver and establishes the connection to the camera
        print("Connecting to camera")
        nRet = ueye.is_InitCamera(self.hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")
        else:
            print("Camera initialised")

    def get_data(self):
        # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure
        # that cInfo points to
        nRet = ueye.is_GetCameraInfo(self.hCam, self.cInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")
        # else:
        #     print("GetCameraInfo complete")
            # print(self.cInfo)  # self.cInfo contains a lot of interesting device information
            # print()

    def sensor_info(self):
        # You can query additional information about the sensor type used in the camera
        nRet = ueye.is_GetSensorInfo(self.hCam, self.sInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")
        else:
            # print("Get Sensor info complete")
            # print(self.sInfo)
            print("Camera model:\t\t", self.sInfo.strSensorName.decode('utf-8'))
            print("Camera serial no.:\t", self.cInfo.SerNo.decode('utf-8'))

        # print(ueye.is_SetExternalTrigger(self.hCam, ueye.IS_GET_EXTERNALTRIGGER))
        # print(ueye.is_SetExternalTrigger(self.hCam, ueye.IS_GET_SUPPORTED_TRIGGER_MODE))
        # print(ueye.is_SetExternalTrigger(self.hCam, ueye.IS_GET_TRIGGER_STATUS))

    def reset_camera(self):
        nRet = ueye.is_ResetToDefault(self.hCam)
        if nRet != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")
        else:
            print("Camera reset complete")

    def set_display_to_DIB(self):
        # Set display mode to DIB
        nRet = ueye.is_SetDisplayMode(self.hCam, ueye.IS_SET_DM_DIB)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetDisplayMode ERROR")
        else:
            print("Camera set to Device Independent Bitmap (DIB) display mode")

    def set_color_mode(self):
        # Set the right color mode
        if int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
            ueye.is_GetColorDepth(self.hCam, self.nBitsPerPixel, self.m_nColorMode)
            bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_BAYER: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_BGRA8_PACKED
            self.nBitsPerPixel = ueye.INT(32)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_CBYCRY: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        else:
            # for monochrome camera models use Y8 mode
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("else")



    def set_camera_settings(self):
        ms = ueye.DOUBLE(19.13)
        nGamma= ueye.INT(100)
        # Gain = ueye.INT(100)
        master_gain = ueye.INT(50)  # Master gain set to 50
        red_gain = ueye.INT(-1)  # No additional gain for red
        green_gain = ueye.INT(-1)  # No additional gain for green
        blue_gain = ueye.INT(-1)  # No additional gain for blue
        gain_boost = ueye.INT(1)  # 1 to enable gain boost
        print("Setting camera settings")
        ret = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ms, ueye.sizeof(ms))
        print('EXP:', ret, ms)
        ret = ueye.is_Gamma(self.hCam,ueye.IS_GAMMA_CMD_SET, nGamma, ueye.sizeof(nGamma))
        print('Gamma:', ret, nGamma)
        # Setting master gain to 50 and enabling Gain Boost
        ret = ueye.is_SetHardwareGain(self.hCam, master_gain, red_gain, green_gain, blue_gain)
        print('Gain Set:', ret, master_gain)
        ret = ueye.is_SetGainBoost(self.hCam, gain_boost)
        print('Gain Boost Enabled:', ret)

    def set_full_auto(self):
        print("Setting mode to full auto")
        disable = ueye.DOUBLE(0)
        enable = ueye.DOUBLE(1)
        zero = ueye.DOUBLE(0)
        ms = ueye.DOUBLE(20)
        rate = ueye.DOUBLE(50)
        newrate = ueye.DOUBLE()
        number = ueye.UINT()

        ret = ueye.is_SetAutoParameter(self.hCam, ueye.IS_SET_ENABLE_AUTO_GAIN, enable, zero)
        print('AG:',ret)
        ret = ueye.is_SetAutoParameter(self.hCam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, enable, zero)
        print('A_SHUTTER:',ret)
        ret = ueye.is_SetFrameRate(self.hCam, rate, newrate)
        print('FR:',ret,newrate)
        ret = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, ms, ueye.sizeof(ms))
        print('EXP:',ret,ms)

    def allocate_memory_for_image(self):
        # Allocates an image memory for an image having its dimensions defined by width and height
        # and its color depth defined by nBitsPerPixel
        nRet = ueye.is_AllocImageMem(self.hCam, self.sInfo.nMaxWidth, self.sInfo.nMaxHeight, self.nBitsPerPixel,
                                     self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)

    def set_external_trigger(self):
        # trigger mode hi_lo triggers on falling edge
        trigmode = ueye.IS_SET_TRIGGER_SOFTWARE
        nRet = ueye.is_SetExternalTrigger(self.hCam, trigmode)
        if nRet != ueye.IS_SUCCESS:
            print("SetExternalTrigger ERROR")
        if(trigmode == ueye.IS_SET_TRIGGER_SOFTWARE):
            print("External trigger mode set: SOFTWARE")
        # print('External trigger mode set', ueye.is_SetExternalTrigger(self.hCam, ueye.IS_GET_EXTERNALTRIGGER), trigmode)

    def set_trigger_counter(self, nValue):
        return ueye.is_SetTriggerCounter(self.hCam, nValue)

    def capture_video(self, wait=False):
        wait_param = ueye.IS_WAIT if wait else ueye.IS_DONT_WAIT
        return ueye.is_CaptureVideo(self.hCam, wait_param)

    def queue_mode(self):
        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID, self.rectAOI.s32Width,
                                       self.rectAOI.s32Height, self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")

    def stop_video(self):
        return ueye.is_StopLiveVideo(self.hCam, ueye.IS_FORCE_VIDEO_STOP)

    def freeze_video(self, wait=False):
        wait_param = ueye.IS_WAIT if wait else ueye.IS_DONT_WAIT
        return ueye.is_FreezeVideo(self.hCam, 30)

    # def take_image(self, display=True, threshold2=200):#TODO
    #
    #
    #     return dp

    def activate_live_video(self):
        # Activates the camera's live video mode (free run mode)
        nRet = self.capture_video(wait=False)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID,
                                       self.rectAOI.s32Width, self.rectAOI.s32Height, self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")
        else:
            print("Live video activated. Press q to quit")
            self.live_video_loop(nRet)
            self.release_memory()



    def triggered_video(self):
        self.set_external_trigger()
        # n_ret = ueye.is_CaptureVideo(self.hCam, ueye.int(2000))
        # if(n_ret != ueye.IS_SUCCESS):
        #     print("is_CaptureVideo ERROR")

        n_ret = self.capture_video()
        if(n_ret != ueye.IS_SUCCESS):
            print("is_FreezeVideo ERROR")

        # self.queue_mode()
        #
        # # Continuous image display
        # while 0 == ueye.IS_SUCCESS:
        #     # ueye.is_Event(self.hCam, UINT
        #     # nCommand, void * pParam, UINT
        #     # nSizeOfParam)
        #     try:
        #         self.freeze_video()
        #         self.take_image()
        #         self.release_memory()
        #     except ValueError:
        #         continue

            # # Press q if you want to end the loop
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     self.stop_video()
            #     break

    def release_memory(self):
        # Release the image memory that was allocated using is_AllocImageMem() and remove it from the driver management
        ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)

    def close_connection(self):
        # Destroys the OpenCv windows
        # cv2.destroyAllWindows()
        # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(self.hCam)
        print()
        print("END")

    def save_image(self,name):
        string = 'D:/SIT/Akid/pictures/'+str(name)+'.bmp'
        print(string)
        self.image_file_params.pwchFileName = ueye.c_wchar_p(string)
        self.image_file_params.nFileType = ueye.IS_IMG_BMP
        self.image_file_params.nQuality = ueye.c_uint(100)
        # self.image_file_params.ppcImageMem = None
        # self.image_file_params.pnImageID = None


        n_ret = ueye.is_ImageFile(hCam = self.hCam,nCommand= ueye.IS_IMAGE_FILE_CMD_SAVE,pParam= self.image_file_params,cbSizeOfParam = ueye.sizeof(self.image_file_params))
        # print(ueye.sizeof(self.image_file_params))
        # print(self.image_file_params.__sizeof__())
        # print(string.__sizeof__())
        if(n_ret != ueye.IS_SUCCESS):
            print("is_ImageFile ERROR")
        if(n_ret == ueye.IS_FILE_READ_INVALID_BMP_ID):
            print("指定的文件不是有效的位图文件")
        if (n_ret == ueye.IS_FILE_READ_OPEN_ERROR):
            print("无法打开文件")
        if (n_ret == ueye.IS_INVALID_PARAMETER):
            print("提交的参数之一超出有效范围，或者此传感器不支持，或者在此模式下不可用")
        if (n_ret == ueye.IS_NOT_SUPPORTED):
            print("此处使用的相机型号不支持此功能或设置。")
        if (n_ret == ueye.IS_NO_SUCCESS):
            print("常规错误消息")
        return n_ret

if __name__ == "__main__":

    cam = Camera()

    cam.configure()

    cam.allocate_memory_for_image()
    cam.triggered_video()
    for i in range(5):
        cam.save_image(name='image'+(str(i)))
    cam.release_memory()
    cam.close_connection()