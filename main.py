import time
from asyncio import wait_for

from pyueye import ueye

from ueyeCamera import Camera
import jena
import numpy as np

port = 'COM10'

def save_to_file(filename, content):
    fh = open(filename, 'w')
    fh.write(content)
    fh.close()


if __name__ == "__main__":
    # initial camera(CAMERA)
    cam = Camera()
    cam.configure()
    cam.allocate_memory_for_image()
    cam.triggered_video()

    #   initial stage-------piezo.get_position()-------piezo.set_position(x~0-100um)
    piezo = jena.NV40(port)

    # take pictures(CAMERA/STAGE)
    max_offset = 30
    sensitivity = max_offset / 500
    enable = 1
    count = 0
    image_error_count = 0
    init_position = expected_position  = 0
    real_position = float((piezo.set_position(init_position))[0:5])
    while enable:
        if expected_position > max_offset:
            print('position reached 10, to get accurate images, stop moving')
            break
        if expected_position*0.99-0.01 <= real_position <= expected_position*1.01 + 0.01:
            n_ret = cam.save_image(name = str(count))
            while n_ret != ueye.IS_SUCCESS:
                time.sleep(0.1)
                n_ret = cam.save_image(name=str(count))
                if n_ret != ueye.IS_SUCCESS:
                    image_error_count = image_error_count + 1
                if image_error_count == 10:
                    print('image save error 10 times')
                    break
                if n_ret == ueye.IS_SUCCESS:
                    image_error_count = 0

            save_to_file(filename='D:/SIT/Akid/pictures/' + str(count) + '.txt', content=str(real_position))
            expected_position = expected_position + 1 * sensitivity
            count +=1
            real_position = float((piezo.set_position(expected_position))[0:5])
            print('expected_positon:'+str(expected_position))
            print('real_position:' + str(real_position))
            # print(type(expected_position))

        else:

            print('position error Or waiting for movement\n')
            time.sleep(1)
            # print('dont wait for movement\n')
            # break


    # release memory & disconnect(CAMERA)
    cam.release_memory()
    cam.close_connection()
    # Return to manual control(STAGE)
    # piezo.__exit__()