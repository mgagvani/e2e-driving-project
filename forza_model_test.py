import cv2
import pandas as pd
import time
import numpy as np
import torch
import pyxinput
import mss
import threading
from PIL import Image

import sys


class VirtualController:
    '''
    Simulate a controller with a virtual joystick.
    See https://github.com/mgagvani/donkeycar/blob/9b1ee54c9c9aa863b21477eeee08cd5562d5b48c/donkeycar/parts/actuator.py#L1176
    '''
    def __init__(self):
        self.angle = 0
        self.throttle = 0
        self.controller = pyxinput.vController()
    def run(self, angle, throttle):
        self.angle = angle
        self.throttle = throttle
        # angle
        self.controller.set_value('AxisLx', angle)
        
        # throttle
        if throttle > 0:
            self.controller.set_value('TriggerR', throttle)
        else:
            self.controller.set_value('TriggerL', -throttle)
    def shutdown(self):
        self.controller.UnPlug()

class ScreenCamera():
    '''
    Camera that uses mss to capture the screen
    For capturing video games
    See https://github.com/mgagvani/donkeycar/blob/9b1ee54c9c9aa863b21477eeee08cd5562d5b48c/donkeycar/parts/camera.py#L334
    '''

    def __init__(self, image_w=160, image_h=120, image_d=3,
                 vflip=False, hflip=False):
        
        self.image_w = image_w
        self.image_h = image_h
        self.image_d = image_d
        self.vflip = vflip
        self.hflip = hflip
        self.sct = mss.mss()
        self.running = True

        self._monitor_thread = threading.Thread(target=self.take_screenshot, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def take_screenshot(self):
        # Capture the screen
        monitor = {"top": 0, 
                   "left": 0, 
                   "width": 1920, 
                   "height": 1080
                   }
        sct_img = self.sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
        img = img.resize((self.image_w, self.image_h))
        img_arr = np.asarray(img)
        if self.vflip:
            img_arr = np.flipud(img_arr)
        if self.hflip:
            img_arr = np.fliplr(img_arr)
        self.frame = img_arr
        # img.save('screen{}.jpg'.format(time.time()))
        return img

    def update(self):
        if self.running:
            img = self.take_screenshot()
            return img

    def run(self):
        self.update()
        assert self.frame is not None
        return self.frame

    def run_threaded(self):
        return self.frame
    
    def shutdown(self):
        self.running = False
        self.sct.close()

if __name__ == "__main__":
    args = sys.argv[1:]
    from models import PilotNet, ResNet18Pilot
    if len(args) == 0:
        model = PilotNet()
    elif args[0] == "resnet":
        model = ResNet18Pilot()

    # load model
    torch.set_default_device("cuda")
    model.load_state_dict(torch.load("model.pth"))
    model = model.to("cuda")
    model.eval()

    # init controller
    controller = VirtualController()

    # init camera
    camera = ScreenCamera()

    # misc
    times = []
    print("starting")

    while True:
        try:
            img = camera.run()
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to("cuda") / 255.0
            img = img[:, :, 40:, :]
            t0 = time.time()
            out = model(img)
            times.append(time.time() - t0)
            out = out.squeeze(0).detach().cpu().numpy()
            print((out[1], out[0]))
            controller.run(out[1], out[0]) # steer, throttle
            # controller.run(out[1], 0.5) # steer, throttle
            
        except KeyboardInterrupt:
            camera.shutdown()
            controller.shutdown()
            break

        except Exception as e:
            print("Error: ", e)
            camera.shutdown()
            controller.shutdown()
            break

    print("Avg inference time (ms): ", sum(times) / len(times) * 1000)