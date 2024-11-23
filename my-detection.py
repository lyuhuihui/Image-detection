#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("cat_0.jpg")

display = jetson.utils.videoOutput("b.jpg") 

img = camera.Capture()
detections = net.Detect(img)

display.Render(img)
display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

for i in detections:
	print(i)
