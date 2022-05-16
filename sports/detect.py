import datetime
import time
from threading import Thread
import numpy as np
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import yaml
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from utils.plots import plot_one_box
from models.experimental import attempt_load
from utils.datasets import  LoadImages
from utils.general import (
	check_img_size, non_max_suppression, apply_classifier, scale_coords,
	xyxy2xywh,  strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import clean_str

with open('data/coco.yaml') as stream:
	config = yaml.safe_load(stream)
classnames=config['names']


def create_rect_box(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
	shape = img.shape[:2]  # current shape [height, width]
	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scaleup:  # only scale down, do not scale up (for better test mAP)
		r = min(r, 1.0)
	ratio = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
	if auto:  # minimum rectangle
		dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
	elif scaleFill:  # stretch
		dw, dh = 0.0, 0.0
		new_unpad = (new_shape[1], new_shape[0])
		ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
	return img, ratio, (dw, dh)




class LoadWebCam:  # multiple IP or RTSP cameras
	def __init__(self, sources='streams.txt', img_size=640, stride=32):
		self.mode = 'stream'
		self.img_size = img_size
		self.stride = stride

		if os.path.isfile(sources):
			with open(sources, 'r') as f:
				sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
		else:
			sources = [sources]

		n = len(sources)
		self.imgs = [None] * n
		self.sources = [clean_str(x) for x in sources]  # clean source names for later
		for i, s in enumerate(sources):
			# Start the thread to read frames from the video stream
			print(f'{i + 1}/{n}: {s}... ', end='')
			url = eval(s) if s.isnumeric() else s

			cap = cv2.VideoCapture(url)
			assert cap.isOpened(), f'Failed to open {s}'
			w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

			_, self.imgs[i] = cap.read()  # guarantee first frame
			thread = Thread(target=self.update, args=([i, cap]), daemon=True)
			print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
			thread.start()
		print('')  # newline

		# check for common shapes
		s = np.stack([create_rect_box(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
		self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
		if not self.rect:
			print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

	def update(self, index, cap):
		# Read next stream frame in a daemon thread
		n = 0
		while cap.isOpened():
			n += 1
			# _, self.imgs[index] = cap.read()
			cap.grab()
			if n == 4:  # read every 4th frame
				success, im = cap.retrieve()
				self.imgs[index] = im if success else self.imgs[index] * 0
				n = 0
			time.sleep(1 / self.fps)  # wait time

	def __iter__(self):
		self.count = -1
		return self

	def __next__(self):
		self.count += 1
		img0 = self.imgs.copy()
		if cv2.waitKey(1) == ord('q'):  # q to quit
			cv2.destroyAllWindows()
			raise StopIteration

		# create_rect_box
		img = [create_rect_box(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

		# Stack
		img = np.stack(img, 0)

		# Convert
		img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
		img = np.ascontiguousarray(img)

		return self.sources, img, img0, None

	def __len__(self):
		return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years





def plot_one_bounding_box(x, img, color_flag, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    #color = color or [random.randint(0, 255) for _ in range(3)]
    color=(0,255,0)
    if color_flag==1:
    	color=(0,0,255)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def set_video_for_saving(save_path,vid_path,vid_cap,vid_writer,im0):
	if vid_path != save_path:  # new video
		vid_path = save_path
		
		if isinstance(vid_writer, cv2.VideoWriter):
			vid_writer.release()  # release previous video writer
		if vid_cap:  # video
			
			fps = 10
			
			w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		else:  # stream
			fps, w, h = 3, im0.shape[1], im0.shape[0]
			save_path += '.mp4'
		
		vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
	return save_path,vid_path,vid_cap,vid_writer,im0

def video_saving_in_folder(vid_writer,im0,flag,obj_identify,cnt):

	fall_text=''
	slide_factor=0
	for object_name, count_value in obj_identify.items():
		fall_text=str(object_name) + ':' + str(count_value) 
		#color = [random.randint(0, 255) for _ in range(3)]
		color = color=(0,0,255)
		font = cv2.FONT_HERSHEY_SIMPLEX
		im0=cv2.putText(im0,str(fall_text),(25,130+slide_factor),font,1,color,4,cv2.LINE_AA)
		slide_factor+=30
	# if flag==1:
		
	# 	standardtime = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y %H-%M-%S')
	# 	imagename=standardtime+' '+str(cnt)
	# 	font = cv2.FONT_HERSHEY_SIMPLEX
	# 	cv2.putText(im0,imagename,(10,100),font,1,(2,100,255),4,cv2.LINE_AA)
	# 	fall_text='Detections !!!!'
	# 	im0=cv2.putText(im0,fall_text,(25,130),font,1,(2,10,255),4,cv2.LINE_AA)

	vid_writer.write(im0)
	cv2.imshow('video',im0)



def detect(weights,output,source,img_size,conf_thres,iou_thres,device,view_img,save_txt,agnostic_nms,augment,update,classes,save_img=False):

	out=output
	imgsz=img_size

   
	webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
	
	# Initialize
	set_logging()
	device = select_device(device)
	if os.path.exists(out):
		shutil.rmtree(out)  # delete output folder
	os.makedirs(out)  # make new output folder
	half = device.type != 'cpu'  # half precision only supported on CUDA

	# Load model
	model = attempt_load(weights, map_location=device)  # load FP32 model
	#print('model',model)
	imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
	if half:
		model.half()  # to FP16

	# Second-stage classifier
	classify = False


	vid_path, vid_writer = None, None
	if webcam:
		
		view_img = True
		cudnn.benchmark = True  # set True to speed up constant image size inference
		dataset = LoadWebCam(source, img_size=imgsz)
	else:
		
		save_img = True
		dataset = LoadImages(source, img_size=imgsz)
   
	names = model.module.names if hasattr(model, 'module') else model.names
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

	t0 = time.time()
	img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
	_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
	cnt=0

							
	for path, img, im0s, vid_cap in dataset:

		cnt+=1
		
		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		
		t1 = time_synchronized()
		pred = model(img, augment=augment)[0]

		# Apply NMS
		pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
		t2 = time_synchronized()

		# Apply Classifier
		if classify:
			pred = apply_classifier(pred, modelc, img, im0s)

		# Process detections
		
		
		flag=0
		for i, det in enumerate(pred):  # detections per image
			if webcam:  # batch_size >= 1
				p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
			else:
				p, s, im0 = path, '', im0s

			im0_org=im0.copy()
			save_path = str(Path(out) / Path(p).name)
			
			img_name=Path(p).name.split('.')[0]
			
			txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
			s += '%gx%g ' % img.shape[2:]  # print string
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

			
			all_labels=[]
			labels=[]
			

			if det is not None and len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, -1].unique():
					if c:

						n = (det[:, -1] == c).sum()  # detections per class
						
						s += '%g %ss, ' % (n, classnames[int(c)])  # add to string
					else:
						pass
				# Write results		
				color_flag=0	
				obj_identify={}
				obj_no=0   
				for *xyxy, conf, cls in reversed(det):
					#print(*xyxy, conf, cls)
					if save_img or view_img:  # Add bbox to image

						label = '%s %s' % (classnames[int(cls)], str(int(conf*100))+'%')
						plot_one_bounding_box(xyxy, im0, color_flag, label=label, color=colors[int(cls)], line_thickness=3)
						category='%s'%classnames[int(cls)]
						
						if (category in obj_identify):
							obj_identify[category] += 1
						else:
							obj_identify[category] = 1

						confidence = str(round((float('%.3f'%conf)*100), 2))
						c_label = str(round((float('%.2f'%cls))))
						bb_x1 = int(round((float('%.2f'%xyxy[0]))))
						bb_y1 = int(round((float('%.2f'%xyxy[1]))))
						bb_x2 = int(round((float('%.2f'%xyxy[2]))))
						bb_y2 = int(round((float('%.2f'%xyxy[3]))))
						bb_box=[bb_x1,bb_y1,bb_x2,bb_y2]
						horizontal_motion=bb_x2-bb_x1
						vertical_motion=bb_y2-bb_y1						

						# if category=='person':
						# 	color_flag=1
						# 	if horizontal_motion>vertical_motion:
						# 		plot_one_bounding_box(xyxy, im0, color_flag, label=label, color=colors[int(cls)], line_thickness=3)
						# 		flag=1
				

				if view_img:
					# cv2.imshow('videos', im0)				
					save_path,vid_path,vid_cap,vid_writer,im0=set_video_for_saving(save_path,vid_path,vid_cap,vid_writer,im0)
					video_saving_in_folder(vid_writer,im0,flag,obj_identify,cnt)

					if cv2.waitKey(1) == ord('q'):  # q to quit
						raise StopIteration
					

				# Save results (image with detections)
				if save_img:
					if dataset.mode == 'image':
						# save in images folder
						cv2.imwrite(save_path, im0)
						
					else:  # 'video' or 'stream'

						save_path,vid_path,vid_cap,vid_writer,im0=set_video_for_saving(save_path,vid_path,vid_cap,vid_writer,im0)
						video_saving_in_folder(vid_writer,im0,flag,obj_identify,cnt)
						# video_saving_in_folder(save_path,vid_path,vid_writer,vid_cap,im0,flag,cnt)

					if cv2.waitKey(1) == ord('q'):  # q to quit
						raise StopIteration



	print('Done. (%.3fs)' % (time.time() - t0))



def fall_detection(svalue,wvalue):
	weights='weights/tiny-trained.pt'
	if wvalue==1:
		weights='weights/trained.pt'
	output='output/'
	source='input/'
	if svalue==0:
		source='0'
	img_size=640
	conf_thres=0.3
	iou_thres=0.5
	device='cpu'
	view_img=False
	save_txt=False
	agnostic_nms=False
	augment=False
	update=False
	classes=None

	with torch.no_grad():     
		
		detect(weights,output,source,img_size,conf_thres,iou_thres,device,view_img,save_txt,agnostic_nms,augment,update,classes)
	 
