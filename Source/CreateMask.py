# this class snippet is inspired by various stckexchange discussion forums

import numpy as np
import cv2

class DrawMask(object):
	def __init__(self,window):
		self.window=window
		self.done=False
		self.current=(0,0)
		self.points=[]

	def ClickMouse(self, event, x,y,buttons,user_param):

		if self.done:
			return
		if event==cv2.EVENT_MOUSEMOVE:
			self.current=(x,y)

		elif event== cv2.EVENT_LBUTTONDOWN:
			self.points.append((x,y))
		elif event==cv2.EVENT_RBUTTONDOWN:
			self.done=True

	def run(self,I):
		cv2.namedWindow(self.window, flags=cv2.WINDOW_AUTOSIZE)
		cv2.imshow(self.window,I)
		cv2.setMouseCallback(self.window, self.ClickMouse)

		while (not self.done):
			if (len(self.points)>0):
				cv2.polylines(I,np.array([self.points]), False, [255,255,255],1)
				cv2.circle(I,self.points[-1],1,[200,0,50],3)
			cv2.imshow(self.window, I)

			if cv2.waitKey(50)==27:
				self.done=True
		if (len(self.points)>0):
			mask=np.zeros(I.shape)
			cv2.fillPoly(mask,np.array([self.points]),[255,255,255])
		cv2.imshow(self.window,mask)
		cv2.waitKey()
		cv2.destroyWindow(self.window), print(mask.shape)
         
		return mask


if __name__ == "__main__":
    pd = DrawMask("Mask")
    I=cv2.imread('aras.jpg')
    image = pd.run(I)
    cv2.imwrite("polygon.png", image)
    print("Polygon = %s" % pd.points)
