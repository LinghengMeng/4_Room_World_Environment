#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 15:07:15 2018

@author: jack.lingheng.meng
"""

def obj_get_vision_image(self, handle):
		resolution, image = self.RAPI_rc(vrep.simxGetVisionSensorImage( self.cID,handle,
			0, # assume RGB
			self.opM_get,))
		dim, im = resolution, image
		nim = np.array(im, dtype='uint8')
		nim = np.reshape(nim, (dim[1], dim[0], 3))
		nim = np.flip(nim, 0)  # horizontal flip
		#nim = np.flip(nim, 2)  # RGB -> BGR
		return nim