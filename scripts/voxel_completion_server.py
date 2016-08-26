#!/usr/bin/env python

from shape_completion_server.srv import *
import rospy

def handle_add_two_ints(req):
    print "Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b))
    return AddTwoIntsResponse(req.a + req.b)

def add_two_ints_server():
    rospy.init_node('add_two_ints_server')
    
    print "Ready to add two ints."
    rospy.spin()


class VoxelCompletionServer(object):

    def __init__(self, patch_size, model_python_module, weight_filepath):

        rospy.loginfo("Starting Voxel Completion Server")

        self.patch_size = patch_size
        self.model_python_module = model_python_module
        self.weight_filepath = weight_filepath

        s = rospy.Service('complete_voxel_grid', AddTwoInts, self.complete_voxel_grid)

    	self.model= importlib.import_module(model_python_module).get_model(self.patch_size)
    	self.model.load_weights(weights_filepath)


    def complete_voxel_grid(self, batch_x_B012C_flat):
    	batch_x_B012C = batch_x_B012C_flat.reshape((1,self.patch_size,self.patch_size,self.patch_size,1))
    	#make batch B2C01 rather than B012C                                                                                                                                                                  
        batch_x = batch_x_B012C_flat.transpose(0, 3, 4, 1, 2)

        pred = self.model._predict(batch_x)
        pred = pred.reshape(1, self.patch_size, 1, self.patch_size, self.patch_size)

        pred_as_b012c = pred.transpose(0, 3, 4, 1, 2)

        mask = reconstruction_utils.get_occluded_voxel_grid(batch_x_B012C[0, :, :, :, 0])
        completed_region = pred_as_b012c[0, :, :, :, 0]

        completed_region *= mask

        return completed_region.flatten()


if __name__ == "__main__":

	patch_size = 40
	model_python_module = ""
	weight_filepath = ""

    vcs = VoxelCompletionServer(
    	patch_size, 
    	model_python_module
    	weight_filepath)

    rospy.spin()