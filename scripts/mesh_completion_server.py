#!/usr/bin/env python


import tf

import graspit_shape_completion.msg
import subprocess
import time
import numpy as np
import rospy
import PyKDL
import tf_conversions
import plyfile

from shape_completion_server.srv import *

class MeshCompletionServer(object):


    def __init__(self):
        rospy.loginfo("Starting Completion Server")

        self.partial_filepath = "m_partial.pcd"
        self.result_mesh_filepath = "m_post_process.ply"
        self.completion_filepath = "m_completion.binvox"
        self.post_process_executable = "/home/adamjri/post_process/complete"

        self._feedback = graspit_shape_completion.msg.CompleteMeshFeedback()
        self._result = graspit_shape_completion.msg.CompleteMeshResult()

        self.tf_listener = tf.TransformListener()

        self.voxel_completion_srv  = rospy.ServiceProxy('complete_voxel_grid', VoxelCompletion)

        self._as = actionlib.SimpleActionServer("/complete_mesh", graspit_shape_completion.msg.CompleteMeshAction,
            execute_cb=self.completion_cb, auto_start=False)

        self._as.start()

    def _partial_mesh_to_pc(self, partial_mesh):
        point_array = np.asarray(partial_mesh.vertices)
        pc = np.zeros((len(point_array), 4), np.float32)
        for i in range(len(point_array)):
            pc[i][0] = point_array[i].x
            pc[i][1] = point_array[i].y
            pc[i][2] = point_array[i].z

        return pc

    def convert_pc(self, pc, in_frame, out_frame):

        (trans,rot) = self.tf_listener.lookupTransform(out_frame,in_frame rospy.Time(0))
        tf_frame = PyKDL.Frame(rot,trans)
        tf_matrix = tf_conversions.posemath.toMatrix(tf_frame)

        return np.dot(tf_matrix, pc.T), tf_matrix

    def completion_cb(self):
        rospy.loginfo('Received Completion Goal')

        start_time = time.time()
        start_pre_process_time = time.time()

        self._feedback = graspit_shape_completion.msg.CompleteMeshFeedback()
        self._result = graspit_shape_completion.msg.CompleteMeshResult()

        pc_in_world = self._partial_mesh_to_pc(goal.partial_mesh)
        pc_in_camera, world_to_camera = self.convert_pc(pc, '/world', '/camera_optical_frame')
        
        batch_x = np.zeros((1, self.patch_size, self.patch_size, self.patch_size, 1), dtype=np.float32)
        batch_x[0, :, :, :, :], voxel_resolution, offset = reconstruction_utils.build_test_from_pc_scaled(pc_in_camera, self.patch_size)
        end_time = time.time()
        self._result.preprocess_time = int(1000*(end_time - start_pre_process_time))

        start_completion_time = time.time()
        output = self.voxel_completion_srv(batch_x.flatten())
        end_time = time.time()
        self._result.completion_time = int(1000 * (end_time - start_completion_time))

        start_post_process_time = time.time()

        output_vox = output > 0.5

        #go from voxel grid back to list of points.                                                                                                                                                              
        #4xn                                                                                                                                                                                                     
        completion = np.zeros((4, len(output_vox.nonzero()[0])))
        completion[0] = (output_vox.nonzero()[0] ) * voxel_resolution + offset[0]
        completion[1] = (output_vox.nonzero()[1] ) * voxel_resolution + offset[1]
        completion[2] = (output_vox.nonzero()[2] ) * voxel_resolution + offset[2]
        completion[3] = 1.0

        voxel_grid = create_voxel_grid_around_point_scaled(completion_rot[:,0:3], patch_center,
                                          voxel_resolution, args.PATCH_SIZE,
                                          (20,20,20))

        vox = binvox_rw.Voxels(voxel_grid[:,:,:,0],
                   (self.patch_size, self.patch_size, self.patch_size),
                   (offset[0], offset[1], offset[2]),
                   voxel_resolution * self.patch_size,
                   "xyz")
                                                                                                                                                         
        binvox_rw.write(vox, open(self.completion_filepath, 'w'))

        partial_pcd_pc_cf = pcl.PointCloud(np.array(pc_in_camera.transpose()[:, 0:3], np.float32))
        pcl.save(partial_pcd_pc_cf, self.partial_filepath)

        cmd_str = self.post_process_executable + " " + str(self.partial_filepath) + " " + str(self.completion_filepath)
        subprocess.call(cmd_str.split(" "))

        result_mesh = plyfile.PlyData.read(self.result_mesh_filepath)

        result_mesh_vertices = np.zeros((result_mesh['vertex']['x'].shape[0], 4))
        result_mesh_vertices[:, 0] = result_mesh['vertex']['x']
        result_mesh_vertices[:, 1] = result_mesh['vertex']['y']
        result_mesh_vertices[:, 2] = result_mesh['vertex']['z']

        #transform back into world coords.
        result_mesh_vertices = np.dot(np.linalg.inv(world_to_camera), result_mesh_vertices)

        end_time = time.time()
        self._result.postprocess_time = int(1000*(end_time - start_post_process_time))

        for i in len(result_mesh_vertices.shape[0])
            point = geometry_msgs.msg.Point(result_mesh_vertices[i, 0], result_mesh_vertices[i, 1], result_mesh_vertices[i, 2])
            self._result.completed_mesh.vertices.append(point)

        for p0,p1,p2 in zip(result_mesh['triangles']['0'],result_mesh['triangles']['1'],result_mesh['triangles']['2'])
            triangle = shape_msgs.msg.MeshTriangle((p0, p1, p2))
            self._result.completed_mesh.triangles.append(triangle)

        end_time = time.time()
        self._result.total_time = int(1000*(end_time - start_time))
        self._as.set_succeeded(self._result)
        rospy.loginfo('Finished Msg')


if __name__ == "__main__":
    rospy.init_node("mesh_completion_server")
	server = MeshCompletionServer()
    rospy.spin()
