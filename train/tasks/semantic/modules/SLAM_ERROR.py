from gradslam.odometry import GradICPOdometryProvider
from kornia.geometry.linalg import compose_transformations
import gradslam
import torch
import open3d as o3d
import numpy as np
from pytorch3d.ops import points_normals

class Graph:
    def __init__(self):
      self.loc={}
      self.fromNode=[]
      self.toNode=[]
      self.measurement=[]
      self.information=[]
      self.Type=[]
    def add_vertex(self,id,pose):
        self.loc[id]=pose
    def add_edge(self,Type,fromNode,toNode,measurement,information):
      self.fromNode.append(fromNode)
      self.toNode.append(toNode)
      self.measurement.append(measurement)
      self.information.append(information)
      self.Type.append(Type)
    def get_pose(self,idx):
      return self.loc.get(idx,None)

def rotationMatrixToEulerAngles(R) :

    #assert(isRotationMatrix(R))

    sy = torch.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = torch.atan2(R[2,1] , R[2,2])
        y = torch.atan2(-R[2,0], sy)
        z = torch.atan2(R[1,0], R[0,0])
    else :
        x = torch.atan2(-R[1,2], R[1,1])
        y = torch.atan2(-R[2,0], sy)
        z = 0

    return x,y,z


    

def Convert_to_pose(trans):
  poseeuler=rotationMatrixToEulerAngles(torch.squeeze(trans)[:3,:3])   # Converting 3x3 orientation of pose of robot to euler angles 
  pose=torch.concat((trans.squeeze()[:2,3],poseeuler[2].unsqueeze(0)))
  return pose

def compute_normals(gnd_scans,gen_scans):
  print('Compute start')
  gnd_pcd_normals,gen_pcd_normals=[],[]
  for i,j in zip(gnd_scans,gen_scans):
    gnd_pcd_normals.append(points_normals.estimate_pointcloud_normals(i.unsqueeze(0)).squeeze(0))
    gen_pcd_normals.append(points_normals.estimate_pointcloud_normals(j.unsqueeze(0)).squeeze(0))
  print('Done')
  return gnd_pcd_normals, gen_pcd_normals
    
def find_poseGraph(scans,iterations=10):
    g=Graph()
    prev_idx=0
    print('2')
    odomprov = GradICPOdometryProvider ( iterations , 1e-4, lambda_max=2.0, B=1.0, B2=1.0, nu=200.0 )
    poses=[torch.eye(4, dtype=torch.float, device = "cuda").view(1,1,4,4)]  # list containing poses of nodes
    vertex_idx=1
    print("3")
    poseeuler=rotationMatrixToEulerAngles(torch.squeeze(poses[-1])[:3,:3])       #Converting init orientation to euler angles
    pose=torch.concat((poses[-1].squeeze()[:2,3],poseeuler[2].unsqueeze(0)))     
    g.add_vertex(vertex_idx,pose)
    print('4')
    for idx in range(len(scans)):
        if idx==0:
            pass
        pcd1=scans[prev_idx].to("cuda")
        pcd2=scans[idx].to("cuda")
        vertex_idx+=1
        transform=odomprov.provide(pcd1,pcd2)
        #print(transform.get_device(),"*********",poses[-1].get_device())
        pcd_pose=compose_transformations(poses[-1].squeeze(1),transform.squeeze(1)).unsqueeze(1)
        poses.append(pcd_pose)
        pose=Convert_to_pose(poses[-1])
        T=Convert_to_pose(transform.squeeze())
        g.add_vertex(vertex_idx,pose)
        g.add_edge("P", vertex_idx-1,vertex_idx,T,torch.eye(3,device="cuda"))
        prev_idx=idx
    print('5')
    return g

def Slam_error(gnd_scans, gen_scans):
    print('here')
    gnd_scan_normals, gen_scan_normals=compute_normals(gnd_scans,gen_scans)
    #print(gnd_scans[0].shape,gnd_scan_normals[0].shape)
    print('1')
    gnd_graph=find_poseGraph(gradslam.Pointclouds(gnd_scans,gnd_scan_normals))
    gen_graph=find_poseGraph(gradslam.Pointclouds(gen_scans,gen_scan_normals))
    print("6")
    loss=0
    for i in range(len(gnd_graph.measurement)):
        err=gnd_graph.measurement[i]-gen_graph.measurement[i]
        loss=loss+err**2
    return loss.mean()
