import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import logging
logger = logging.getLogger(__name__)

def timeit(tag, t):
   print("{}: {}s".format(tag, time() - t))
   return time()

def pc_normalize(pc):
   l = pc.shape[0]
   centroid = np.mean(pc, axis=0)
   pc = pc - centroid
   m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
   pc = pc / m
   return pc

def square_distance(src, dst):
   """
   Calculate Euclid distance between each two points.

   src^T * dst = xn * xm + yn * ym + zn * zm；
   sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
   sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
   dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
      = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

   Input:
     src: source points, [B, N, C]
     dst: target points, [B, M, C]
   Output:
     dist: per-point square distance, [B, N, M]
   """
   B, N, _ = src.shape
   _, M, _ = dst.shape
   dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
   dist += torch.sum(src ** 2, -1).view(B, N, 1)
   dist += torch.sum(dst ** 2, -1).view(B, 1, M)
   return dist


def index_points(points, idx):
   """

   Input:
     points: input points data, [B, N, C]
     idx: sample index data, [B, .. , C ] shape varies
   Return:
     new_points:, indexed points data, [B, S, C]
   """
   device = points.device
   B = points.shape[0]
   # get idx shape as a list
   view_shape = list(idx.shape)
   # change shape to [B, 1, 1 ...]
   view_shape[1:] = [1] * (len(view_shape) - 1)
   # get idx shape again
   repeat_shape = list(idx.shape)
   # change to [1,...]
   repeat_shape[0] = 1
   # build vector [B,...] where entries are set to the batch number
   # logger.info('view_shape = %s repeat_shape = %s',view_shape,repeat_shape)
   batch_indices = torch.arange(B, dtype=torch.long,device=device).view(view_shape).repeat(repeat_shape)
   
   # select points based on indices
   new_points = points[batch_indices, idx, :]
   return new_points


def farthest_point_sample(xyz, npoint):
   """
   Input:
     xyz: pointcloud data, [B, N, 3]
     npoint: number of clusters
   Return:
     centroids: sampled pointcloud index, [B, npoint]
   """
   # logger.info(f'fps: xyz.shape={xyz.shape}')
   device = xyz.device
   B, N, C = xyz.shape

   # create empty vector for npoint number of centroids
   centroids = torch.zeros(B, npoint, dtype=torch.long,device=device)
   # for each input point, a distance
   distance = torch.ones(B, N, dtype=xyz.dtype, device=device) * 1e10
   # for each input set, the farthest point is randomly set between 0-N
   farthest = torch.randint(0, N, (B,), dtype=torch.long,device=device)
   # just [0...B]
   batch_indices = torch.arange(B, dtype=torch.long,device=device)
   # logger.info(f'fps: batch_indices.shape={batch_indices.shape} farthest.shape={farthest.shape} centroids.shape={centroids.shape}')

   # loop over each cluster centroid
   for i in range(npoint):
      # for all batches, set the initial farthest point randomly
      centroids[:, i] = farthest
      # select the farthest point for each batch
      centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
      # calculate the distance of each point to this farthest point
      dist = torch.sum((xyz - centroid) ** 2, -1)
      # use this new distance if it is less than the old distance
      mask = dist < distance
      distance[mask] = dist[mask]

      # set the farthest point to the most distant point
      farthest = torch.max(distance, -1)[1]
   return centroids


def query_ball_point_prev(radius, nsample, xyz, new_xyz):
   """
   Input:
      radius: local region radius
      nsample: max sample number in local region
      xyz: all points, [B, N, 3]
      new_xyz: query points, [B, S, 3]
   Return:
      group_idx: grouped points index, [B, S, nsample]
   """
   device = xyz.device
   B, N, C = xyz.shape
   _, S, _ = new_xyz.shape
   # logger.info(f'qb: xyz.shape={xyz.shape} new_xyz.shape={new_xyz.shape} radius={radius} nsample={nsample}')
   # logger.info(f'qb: xyz.shape={xyz.shape} new_xyz.shape={new_xyz.shape}')
   # For each batch and point group, create indices of all points
   group_idx = torch.arange(N, dtype=torch.long,device=device).view(1, 1, N).repeat([B, S, 1])  # [B, S, N]
   # calculate the square distance between all points and the subsample
   sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
   # logger.info(f'qb: sqrdists.shape={sqrdists.shape} group_idx.shape={group_idx.shape}')
   # set points outside the radius to the max npoint index
   group_idx[sqrdists > radius ** 2] = N
   # sort indices from smallest to largest and clip to nsample
   group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # [B, S, nsample]
   # create array with only first entry
   group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])  # [B, S, nsample]
   # logger.info(f'qb: group_first={group_first.shape} {group_first.min()} {group_first.max()}')
   # get mask of entries set to max index
   mask = group_idx == N
   # logger.info(f'qb: mask={mask.int().sum()} mask.shape={mask.shape}')
   # set all the points outside the group to be the index of the first point in the group.
   group_idx[mask] = group_first[mask]
   # logger.info(f'qb: group_idx={group_idx.shape} {group_idx.min()} {group_idx.max()}')
   return group_idx

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample] - selected by closest to centroid
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = square_distance(new_xyz, xyz)
    sorted_sqrdists, sorted_sqrdist_idxs = torch.sort(sqrdists)
    sorted_sqrdists = sorted_sqrdists[:, :, :nsample]
    sorted_sqrdist_idxs = sorted_sqrdist_idxs[:, :, :nsample]

    group_first = sorted_sqrdist_idxs[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    out_of_ball = (sorted_sqrdists > radius**2)
    # set points not in radius to the centroid idx
    sorted_sqrdist_idxs[out_of_ball] = group_first[out_of_ball]
    
    return sorted_sqrdist_idxs


def sample_and_group(npoint, radius, nsample, xyz, points, tracks=[]):
   """
   Input:
      npoint:
      radius:
      nsample:
      xyz: input points position data, [B, N, 3]
      points: input points data, [B, N, D]
   Return:
      new_xyz: sampled points position data, [B, npoint, nsample, 3]
      new_points: sampled points data, [B, npoint, nsample, 3+D]
   """
   # logger.info(f'sg: xyz.shape={xyz.shape}')
   B, N, C = xyz.shape
   S = npoint
   # get npoint indices for each input point set
   if len(tracks) != 0:
      new_xyz = tracks
   else:
      centroids = farthest_point_sample(xyz, npoint) # [B, npoint]
      new_xyz = index_points(xyz, centroids) # [B, npoint, 3]

   # logger.info(f'sg: fps_idx.shape={fps_idx.shape}')

   # logger.info(f'sg: new_xyz.shape={new_xyz.shape}')
   idx = query_ball_point(radius, nsample, xyz, new_xyz) # [B, npoint, nsample]
   # logger.info(f'sg: idx.shape={idx.shape} {idx.max()}')
   grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
   # logger.info(f'sg: grouped_xyz={grouped_xyz.shape}')
   # grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

   grouped_points = index_points(points, idx)
   # logger.info(f'sg: grouped_points={grouped_points.shape}')
   new_points = grouped_points
   return new_xyz, new_points


def sample_and_group_all(xyz, points):
   """
   Input:
      xyz: input points position data, [B, N, 3]
      points: input points data, [B, N, D]
   Return:
      new_xyz: sampled points position data, [B, 1, 3]
      new_points: sampled points data, [B, 1, N, 3+D]
   """
   device = xyz.device
   B, N, C = xyz.shape
   new_xyz = torch.zeros(B, 1, C,device=device)
   grouped_xyz = xyz.view(B, 1, N, C)
   new_points = points.view(B, 1, N, -1)

   return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
   def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
      super(PointNetSetAbstraction, self).__init__()
      self.npoint = npoint
      self.radius = radius
      self.nsample = nsample
      self.mlp_convs = nn.ModuleList()
      self.mlp_bns = nn.ModuleList()
      last_channel = in_channel
      for out_channel in mlp:
         self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
         self.mlp_bns.append(nn.BatchNorm2d(out_channel))
         last_channel = out_channel
      self.group_all = group_all

   def forward(self, xyz, points, tracks=[]):
      """
      Input:
         xyz: input points position data, [B, N, C]
         points: input points data, [B, N, D]
      Return:
         new_xyz: sampled points position data, [B, S, C]
         new_points_concat: sample points feature data, [B, S, D']
      """
      # logger.info(f'A xyz.shape={xyz.shape} points.shape={points.shape}')

      if self.group_all:
         new_xyz, new_points = sample_and_group_all(xyz, points)
      else:
         new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, tracks)
      
      # logger.info(f'B new_xyz.shape={new_xyz.shape} new_points.shape={new_points.shape}')
      new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
      
      # new_xyz: sampled points position data, [B, npoint, C]
      # new_points: sampled points data, [B, npoint, nsample, C+D]
      # logger.info(f'C new_points.shape={new_points.shape}')
      for i, conv in enumerate(self.mlp_convs):
         bn = self.mlp_bns[i]
         new_points =  F.relu(bn(conv(new_points)))
         # logger.info(f'D new_points.shape={new_points.shape}')

      new_points = torch.max(new_points, 2)[0]
      new_points = new_points.permute(0, 2, 1) # [B, nsample, D'] # hmmm npoint instead of nsample ??
      # new_xyz = new_xyz.permute(0, 2, 1)
      # logger.info(f'E new_xyz.shape={new_xyz.shape} new_points.shape={new_points.shape}')
      return new_xyz, new_points

   def to(self, memory_format):
      self.mlp_convs = self.mlp_convs.to(memory_format = memory_format)
      self.mlp_bns = self.mlp_bns.to(memory_format = memory_format)
      return self


class PointNetSetAbstractionMsg(nn.Module):
   def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
      super(PointNetSetAbstractionMsg, self).__init__()
      self.npoint = npoint
      self.radius_list = radius_list
      self.nsample_list = nsample_list
      self.conv_blocks = nn.ModuleList()
      self.bn_blocks = nn.ModuleList()
      for i in range(len(mlp_list)):
         convs = nn.ModuleList()
         bns = nn.ModuleList()
         last_channel = in_channel + 3
         for out_channel in mlp_list[i]:
            convs.append(nn.Conv2d(last_channel, out_channel, 1))
            bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
         self.conv_blocks.append(convs)
         self.bn_blocks.append(bns)

   def forward(self, xyz, points, tracks=[]):
      """
      Input:
         xyz: input points position data, [B, C, N]
         points: input points data, [B, D, N]
      Return:
         new_xyz: sampled points position data, [B, C, S]
         new_points_concat: sample points feature data, [B, D', S]
      """
      # xyz = xyz.permute(0, 2, 1)
      # if points is not None:
      #    points = points.permute(0, 2, 1)

      B, N, C = xyz.shape
      S = self.npoint if len(tracks) == 0 else tracks.shape[1]

      if len(tracks) == 0:
         new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
      else:
         new_xyz = tracks

      new_points_list = []
      for i, radius in enumerate(self.radius_list):
         K = self.nsample_list[i]
         group_idx = query_ball_point(radius, K, xyz, new_xyz)
         grouped_xyz = index_points(xyz, group_idx)
         grouped_xyz -= new_xyz.view(B, S, 1, C)
         if points is not None:
            grouped_points = index_points(points, group_idx)
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
         else:
            grouped_points = grouped_xyz

         grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
         for j in range(len(self.conv_blocks[i])):
            conv = self.conv_blocks[i][j]
            bn = self.bn_blocks[i][j]
            grouped_points =  F.relu(bn(conv(grouped_points)))
         new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
         new_points_list.append(new_points)

      # new_xyz = new_xyz.permute(0, 2, 1)
      new_points_concat = torch.cat(new_points_list, dim=1)
      return new_xyz, new_points_concat

   def to(self, memory_format):
      self.mlp_convs = self.mlp_convs.to(memory_format = memory_format)
      self.mlp_bns = self.mlp_bns.to(memory_format = memory_format)

class PointNetFeaturePropagation(nn.Module):
   def __init__(self, in_channel, mlp):
      super(PointNetFeaturePropagation, self).__init__()
      self.mlp_convs = nn.ModuleList()
      self.mlp_bns = nn.ModuleList()
      last_channel = in_channel
      for out_channel in mlp:
         self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
         self.mlp_bns.append(nn.BatchNorm1d(out_channel))
         last_channel = out_channel

   def forward(self, xyz1, xyz2, points1, points2):
      """
      Input:
         xyz1: input points position data, [B, C, N]
         xyz2: sampled input points position data, [B, C, S]
         points1: input points data, [B, D, N]
         points2: input points data, [B, D, S]
      Return:
         new_points: upsampled points data, [B, D', N]
      """
      # xyz1 = xyz1.permute(0, 2, 1)
      # xyz2 = xyz2.permute(0, 2, 1)

      # logger.info(f'xyz1.shape={xyz1.shape}')
      # logger.info(f'xyz2.shape={xyz2.shape}')
      # if points1 is not None:
      #    logger.info(f'points1.shape={points1.shape}')
      # logger.info(f'points2.shape={points2.shape}')
      # points2 = points2.permute(0, 2, 1)
      B, N, C = xyz1.shape
      _, S, _ = xyz2.shape


      # if torch.any(torch.isnan(points2)):
      #   logger.error('points2 is nan')

      # if points1 is not None:
      #    logger.info(f'xyz1.shape={xyz1.shape} xyz2.shape={xyz2.shape} points1.shape={points1.shape} points2.shape={points2.shape}')
      # else:
      #    logger.info(f'xyz1.shape={xyz1.shape} xyz2.shape={xyz2.shape} points2.shape={points2.shape}')

      if S == 1:
         interpolated_points = points2.repeat(1, N, 1)
      else:
         dists = square_distance(xyz1, xyz2)
         dists, idx = dists.sort(dim=-1)
         dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

         dist_recip = 1.0 / (dists + 1e-8)

         # if torch.any(dist_recip == 0):
         #    logger.error('dist_recip is zero')
         # logger.info('dist_recip.shape: %s',dist_recip.shape)
         norm = torch.sum(dist_recip, dim=2, keepdim=True)
         norm[(norm == 0).nonzero(as_tuple=True)] = 1e8
         # if torch.any(norm == 0):
         #    nz = (norm == 0).nonzero(as_tuple=True)
         #    logger.error('norm is zero. div-0 coming\n sum = %s\n nz = %s\n dr = %s\n d = %s\n x1 = %s\n x2 = %s',
         #       torch.sum(norm == 0),
         #       nz,
         #       dist_recip[nz[0],nz[1],:],
         #       dists[nz[0],nz[1],:],
         #       xyz1[nz[0],nz[1],:],
         #       xyz2[nz[0],nz[1],:])
         weight = dist_recip / norm
         interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
      
      # logger.info(f'interpolated_points.shape={interpolated_points.shape}')

      # if torch.any(torch.isnan(interpolated_points)):
      #   logger.error('interpolated_points is nan')

      if points1 is not None:
         points1 = points1.permute(0, 2, 1)
         new_points = torch.cat([points1, interpolated_points], dim=-1)
      else:
         new_points = interpolated_points

      # logger.info(f'1 new_points.shape={new_points.shape}')


      # if torch.any(torch.isnan(new_points)):
      #   logger.error('new_points is nan')

      new_points = new_points.permute(0, 2, 1)
      # logger.info(f'1 new_points.shape={new_points.shape}')
      for i, conv in enumerate(self.mlp_convs):
         bn = self.mlp_bns[i]
         new_points = F.relu(bn(conv(new_points)))
         # logger.info(f'1 new_points.shape={new_points.shape}')
      new_points = new_points.permute(0, 2, 1)

      # logger.info(f'2 new_points.shape={new_points.shape}')

      return new_points

   def to(self, memory_format):
      self.mlp_convs = self.mlp_convs.to(memory_format = memory_format)
      self.mlp_bns = self.mlp_bns.to(memory_format = memory_format)
      return self

