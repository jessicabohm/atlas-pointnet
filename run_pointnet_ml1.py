#!/usr/bin/env python3
import json,time,os
import numpy as np
import model
import torch
import CalcMean
import glob
import math
import csv

NUM_TRACK_POINTS = 19

# DATA GENERATORS
def batched_data_generator(file_names, batch_size, max_num_points, normalize_coords, isolate_padding):
   for file in file_names:
      point_net_data = np.load(file)
      cluster_data = point_net_data['X']

      # normalize with precomputed min and max of dataset
      min = {
         'x': -3626,
         'y': -3626,
         'z': -5906,
      }
      max = {
         'x': 3626,
         'y': 3626,
         'z': 5906,
      }

      # normalize x
      if normalize_coords:
         for coord_idx, coord in enumerate(['x', 'y', 'z']):
            cluster_data[:, :, coord_idx + 1] = (cluster_data[:, :, coord_idx + 1] - min[coord]) / (max[coord] - min[coord])

      Y = point_net_data['Y']

      # pad X data to have y dimension of max_num_points
      if isolate_padding:
         X_padded = np.full((cluster_data.shape[0], max_num_points, cluster_data.shape[2]), -100.0)
      else:
         X_padded = np.zeros((cluster_data.shape[0], max_num_points, cluster_data.shape[2]))

      Y_padded = np.negative(np.ones(((cluster_data.shape[0], max_num_points, 1)))) # NOTE: update for weighted cells
      
      tracks = np.zeros((cluster_data.shape[0], NUM_TRACK_POINTS, 3))
      
      for i, cluster in enumerate(cluster_data):
         X_padded[i, :len(cluster), 0:3] = cluster[:, 1:4] # feat 1-3 are x,y,z
         X_padded[i, :len(cluster), 3] = cluster[:, 0] # Energy
         X_padded[i, :len(cluster), 4] = cluster[:, 4] # track flag
         Y_padded[i, :len(cluster), :] = Y[i] # NOTE: perhaps add squeeze

         num_track_hits = len(cluster[:, 4][cluster[:, 4] == 1])
         tracks[i, :num_track_hits, :] = (cluster[:, 1:4][cluster[:, 4] == 1]).reshape((num_track_hits, 3))
         tracks[i, num_track_hits:] = np.tile(tracks[i, 0, :], NUM_TRACK_POINTS - num_track_hits).reshape(NUM_TRACK_POINTS - num_track_hits, 3)

      # convert Y_padded to mask
      mask = np.ones(((cluster_data.shape[0], max_num_points, 1)))
      mask[Y_padded == -1] = 0

      # split into batch_size groups of clusters
      for i in range(1, math.ceil(cluster_data.shape[0]/batch_size)):
         yield torch.from_numpy(X_padded[(i-1)*batch_size:i*batch_size]), torch.from_numpy(Y_padded[(i-1)*batch_size:i*batch_size]), torch.from_numpy(mask[(i-1)*batch_size:i*batch_size]), torch.from_numpy(tracks[(i-1)*batch_size:i*batch_size])

def get_accuracy(preds, labels, mask):
   return (((preds > 0.5).float() == labels).float()*mask).sum() / mask.sum()

def main():
   config_file = json.load(open("/home/jbohm/start_tf/atlas-pointnet/configs/pointnet2_test1.json"))

   # DATA AND OUTPUT DIRS
   os.environ["CUDA_VISIBLE_DEVICES"] = '3'
   data_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_tracks_cor'
   #output_dir = '/fast_scratch_1/jbohm/train_testing_data/pointnet_train_classify/pnet_part_seg_no_tnets_charged_events_thresh_0.787_tr_707_val_210_tst_10_lr_1e-6'
   output_dir = "/fast_scratch_1/jbohm/train_testing_data/pointnet_train_tracks_cor/pnet2msg_charged_events_all_one_track_thresh_0.787_tr_50_val_5_prenormalize_xyz_no_dp"
   # mkdirs if not present
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   if not os.path.exists(output_dir + "/states"):
      os.makedirs(output_dir + "/states")
   if not os.path.exists(output_dir + "/tests"):
      os.makedirs(output_dir + "/tests")

   num_train_files = 50 #707
   num_val_files = 5 #210
   events_per_file = 3800
   start_at_epoch = 0 # load start_at_epoch - 1
   validate_only = False
   validate_mixed = False
   normalize_coords = True
   isolate_padding = False

   EPOCHS = 100
   BATCH_SIZE = 32
   LEARNING_RATE = 1e-2

   train_output_dir = data_dir + '/train_1_track_larger/'
   val_output_dir = data_dir + '/test_1_track_larger/'#_mixed_thresh_0.787/'

   if validate_only and validate_mixed:
      val_output_dir = data_dir + '/test_mixed_thresh_0.787/'
      num_val_files = 10

   train_files = np.sort(glob.glob(train_output_dir+'*.npz'))[:num_train_files]
   val_files = np.sort(glob.glob(val_output_dir+'*.npz'))[:num_val_files]
   num_batches_train = (len(train_files) * events_per_file) / BATCH_SIZE 
   num_batches_val = (len(val_files) * events_per_file) / BATCH_SIZE

   print(val_files)

   # load the max number of points (N) - saved to data dir
   with open(data_dir + '/max_points_1_track.txt') as f:
      N = int(f.readline())

   net = model.get_model(config_file)

   if start_at_epoch:
      net.load_state_dict(torch.load(output_dir + "/states/state_" + str(start_at_epoch - 1) + ".pth"))

   print('model = \n %s',net)

   total_params = sum(p.numel() for p in net.parameters())
   print('trainable parameters: %s',total_params)

   train_model(net, BATCH_SIZE, EPOCHS, train_files, val_files, N, num_batches_train, num_batches_val, output_dir, start_at_epoch, validate_only, validate_mixed, normalize_coords, isolate_padding)

def bce_loss(preds, labels, weights):
   labels = torch.LongTensor(labels)
   criterion = torch.nn.CrossEntropyLoss()
   loss = criterion(preds, labels)*weights
   return loss

def train_model(model, batch_size, epochs, train_files, val_files, N, num_batches_train, num_batches_val, output_dir, start_at_epoch, validate_only, validate_mixed, normalize_coords, isolate_padding):
   status = math.ceil(num_batches_train / 200) # print status of train 100 times
   device = 'cuda'

   loss_func = torch.nn.BCELoss(reduction='none') 
   acc_func= get_accuracy

   opt = torch.optim.Adam(model.parameters())
   lrsched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2,4,6,10,14,18,20], gamma=0.5)

   model.to(device)
   model.double()

   
   for epoch in range(epochs):
      print()
      print()
      mean_train_loss = 0
      mean_train_acc = 0
      mean_val_loss = 0
      mean_val_acc = 0
      
      if not validate_only:
         # TRAIN EPOCH
         model.to(device)
         print("*************TRAIN EPOCH " + str(epoch + start_at_epoch) + "*************")
         train_data = batched_data_generator(train_files, batch_size, N, normalize_coords, isolate_padding)
         start_train_time = time.time()
         for batch_counter,(inputs,targets,nonzero_mask,tracks) in enumerate(train_data):           
            # move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            nonzero_mask = nonzero_mask.to(device)
            tracks = tracks.to(device)
            
            # zero grads
            opt.zero_grad()

            outputs,endpoints = model(inputs, tracks)

            loss_value = loss_func(outputs,targets)
            loss_value = torch.sum(loss_value * nonzero_mask)/torch.sum(nonzero_mask)
            
            # backward calc grads
            loss_value.backward()

            # apply grads
            opt.step()

            loss_value_cpu = float(loss_value.to('cpu'))

            # calc acc
            acc_value_cpu = float(acc_func(outputs,targets,nonzero_mask).to('cpu'))

            # add loss and acc to mean
            mean_train_loss += loss_value_cpu
            mean_train_acc += acc_value_cpu

            # print statistics
            if batch_counter % status == 0:
               print('epoch', epoch + start_at_epoch, 'of', + start_at_epoch + epochs, '[batch', batch_counter,'of', str(int(num_batches_train)) + '] - loss:', round(mean_train_loss/(batch_counter + 1), 4), 'acc:', round(mean_train_acc/(batch_counter + 1), 4))

            # release tensors for memory
            del inputs,targets,endpoints,loss_value
         train_time = time.time() - start_train_time

      print()
      print("*************VALIDATE*************")
      model.to(device)
      # every epoch, evaluate validation data set
      val_data = batched_data_generator(val_files, batch_size, N, normalize_coords, isolate_padding)
      preds = []
      labels = []
      start_val_time = time.time()
      with torch.no_grad():
         for valid_batch_counter,(inputs,targets,nonzero_mask,tracks) in enumerate(val_data):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            nonzero_mask = nonzero_mask.to(device)
            tracks = tracks.to(device)

            outputs,endpoints = model(inputs, tracks)
            
            loss_value = loss_func(outputs,targets)
            loss_value = torch.sum(loss_value * nonzero_mask)/torch.sum(nonzero_mask)
            loss_value_cpu = float(loss_value.to('cpu'))
            
            # calc acc
            acc_value_cpu = float(acc_func(outputs,targets,nonzero_mask).to('cpu'))

            # add loss and acc to mean
            mean_val_loss += loss_value_cpu
            mean_val_acc += acc_value_cpu

            preds.extend(outputs.to('cpu').numpy())
            labels.extend(targets.to('cpu').numpy())

            if valid_batch_counter % status == 0:
               print('epoch', epoch + start_at_epoch, 'of', + start_at_epoch + epochs, '[batch', valid_batch_counter,'of', str(int(num_batches_val)) + '] - loss:', round(mean_val_loss/(valid_batch_counter + 1), 4), 'acc:', round(mean_val_acc/(valid_batch_counter + 1), 4)) 
      val_time = time.time() - start_val_time
      # save preds/truth
      if validate_only and validate_mixed:
         np.save(output_dir + "/tests/mixed_preds_" + str(epoch + start_at_epoch - 1) + ".npy", preds)
      else:
         np.save(output_dir + "/tests/preds_" + str(epoch + start_at_epoch) + ".npy", preds)

      if epoch == 0 or validate_only:
         if validate_only and validate_mixed:
            np.save(output_dir + "/tests/mixed_labels.npy", labels)
         else:
            np.save(output_dir + "/tests/labels.npy", labels)

      if not validate_only:
         # save loss
         with open(output_dir + "/log_loss.csv" ,'a') as file:
            writer = csv.writer(file)
            writer.writerow([start_at_epoch + epoch , mean_train_loss / (batch_counter + 1), mean_val_loss / (valid_batch_counter + 1)])
         
         with open(output_dir + "/log_accuracy.csv" ,'a') as file:
            writer = csv.writer(file)
            writer.writerow([start_at_epoch + epoch , mean_train_acc / (batch_counter + 1), mean_val_acc / (valid_batch_counter + 1)])

         # save models state dict
         torch.save(model.state_dict(), output_dir + '/states/state_' + str(start_at_epoch + epoch) + '.pth')

         with open(output_dir + "/log_runtime.csv" ,'a') as file:
            writer = csv.writer(file)
            writer.writerow([start_at_epoch + epoch, train_time, val_time])
      
         # update learning rate
         lrsched.step()    

if __name__ == "__main__":
   main()
