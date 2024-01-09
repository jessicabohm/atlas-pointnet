#!/usr/bin/env python3
import json,time,os
import numpy as np
import model
import torch
import CalcMean
import glob
import math
import csv
from model.pointnet2 import PointNet2, PointNet2MSG

def main():
   # GPU
   os.environ["CUDA_VISIBLE_DEVICES"] = '0'

   # DATA AND OUTPUT DIRS
   data_dir = '/fast_scratch_1/jbohm/cell_particle_deposit_learning/train_dirs/pnet_train_1'
   output_dir = '/fast_scratch_1/jbohm/cell_particle_deposit_learning/train_dirs/pnet_train_1/pnet2_tr_50_val_5_tst_5_lr_1e-2_BS_32_npoint_75_50_tracks_4_rad_0.2_0.4_0.1_0.2_nsamp_100_30_20_19'#pnet2_msg_tr_50_val_5_tst_5_lr_1e-2_BS_32_rad_0.2_0.2_0.1_0.05_nsamp_150_100_75_50'
   max_points_file = 'max_points_1_track.txt'

   # SET DATA PARAMETERS
   num_train_files = 68 #707
   num_val_files = 5 #210
   events_per_file = 6000
   start_at_epoch = 0 # load start_at_epoch - 1
   normalize_coords = True
   isolate_padding = True
   include_tracks = True
   num_classes = 1
   num_features = 5

   # IF ONLY DOING PREDICTIONS
   validate_only = False
   validate_mixed = False

   # SET HYPERPARAMETERS
   EPOCHS = 100
   BATCH_SIZE = 32
   LEARNING_RATE = 1e-2

   """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
   
   train_output_dir = data_dir + '/train_1_track/'
   val_output_dir = data_dir + '/test_1_track/'

   if validate_only and validate_mixed:
      val_output_dir = data_dir + '/test_mixed_thresh_0.787/'
      num_val_files = 10

   # make model states and tests dirs if not present
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   if not os.path.exists(output_dir + "/states"):
      os.makedirs(output_dir + "/states")
   if not os.path.exists(output_dir + "/tests"):
      os.makedirs(output_dir + "/tests")

   # load list of file paths from data dir - glob finds all path in the folder, and sort sorts them by lexigraphical order (ie. ai_0_, ai_20_, ai_3_, pi_1_, pi_30, pi_9_)
   train_files = np.sort(glob.glob(train_output_dir+'*.npz'))[:num_train_files]
   val_files = np.sort(glob.glob(val_output_dir+'*.npz'))[:num_val_files]

   # estimate the num batches by (#files * #events per file) / # events per batch
   num_batches_train = (len(train_files) * events_per_file) / BATCH_SIZE 
   num_batches_val = (len(val_files) * events_per_file) / BATCH_SIZE

   # load the max number of points (N) - saved to data dir
   with open(data_dir + '/' + max_points_file) as f:
      N = int(f.readline())

   # load the model
   net = PointNet2(num_classes, num_features)

   # if passed an epoch other than 0 to start at - load the state of the partially trained model
   if start_at_epoch:
      net.load_state_dict(torch.load(output_dir + "/states/state_" + str(start_at_epoch - 1) + ".pth"))

   # print model structure and trainable parameters counts
   print('model = \n',net)

   total_params = sum(p.numel() for p in net.parameters())
   print('trainable parameters:',total_params)

   # call train model function
   train_model(net, BATCH_SIZE, EPOCHS, train_files, val_files, N, num_batches_train, num_batches_val, output_dir, start_at_epoch, validate_only, validate_mixed, normalize_coords, isolate_padding, include_tracks)

#########################################################################################################################################################################################################
#########################################################################################################################################################################################################

# DATA GENERATOR
def batched_data_generator(file_names, batch_size, max_num_points, normalize_coords, isolate_padding, include_tracks):
   NUM_TRACK_POINTS = 19

   for file in file_names:
      point_net_data = np.load(file)
      event_data = point_net_data['X']
      Y = point_net_data['Y']

      # normalize with precomputed min and max of x, y, z in the dataset
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

      # normalize coords to between 0 and 1
      if normalize_coords:
         for coord_idx, coord in enumerate(['x', 'y', 'z']):
            # use coord_idx + 1 since E_idx = 0, x_idx = 1, y_idx = 2, z_idx = 3
            event_data[:, :, coord_idx + 1] = (event_data[:, :, coord_idx + 1] - min[coord]) / (max[coord] - min[coord])

      # pad X data to have dimension 2 of max_num_points -> [num events, max num points, num features]
      num_features = 5 if include_tracks else 4
      if isolate_padding:
         # set the coords of each padded point to (100, 100, 100) & the energy to 100 & track flag to 100
         X_padded = np.full((event_data.shape[0], max_num_points, num_features), 100.0)
         X_padded[:, :, 0] = 0.0 # update padded energy to be 0
         if include_tracks:
            X_padded[:, :, 4] = 0.0 # update padded track flag to be 0
      else:
         # pad everything with 0s
         X_padded = np.zeros((event_data.shape[0], max_num_points, num_features))

      # pad the labels with -1 - flag that the point is padding or a track
      Y_padded = np.negative(np.ones(((event_data.shape[0], max_num_points, 1))))
      
      if include_tracks:
         # construct an array of tracks [# evetns, # track points, 3 coords]
         tracks = np.full((event_data.shape[0], NUM_TRACK_POINTS, 3), -100.0) # pad with -100's so the padding is never interpolated to a real track value
      
      # for each event copy cell values into padded array
      for event_idx, event in enumerate(event_data):
         X_padded[event_idx, :len(event), 0:3] = event[:, 1:4] # feat indexed 1-3 of event are x, y, z - copy to first 3 indicies of cell data in point cloud
         X_padded[event_idx, :len(event), 3] = event[:, 0] # put energy as 4th index
         if include_tracks:
            X_padded[event_idx, :len(event), 4] = event[:, 4] # put track flag as 5th index
         Y_padded[event_idx, :len(event), :] = Y[event_idx] # copy truth labels into padded array (tracks are already padded w -1s)

         if include_tracks:
            # num track hits = num track bools set to True
            num_track_hits = len(event[:, 4][event[:, 4] == 1])
            # copy track coords from event array into the tracks array
            tracks[event_idx, :num_track_hits, :] = (event[:, 1:4][event[:, 4] == 1]).reshape((num_track_hits, 3))
            # replace any missing track coords with the first track coord -> when grouping around tracks it uses these points NOTE: potential error - extrapolating out to the same point 3 times :/
            #tracks[event_idx, num_track_hits:] = np.tile(tracks[event_idx, 0, :], NUM_TRACK_POINTS - num_track_hits).reshape(NUM_TRACK_POINTS - num_track_hits, 3) # don't pad tracks with other track points anymore

      # convert Y_padded to mask - 1's where there is a measuremnt and 0 where it's just padding
      mask = np.ones(((event_data.shape[0], max_num_points, 1)))
      mask[Y_padded == -1] = 0

      # split into batch_size groups of events
      for i in range(1, math.ceil(event_data.shape[0]/batch_size)):
         if include_tracks:
            yield torch.from_numpy(X_padded[(i-1)*batch_size:i*batch_size]), torch.from_numpy(Y_padded[(i-1)*batch_size:i*batch_size]), torch.from_numpy(mask[(i-1)*batch_size:i*batch_size]), torch.from_numpy(tracks[(i-1)*batch_size:i*batch_size])
         else:
            yield torch.from_numpy(X_padded[(i-1)*batch_size:i*batch_size]), torch.from_numpy(Y_padded[(i-1)*batch_size:i*batch_size]), torch.from_numpy(mask[(i-1)*batch_size:i*batch_size]), []

# masked out accuracy function - using prob threshold of 0.5 
def get_accuracy(preds, labels, mask):
   return (((preds > 0.5).float() == labels).float()*mask).sum() / mask.sum()

def train_model(model, batch_size, epochs, train_files, val_files, N, num_batches_train, num_batches_val, output_dir, start_at_epoch, validate_only, validate_mixed, normalize_coords, isolate_padding, include_tracks):
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

      sum_train_loss = 0
      sum_train_acc = 0
      sum_val_loss = 0
      sum_val_acc = 0
      
      if not validate_only:
         # TRAIN EPOCH
         print("*************TRAIN EPOCH " + str(epoch + start_at_epoch) + "*************")
         model.to(device)
         train_data = batched_data_generator(train_files, batch_size, N, normalize_coords, isolate_padding, include_tracks)
         start_train_time = time.time()
         
         for batch_counter, (inputs,targets,nonzero_mask,tracks) in enumerate(train_data):           
            # move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            nonzero_mask = nonzero_mask.to(device)
            if include_tracks:
               tracks = tracks.to(device)
            
            # zero grads
            opt.zero_grad()

            outputs,endpoints = model(inputs, tracks)

            loss_value = loss_func(outputs,targets)
            loss_value = torch.sum(loss_value * nonzero_mask, dim=None)/torch.sum(nonzero_mask, dim=None)
            
            # backward calc grads
            loss_value.backward()

            # apply grads
            opt.step()

            loss_value_cpu = float(loss_value.to('cpu'))

            # calc acc
            acc_value_cpu = float(acc_func(outputs,targets,nonzero_mask).to('cpu'))

            # add loss and acc to mean
            sum_train_loss += loss_value_cpu
            sum_train_acc += acc_value_cpu

            # print statistics
            if batch_counter % status == 0:
               print('epoch', epoch + start_at_epoch, 'of', + start_at_epoch + epochs, '[batch', batch_counter,'of', str(int(num_batches_train)) + '] - loss:', round(sum_train_loss/(batch_counter + 1), 4), 'acc:', round(sum_train_acc/(batch_counter + 1), 4))

            # release tensors for memory
            del inputs,targets,endpoints,loss_value
         train_time = time.time() - start_train_time

      print()
      print("*************VALIDATE*************")
      model.to(device)
      # every epoch, evaluate validation data set
      val_data = batched_data_generator(val_files, batch_size, N, normalize_coords, isolate_padding, include_tracks)
      preds = []
      labels = []
      start_val_time = time.time()
      with torch.no_grad():
         for valid_batch_counter,(inputs,targets,nonzero_mask,tracks) in enumerate(val_data):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            nonzero_mask = nonzero_mask.to(device)
            if include_tracks:
               tracks = tracks.to(device)

            outputs,endpoints = model(inputs, tracks)
            
            # use default BCE loss to get the pointwise loss [BS, max num points, 1]
            loss_value = loss_func(outputs,targets)
            # mask out the padded values
            loss_value = torch.sum(loss_value * nonzero_mask)/torch.sum(nonzero_mask)
            loss_value_cpu = float(loss_value.to('cpu'))
            
            # calc acc
            acc_value_cpu = float(acc_func(outputs,targets,nonzero_mask).to('cpu'))

            # add loss and acc to mean
            sum_val_loss += loss_value_cpu
            sum_val_acc += acc_value_cpu

            preds.extend(outputs.to('cpu').numpy())
            labels.extend(targets.to('cpu').numpy())

            if valid_batch_counter % status == 0:
               print('epoch', epoch + start_at_epoch, 'of', + start_at_epoch + epochs, '[batch', valid_batch_counter,'of', str(int(num_batches_val)) + '] - loss:', round(sum_val_loss/(valid_batch_counter + 1), 4), 'acc:', round(sum_val_acc/(valid_batch_counter + 1), 4)) 
      
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
            writer.writerow([start_at_epoch + epoch , sum_train_loss / (batch_counter + 1), sum_val_loss / (valid_batch_counter + 1)])
         
         with open(output_dir + "/log_accuracy.csv" ,'a') as file:
            writer = csv.writer(file)
            writer.writerow([start_at_epoch + epoch , sum_train_acc / (batch_counter + 1), sum_val_acc / (valid_batch_counter + 1)])

         # save models state dict
         torch.save(model.state_dict(), output_dir + '/states/state_' + str(start_at_epoch + epoch) + '.pth')

         with open(output_dir + "/log_runtime.csv" ,'a') as file:
            writer = csv.writer(file)
            writer.writerow([start_at_epoch + epoch, train_time, val_time])
      
         # update learning rate
         lrsched.step()    

if __name__ == "__main__":
   main()
