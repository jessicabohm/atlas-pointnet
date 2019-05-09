#!/usr/bin/env python3
import argparse,logging,socket,json
import numpy as np
from data_handlers import utils as datautils
import torch
import tensorboardX

logger = logging.getLogger(__name__)
torch.set_printoptions(sci_mode=False,precision=3)


def main():
   ''' simple starter program that can be copied for use when starting a new script. '''

   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-c','--config_file',help='configuration file in json format',required=True)
   parser.add_argument('--num_files','-n', default=-1, type=int,
                       help='limit the number of files to process. default is all')
   parser.add_argument('--model_save',default='model_saves',help='base name of saved model parameters for later loading')
   parser.add_argument('--nsave',default=100,type=int,help='frequency in batch number to save model')

   parser.add_argument('--nval',default=100,type=int,help='frequency to evaluate validation sample in batch numbers')
   parser.add_argument('--nval_tests',default=1,type=int,help='number batches to test per validation run')

   parser.add_argument('--status',default=20,type=int,help='frequency to print loss status in batch numbers')

   parser.add_argument('--batch',default=-1,type=int,help='set batch size, overrides file config')

   parser.add_argument('--random_seed',default=0,type=int,help='numpy random seed')

   parser.add_argument('-i','--input_model_pars',help='if provided, the file will be used to fill the models state dict from a previous run.')
   parser.add_argument('-e','--epochs',type=int,default=-1,help='number of epochs')
   parser.add_argument('-l','--logdir',help='log directory for tensorboardx')

   parser.add_argument('--horovod',default=False, action='store_true', help="Setup for distributed training")

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(process)s:%(thread)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   log_level = logging.INFO

   if args.debug and not args.error and not args.warning:
      log_level = logging.DEBUG
   elif not args.debug and args.error and not args.warning:
      log_level = logging.ERROR
   elif not args.debug and not args.error and args.warning:
      log_level = logging.WARNING

   rank = 0
   nranks = 1
   hvd = None
   if args.horovod:
      print('importing horovod')
      import horovod.torch as hvd
      print('imported horovod')
      hvd.init()
      rank = hvd.rank()
      nranks = hvd.size()
      logging_format = '%(asctime)s %(levelname)s:' + '{:05d}'.format(rank) + ':%(name)s:%(process)s:%(thread)s:%(message)s'

   if rank > 0 and log_level == logging.INFO:
      log_level = logging.WARNING

   logging.basicConfig(level=log_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)

   logger.info('rank %s of %s',rank,nranks)
   logger.info('hostname:           %s',socket.gethostname())

   logger.info('config file:        %s',args.config_file)
   logger.info('num files:          %s',args.num_files)
   logger.info('model_save:         %s',args.model_save)
   logger.info('random_seed:        %s',args.random_seed)
   logger.info('nsave:              %s',args.nsave)
   logger.info('nval:               %s',args.nval)
   logger.info('nval_tests:         %s',args.nval_tests)
   logger.info('status:             %s',args.status)
   logger.info('input_model_pars:   %s',args.input_model_pars)
   logger.info('epochs:             %s',args.epochs)
   logger.info('horovod:            %s',args.horovod)
   logger.info('logdir:             %s',args.logdir)
   logger.info('num_threads:        %s',torch.get_num_threads())

   np.random.seed(args.random_seed)

   config_file = json.load(open(args.config_file))
   config_file['rank'] = rank
   config_file['nranks'] = nranks
   config_file['input_model_pars'] = args.input_model_pars
   config_file['horovod'] = args.horovod
   config_file['status'] = args.status
   config_file['nval'] = args.nval
   config_file['nval_tests'] = args.nval_tests
   config_file['nsave'] = args.nsave
   config_file['model_save'] = args.model_save

   if args.batch > 0:
      config_file['training']['batch_size'] = args.batch
   if args.epochs > 0:
      config_file['training']['epochs'] = args.epochs

   logger.info('configuration = \n%s',json.dumps(config_file, indent=4, sort_keys=True))

   if 'csv' in config_file['data_handling']['input_format']:
      logger.info('using CSV data handler')
      from data_handlers.csv import BatchGenerator
   else:
      raise Exception('no input file format specified in configuration')
   
   logger.info('getting filelists')
   trainlist,validlist = datautils.get_filelist(config_file)

   logger.info('creating batch generators')
   trainds = BatchGenerator(trainlist,config_file)
   validds = BatchGenerator(validlist,config_file)

   writer = None
   if args.logdir:
      writer = tensorboardX.SummaryWriter(log_dir=args.logdir)
   
   logger.info('building model')
   if 'pytorch' in config_file['model']['framework']:
      from pytorch import model,loss

      net = model.get_model(config_file)

      opt,lrsched = model.setup(net,hvd,config_file)

      lossfunc = loss.get_loss(config_file)
      accfunc = loss.get_accuracy(config_file)

      logger.info('model = \n %s',net)

      #total_params = sum(p.numel() for p in model.parameters())
      #logger.info('trainable parameters: %s',total_params)

      model.train_model(net,opt,lossfunc,accfunc,lrsched,trainds,validds,config_file,writer)
            

def print_module(module,input_shape,input_channels,name=None,indent=0):

   output_string = ''
   output_channels = input_channels
   output_shape = input_shape

   output_string += '%10s' % ('>' * indent)
   if name:
      output_string += ' %20s' % name
   else:
      output_string += ' %20s' % module.__class__.__name__

   # convolutions change channels
   if 'submanifoldconv' in module.__class__.__name__.lower():
      output_string += ' %4d -> %4d ' % (module.nIn,module.nOut)
      output_channels = module.nOut
   elif 'conv' in module.__class__.__name__.lower():
      output_string += ' %4d -> %4d ' % (module.in_channels,module.out_channels)
      output_channels = module.out_channels
   elif 'pool' in module.__class__.__name__.lower():
      output_shape = [int(input_shape[i] / module.pool_size[i]) for i in range(len(input_shape))]
      output_string += ' %10s -> %10s ' % (input_shape, output_shape)
   elif 'batchnormleakyrelu' in module.__class__.__name__.lower():
      output_string += ' (%10s) ' % module.nPlanes
   elif 'batchnorm2d' in module.__class__.__name__.lower():
      output_string += ' (%10s) ' % module.num_features

   output_string += '\n'

   for name, child in module.named_children():
      string,output_shape,output_channels = print_module(child, output_shape, output_channels, name, indent + 1)
      output_string += string

   return output_string,output_shape,output_channels


def summary(input_shape,input_channels,model):

   return print_module(model,input_shape,input_channels)


if __name__ == "__main__":
   main()