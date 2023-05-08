#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger
import imp
from common.logger import Logger
from common.avgmeter import *
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import *
from tasks.semantic.modules.segmentator import *
from tasks.semantic.modules.ioueval import *
from tasks.semantic.modules.SLAM_ERROR import *

from tasks.semantic.modules.trainer import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./train.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to train with. No Default',
  )
  parser.add_argument(
      '--arch_cfg', '-ac',
      type=str,
      required=True,
      help='Architecture yaml cfg file. See /config/arch for sample. No default!',
  )
  parser.add_argument(
      '--data_cfg', '-dc',
      type=str,
      required=False,
      default='config/labels/semantic-kitti.yaml',
      help='Classification yaml cfg file. See /config/labels for sample. No default!',
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      default=os.path.expanduser("~") + '/logs/' +
      datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
      help='Directory to put the log data. Default: ~/logs/date+time'
  )
  parser.add_argument(
      '--pretrained', '-p',
      type=str,
      required=False,
      default=None,
      help='Directory to get the pretrained model. If not passed, do from scratch!'
  )

  parser.add_argument('--init_method', default='', type=str, help='')
  parser.add_argument('--dist-backend', default='gloo', type=str, help='')
  parser.add_argument('--world_size', default=1, type=int, help='')
  parser.add_argument('--distributed', action='store_true', help='')
  parser.add_argument('--num_workers', default=8, type=int, help='')

  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("dataset", FLAGS.dataset)
  print("arch_cfg", FLAGS.arch_cfg)
  print("data_cfg", FLAGS.data_cfg)
  print("log", FLAGS.log)
  print("pretrained", FLAGS.pretrained)
  print("----------\n")
  print("Commit hash (training version): ", str(
      subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
  print("----------\n")

  # open arch config file
  try:
    print("Opening arch config file %s" % FLAGS.arch_cfg)
    ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

  # open data config file
  try:
    print("Opening data config file %s" % FLAGS.data_cfg)
    DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
  except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()

  # create log folder
  try:
    if os.path.isdir(FLAGS.log):
      shutil.rmtree(FLAGS.log)
    os.makedirs(FLAGS.log)
  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    # quit()

  # does model folder exist?
  if FLAGS.pretrained is not None:
    if os.path.isdir(FLAGS.pretrained):
      print("model folder exists! Using model from %s" % (FLAGS.pretrained))
    else:
      print("model folder doesnt exist! Start with random weights...")
  else:
    print("No pretrained directory found.")

  # copy all files to log folder (to remember what we did, and make inference
  # easier). Also, standardize name to be able to open it later
  try:
    print("Copying files to %s for further reference." % FLAGS.log)
    copyfile(FLAGS.arch_cfg, FLAGS.log + "/arch_cfg.yaml")
    copyfile(FLAGS.data_cfg, FLAGS.log + "/data_cfg.yaml")
  except Exception as e:
    print(e)
    print("Error copying files, check permissions. Exiting...")
    # quit()

class MyCallback(pl.Callback):
    def __init__(self,model=None):
        super().__init__()
        self.model=model
    def set_model(self, model):
        self.model = model
    def on_train_start(self,trainer,pl_module):
        self.best_train_iou = 0.0


        self.model.ignore_class = []
        for i, w in enumerate(self.model.loss_w):
            if w < 1e-10:
                self.model.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.model.parser.get_n_classes(),
                              self.model.ignore_class)

        # train for n epochs
        for epoch in range(self.model.ARCH["train"]["max_epochs"]):
        # get info for learn rate currently
            groups = self.model.optimizer.param_groups
            for name, g in zip(self.model.lr_group_names, groups):
                self.model.info[name] = g['lr']
    
    def on_train_epoch_start(self,trainer,pl_module):
        start = time.time()
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.acc = AverageMeter()
        self.iou = AverageMeter()
        self.update_ratio_meter = AverageMeter()
        self.model.model.train()
        self.end = time.time()
        counter = 0


    def on_train_epoch_end(self,trainer, pl_module):
        print("Epoch over!!")
      # update info
        acc1, iou, loss, update_mean=self.model.training_step_outputs
        self.model.info["train_update"] = update_mean
        self.model.info["train_loss"] = loss
        self.model.info["train_acc"] = acc1
        self.model.info["train_iou"] = iou

      




      # remember best iou and save checkpoint
        if iou > self.best_train_iou:
            print("Best mean iou in training set so far, save model!")
            self.best_train_iou = iou
            self.model.model_single.save_checkpoint(self.model.log, suffix="_train")
        self.model.training_step_outputs.clear()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.model.validation_step_outputs.clear()
        self.model.ignore_class = []
        self.evaluator = iouEval(self.model.parser.get_n_classes(),
                              self.model.ignore_class)
        self.best_val_iou = 0.0
        self.val_batch_time = AverageMeter()
        self.val_losses = AverageMeter()
        self.val_acc = AverageMeter()
        self.val_iou = AverageMeter()
        self.model.model.eval()
        self.evaluator.reset()
        self.val_end=time.time()
        
        

    def on_validation_epoch_end(self, trainer, pl_module):
        print(self.model.validation_step_outputs)
        acc, iou, loss = self.model.validation_step_outputs
        self.model.info["valid_loss"] = loss
        self.model.info["valid_acc"] = acc
        self.model.info["valid_iou"] = iou
        if iou > self.best_val_iou:
          print("Best mean iou in validation so far, save model!")
          print("*" * 80)
          self.best_val_iou = iou

          # save the weights!
          self.model.model_single.save_checkpoint(self.log, suffix="")
        self.model.validation_step_outputs.clear()
    






import sys
TRAIN_PATH = "../../"
DEPLOY_PATH = "../../../deploy"
sys.path.insert(0, TRAIN_PATH)
parserPath = os.path.join(TRAIN_PATH, "tasks", "semantic",  "dataset", DATA["name"], "parser.py")

my_callback=MyCallback()
model = BonnetalSeg(ARCH, DATA, FLAGS.dataset, FLAGS.log,my_callback, FLAGS.pretrained)
my_callback.set_model(model)
  # create trainer and start the training
trainer = pl.Trainer(devices=4, accelerator="gpu" ,max_epochs=100,callbacks=[my_callback])


parserModule = imp.load_source("parserModule", parserPath)
parser_load = parserModule.Parser(
                            root=FLAGS.dataset,
                            train_sequences=DATA["split"]["train"],
                            valid_sequences=DATA["split"]["valid"],
                            test_sequences=None,
                            labels=DATA["labels"],
                            color_map=DATA["color_map"],
                            learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],
                            max_points=ARCH["dataset"]["max_points"],
                            batch_size=ARCH["train"]["batch_size"],
                            workers=ARCH["train"]["workers"],
                            gt=True,
                            shuffle_train=False
                                      )

train_loader =  parser_load.get_train_set()
val_loader   =  parser_load.get_valid_set()
trainer.fit(model, train_loader, val_loader)

