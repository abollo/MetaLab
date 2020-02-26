#!/usr/bin/python
# coding: utf-8

import abc


class Detector(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def start(self):
      """
      Start Detection Service
      :return: Dict with the following keys
            success: (bool)
            errorMsg (string)
      """
      return

  @abc.abstractmethod
  def stop(self):
      """
      Stop Detection Service, tear up all processes
      :return: Dict with the following keys
            success: (bool)
            errorMsg (string)
      """

  @abc.abstractmethod
  def accept_task(self):
      """
      Returns whether the detector can accept new tasks
      :return: (bool)
      """

  @abc.abstractmethod
  def add_task(self, taskDict):
      """
      :param inputDict: a dictionary with the following fields
                          'img' (np array)
                          'requestID' (str)
                          'apiname' (str)
                          'filename' (str) optional filename on OSS
                          'localFileName' (str) optional filename on local file system
                          'arriveTime' (float) optional arrive time of file
      :return: Dict with the following keys
            success (bool)
            errorMsg (string)
      """

  @abc.abstractmethod
  def get_results(self):
      """
      Return a list of dicts
      :return: (list) resultList, each entry is a python dict with the following fields
            'requestID' (str)
            'apiname' (str)
            'faces' (for face detection api)
            'bodies' (for body detection api)
            ...
      """