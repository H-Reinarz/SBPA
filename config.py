# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 18:40:16 2017

@author: Jannik
"""

import configparser

class dict_object(object):
    '''Object for ini'''


    def __init__(self, d):
        self.__dict__ = d



def get_params(ini):
    # https://wiki.python.org/moin/ConfigParserExamples
    config = configparser.ConfigParser()
    config.read(ini)
    config.sections()

    paramsDict = {}
    for section in config.sections():
        dict1 = {}
        options = config.options(section)
        for option in options:
            try:
                dict1[option] = eval(config.get(section, option))
                if dict1[option] == -1:
                    print("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None

        paramsDict[section] = dict_object(dict1)

    Params = type("Params", (), paramsDict)
    p = Params()
    return p
