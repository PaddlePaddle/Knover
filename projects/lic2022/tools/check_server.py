# -*- encoding: utf-8 -*-
"""
 Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
@File    :   check_server.py
@Time    :   2022/06/20 17:24:41
@Author  :   zhouhan 
@Version :   1.0
@Contact :   zhouhan05@baidu.com
"""
import sys
import json
import os
import random
import time

import urllib
import requests

ip = ""
port = ""
LIC2022_API = "http://" + ip + ":" + port + "/api/lic2022"

def server_check(context, location, longitude, latitude, topical):
    """
    检查服务是否可用
    """
    try:
        req = {}
        req["context"] = context
        req["location"] = location
        req["longitude"] = longitude
        req["latitude"] = latitude
        req["topical"] = topical
        data = json.dumps(req)
        post_res_str = requests.post(LIC2022_API, data).json()
        reply = post_res_str["reply"]
        if isinstance(reply, str):
            return True
    except:
        return False
    
if __name__ == "__main__":
    context = ["你好", "你也好呀", "你叫什么名字"]
    location = "北京市海淀区"
    longitude = "116.279853"
    latitude = "40.049879"
    topical = ["学习生活", "考试", "六级"]
    return_flag = server_check(context, location, longitude, latitude, topical)
    if return_flag:
        print("check pass!!!")
    else:
        print("check fail!!!")