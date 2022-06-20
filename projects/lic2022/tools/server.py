# -*- encoding: utf-8 -*-
"""
 Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
@File    :   server.py
@Time    :   2022/06/20 09:46:34
@Author  :   zhouhan 
@Version :   1.0
@Contact :   zhouhan05@baidu.com
"""

import json
import os
import random
import time

import flask
import urllib
import requests

KG_SEARCH_API = ""

def generate_query(context, location):
    """
    生成query的方法
    一般接收context与location等信息，判断是否需要查询知识，如果需要查询生成查询query
    """
    pass

def kg_search_client(query, longitude, latitude):
    """
    请求知识查询服务
    接收查询query，用户所处的地理位置经纬度信息（服务端每次会同步传参）
    返回查询到的知识
    """
    req = {}
    req["query"] = query.replace(" ", "")
    req["longitude"] = longitude
    req["latitude"] = latitude
    data = json.dumps(req)
    post_res_str = requests.post(KG_SEARCH_API, data).json()
    knowledge = post_res_str["reply"]
    return knowledge
    
def generate_resposne(context, knowledge):
    """
    生成resposne的方法
    一般接收context、knowledge等信息
    返回最终的回复
    """
    pass

@app.route("/api/lic2022", methods=["POST"])
def conv():
    req = flask.request.get_json(force=True)
    context = req["context"] # 多轮对话上下文
    location = req["location"] # 用户所处的地理位置
    longitude = req["longitude"] # 地理位置对应的经度，可用于请求个性化知识
    latitude = req["latitude"] # 地理位置对应的纬度，可用于请求个性化知识
    topical = req["topical"] # 当前对话的初始话题（可选择性使用）
    query = generate_query(context, location)
    knowledge = kg_search_client(query, longitude, latitude)
    response = generate_resposne(context, knowledge)
    ret = {
        "name": "lic2022", # 返回结果中应包含该字段，无需修改
        "reply": response.replace(" ", "")  # 最终回复的结果，字符串格式，无需保留分词
    }
    return flask.jsonify(ret)

app.run(host="0.0.0.0", port=8312, debug=False) # port可自己指定，将部署好的服务地址提供给主办方即可