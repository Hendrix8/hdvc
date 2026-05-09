# -*- coding: UTF-8 -*-

# import psutil
import os


# 这个似乎不WORK...
# #改编自：阙赞: Python小技巧：利用Python如何实时检测自身内存占用
# # https://zhuanlan.zhihu.com/p/143437979
# # 获取当前进程内存占用。
# #待检查
# def get_current_memory_mb():
#     pid = os.getpid()
#     p = psutil.Process(pid)
#     info = p.memory_full_info()
#     return info.uss / 1024. / 1024.