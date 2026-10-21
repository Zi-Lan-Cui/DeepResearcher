"""可丢弃的预览通道：token 级 text_delta 帧的传输协议与两个实现。

无序号、不写入数据库、丢失可容忍、不参与 run 状态判定。装配期按部署形态选定
redis（跨进程，生产）或 local（同事件循环，harness）；生产代码不得 import local。
"""
