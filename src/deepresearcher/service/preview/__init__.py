"""可丢弃预览面:token 级 text_delta 帧的传输协议与两个实现。

与持久事件面(service.events)的分工是本包的合同:无序号、不落库、
丢失可容忍、永不参与 run 状态判定;帧形状由 protocol.preview_event 白名单
重建。两实现按部署形态在装配期选定:redis(跨进程,生产)与 local
(同事件循环,单栈 harness)。生产代码不得 import local——守卫钉住。
"""
