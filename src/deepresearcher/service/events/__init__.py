"""持久事件面:RunEvent 序号分配与落库(store)、席位与投递闸口(hub)、
DB 行到 SSE 帧的安全投影(projector)、写入聚合(sinks)。
可丢弃预览面在 service.preview;两平面互不 import。"""
