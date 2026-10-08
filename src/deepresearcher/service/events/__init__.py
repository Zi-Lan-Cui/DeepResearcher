"""持久事件面:RunEvent 序号分配与落库(store)、open 登记与批量投递(hub)、
DB 行到 SSE 帧的投影(projector)、写入聚合(sinks)。
可丢弃预览面在 service.preview;两平面互不 import。"""
