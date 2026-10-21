"""持久事件层：RunEvent 序号分配与写入数据库(store)、open 登记与批量投递(hub)、
DB 行到 SSE 帧的投影(projector)、写入聚合(sinks)。
可丢弃的预览通道在 service.preview；两层互不 import。"""
