"""进程装配：把各层接起来，只在启动时执行一次。

main 与 worker 是两个入口，形状对称：configure_logging → 换事件循环 → 跑。
layers 装配跨进程共用的基础设施，lifespan 在它之上装配 API 进程的运行期对象，
api 装配路由表。

本包不含业务规则。layers 与 lifespan 可被其他模块引用；main / worker / api 是入口，
只允许被 __main__ 与 console script 调用 —— 其他模块引用它们，意味着运行期要重新
装配一个进程。
"""
