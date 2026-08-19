# internal：模块共享地图

`minillm_internal.h` 不是公开 API，而是所有 `.c` 模块共享的数据结构和内部函数
声明。先看每个字段旁的形状注释，不必一次理解全部原型。

外部程序应只包含 `include/minillm.h`；internal 目录可以自由随实现演进。
