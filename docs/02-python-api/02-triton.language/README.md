---
title: triton.language
---

## 编程模式

| tensor |表示一个值或指针的 N 维数组。|
|:----|:----|
| tensor_descriptor |表示全局内存中一个张量的描述符。|
| program_id |沿着给定的 `axis` 返回当前程序实例的 ID。|
| num_programs |返回沿着指定 `axis` 启动的程序实例的数量。|
| map_elementwise |将 1 个标量函数映射到张量上。|

## 创建操作

| arange |返回半开区间 `[start, end)` 内的连续值。`end - start` 必须小于等于 `TRITON_MAX_TENSOR_NUMEL = 1048576`。|
|:----|:----|
| cat |连接给定的块。|
| full |返回一个张量，该张量填充了指定 `shape` 和 `dtype` 的标量值。|
| zeros |对于给定的`shape` 和 `dtype`，返回 1 个填充了标量值 0 的张量。|
| zeros_like |返回 1 个形状和类型与给定张量相同的全零张量。|
| cast |将张量转换为指定的 `dtype`。|
| to_tensor |把一个 Python 标量转换为 0 维的 `tensor`。|

## 形状变换操作

| broadcast |尝试将 2 个给定的块广播到 1 个共同兼容的形状。|
|:----|:----|
| broadcast_to |尝试将给定的张量广播到新的 `shape`。|
| expand_dims |通过插入新的长度为 1 的维度来扩展张量的形状。|
| interleave |沿着最后 1 个维度交错 2 个张量的值。这 2 个张量必须有相同形状。等同于 *tl.join(a, b).reshape(a.shape[-1:] + [2*****a.shape[-1]])**。*|
| join |在 1 个新的次要维度中连接给定的张量。|
| permute |排列张量的维度。|
| ravel |返回 `x` 的连续扁平视图。|
| reshape |返回一个具有与输入相同元素数量，但具有所提供形状的张量。|
| split |将张量沿着其最后 1 个维度分成 2 部分，该维度的大小必须为 2。|
| squeeze |从 `x` 中移除索引 `dim` 处长度为 1 的维度。|
| trans |置换张量的维度。|
| unsqueeze |在索引 `dim` 处向 `x` 插入一个长度为 1 的维度。|
| view |返回与 `input` 具有相同元素但形状不同的张量。元素的顺序会保持不变。|

## 线性代数操作

| dot |返回 2 个块的矩阵乘积。|
|:----|:----|
| dot_scaled |返回 2 个采用微缩放（microscaling）格式的块的矩阵乘积。|

## 内存/指针操作

| load |返回 1 个数据张量，其值从由 `pointer` 所定义的内存位置处加载：|
|:----|:----|
| store |将数据张量存储到由 `pointer` 定义的内存位置。|
| make_tensor_descriptor |创建一个张量描述符（tensor descriptor）对象。|
| load_tensor_descriptor |从张量描述符加载 1 个块的数据。|
| store_tensor_descriptor |将 1 个块的数据存储到张量描述符。|

## 索引操作

| flip |沿维度 `dim` 翻转张量 `x`。|
|:----|:----|
| where |根据 `condition` 返回由 `x` 或 `y` 中的元素组成的张量。|
| swizzle2d |将行主序的 *size_i* size_j 矩阵的索引转换为每组 size_g* 行的列主序矩阵的索引。|

## 数学操作

| abs |计算 `x` 的逐元素绝对值。|
|:----|:----|
| add |计算 `x` 与 `y` 的逐元素和。|
| cdiv |计算 `x` 除以 `div` 的向上取整除法（ceiling division）。|
| ceil |计算 `x` 的逐元素向上取整结果。|
| clamp |将输入张量 `x` 钳制在 [min, max] 范围内。当 `min` > `max` 时，行为未定义。|
| clmul |计算 `x` 与 `y` 的逐元素无进位乘法（carry-less multiplication）。|
| cos |计算 `x` 的逐元素余弦值。|
| div_rn |计算 `x` 和 `y` 的逐元素精确除法（根据 IEEE 标准四舍五入到最近的值）。|
| erf |计算 `x` 的逐元素误差函数。|
| exp |计算 `x` 的逐元素指数函数。|
| exp2 |计算 `x` 的逐元素指数函数（以 2 为底）。|
| fdiv |计算 `x` 除以 `y` 的逐元素除法。|
| floor |计算 `x` 的逐元素向下取整。|
| fma |计算 `x` 、 `y` 和 `z` 的逐元素融合乘加运算。|
| log |计算 `x` 的逐元素自然对数。|
| log2 |计算 `x` 的逐元素对数（以 2 为底）。|
| maximum |计算 `x` 与 `y` 的逐元素最大值。|
| minimum |计算 `x` 与 `y` 的逐元素最小值。|
| mul |计算 `x` 与 `y` 的逐元素乘积。|
| rsqrt |计算 `x` 的逐元素倒数平方根。|
| sigmoid |计算 `x` 的逐元素 sigmoid 函数值。|
| sin |计算 `x` 的逐元素正弦值。|
| softmax |沿给定维度对 `x` 计算 softmax。|
| sqrt |计算 `x` 的逐元素快速平方根。|
| sqrt_rn |计算 `x` 的逐元素精确平方根（根据 IEEE 标准四舍五入到最近值）。|
| sub |计算 `x` 与 `y` 的逐元素差。|
| umulhi |计算 `x` 和 `y` 的 2N 位乘积中逐元素的最高有效 N 位。|

## 归约操作

| all |返回 `input` 中的元素沿指定 `axis` 是否全部为非零值。|
|:----|:----|
| any |返回 `input` 中的元素沿指定 `axis` 是否存在非零值。|
| argmax |返回 `input` 张量沿指定 `axis` 的所有元素的最大索引。|
| argmin |返回 `input` 张量沿指定 `axis` 的所有元素的最小索引。|
| max |返回 `input` 张量沿指定 `axis` 的所有元素的最大值。|
| min |返回 `input` 张量沿指定 `axis` 的所有元素的最小值。|
| reduce |将 combine_fn 应用于 `input` 中各张量沿指定 `axis` 的所有元素。|
| reduce_or |返回 `input` 张量沿指定 `axis` 对所有元素做 reduce_or 的结果。|
| sum |返回 `input` 张量沿指定 `axis` 的所有元素的总和。|
| xor_sum |返回 `input` 张量沿指定 `axis` 的所有元素的异或和。|

## 扫描/排序操作

| associative_scan |沿指定 `axis` 将 combine_fn 应用于 `input` 张量的每个元素和携带的值，并更新携带的值。|
|:----|:----|
| cumprod |返回沿指定 `axis` 的 `input` 张量中所有元素的累积乘积。|
| cumsum |返回沿指定 `axis` 的 `input` 张量中所有元素的累积和。|
| histogram |基于 `input` 张量计算 1 个具有 num_bins 个 bin 的直方图，每个 bin 宽度为 1，起始于 0。|
| sort |沿着指定维度对张量进行排序。|
| topk |返回输入张量沿指定维度的 k 个最大（或最小）元素。|
| gather |沿指定维度从张量中收集（gather）元素。|

## 原子操作

| atomic_add |在由 `pointer` 指定的内存位置执行原子加法。|
|:----|:----|
| atomic_and |在由 `pointer` 指定的内存位置执行原子逻辑与操作。|
| atomic_cas |在由 `pointer` 指定的内存位置执行原子比较和交换操作。|
| atomic_load |从由 `pointer` 指定的内存位置原子地加载数据。|
| atomic_max |在由 `pointer` 指定的内存位置执行原子最大值操作。|
| atomic_min |在由 `pointer` 指定的内存位置执行原子最小值操作。|
| atomic_or |在由 `pointer` 指定的内存位置执行原子逻辑或操作。|
| atomic_poll |等待 `pointer` 处的值等于 `expected_value`。|
| atomic_store |将 `val` 原子地存储到由 `pointer` 指定的内存位置。|
| atomic_xchg |在由 `pointer` 指定的内存位置执行原子交换操作。|
| atomic_xor |在由 `pointer` 指定的内存位置执行原子逻辑异或操作。|

## 随机数生成

| randint4x |给定 1 个 `seed` 标量和 1 个 `offset` 块，返回 4 个 `int32` 类型的随机块。|
|:----|:----|
| randint |给定 1 个 `seed` 标量和 1 个 `offset` 块，返回 1 个 `int32` 类型的随机块。|
| rand |给定 1 个 `seed` 标量和 1 个 `offset` 块，返回 1 个在 $$U(0,1)$$ 中的 float32 类型的随机块。|
| rand4x |给定 1 个 `seed` 标量和 1 个 `offsets` 块，返回 4 个在 $$U(0, 1)$$ 中的随机 `float32` 块。|
| randn |给定 1 个 `seed` 标量和 1 个 `offset` 块，返回 1 个在 $$\mathcal{N}(0,1)$$ 中的 float32 类型的随机块。|
| randn4x |给定 1 个 `seed` 标量和 1 个 `offset` 块，返回 4 个在 $$\mathcal{N}(0, 1)$$ 中的随机 `float32` 块。|

## 迭代器

| range |永远向上计数的迭代器 。|
|:----|:----|
| static_range |永远向上计数的迭代器。|

## 内联汇编

| inline_asm_elementwise |在张量上执行内联汇编。本质上，这是一个映射，其中函数体是内联汇编。|
|:----|:----|

## 编译器提示操作

| assume |允许编译器假定 `cond` 为 True。|
|:----|:----|
| debug_barrier |插入一个屏障以同步块中的所有线程。|
| max_constancy |告知编译器：`input` 沿维度 `d` 的元素构成长度为 `values[d]` 的常量组。`values` 必须为 `input` 的每个维度各提供 1 个条目，且每个条目都必须是 2 的幂。|
| max_contiguous |告知编译器：`input` 沿维度 `d` 的元素构成长度为 `values[d]` 的连续组。`values` 必须为 `input` 的每个维度各提供 1 个条目，且每个条目都必须是 2 的幂。|
| multiple_of |告知编译器：`values[d]` 是能整除 `input` 沿维度 `d` 的每个连续组的第 1 个元素的、最大的 2 的幂（连续组的定义见 `max_contiguous`）。`values` 必须为 `input` 的每个维度各提供|

## 浮点检查（FpSan）

| fpsan.barrier |把一个中间表达式标记为对浮点检查（floating-point sanitization）有意义。|
|:----|:----|
| fpsan.homomorphic_cast |转换到同宽度或更窄的浮点类型，并保留 FpSan 的载荷代数。|

## 调试操作

| static_print |在编译时打印数值。 参数与内置的 `print` 相同。|
|:----|:----|
| static_assert |在编译时断言条件。 无需设置 `TRITON_DEBUG` 环境变量。|
| device_print |在运行时从设备端打印数值。字符串格式化对运行时的值不起作用，因此你要把想打印的值作为参数提供。第 1 个值必须是字符串，其后的所有值必须是标量或张量。|
| device_assert |在运行时从设备端断言该条件。要让这个断言生效，需要把环境变量 `TRITON_DEBUG` 设置为 `0` 以外的值。|
| expect_zero |标记那些预期已经下溢为 0 的值。|
