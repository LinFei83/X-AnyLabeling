# 图像级分类示例

## 简介

**图像分类**是为整个图像分配特定标签或类别的任务，可以分为不同的类别，即多类分类和多标签分类。

<img src=".data/classification.png" width="100%" />

> **多类分类**专注于将数据分类为两个以上的类别。

<img src=".data/annotated_multiclass_example.png" width="100%" />

> **多标签分类**允许每个实例同时关联多个标签，能够为单个图像分配多个二进制标签。

<img src=".data/annotated_multilabel_example.png" width="100%" />

## 使用方法

### GUI 导入

**步骤 0：准备工作**

准备一个标志文件，如 [logo_flags.txt](./logo_flags.txt) 或 [fruit_flags.txt](./fruit_flags.txt)。示例如下：

```txt
Apple
Meta
Google
```

**步骤 1：运行应用程序**

```bash
python anylabeling/app.py
```

**步骤 2：上传配置文件**

点击顶部菜单栏中的 `上传 -> 上传图像标志文件`，选择准备好的配置文件进行上传。

### 命令行加载

**选项 1：快速启动**

> [!TIP]
> 此选项适用于快速启动。

```bash
python anylabeling/app.py --flags Apple,Meta,Google
```

> [!CAUTION]
> 记住用逗号 ',' 分隔自定义标签。

**选项 2：使用配置文件**

```bash
python anylabeling/app.py --flags flags.txt
```

> [!NOTE]
> 这里，每一行代表一个类别。


详细的输出示例，请参考[此文件夹](./sources/)。
