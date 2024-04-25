<p align="center">

![jiemian](caches\jiemian.png)</p>
<h1 align="center" style="margin: 30px 0 30px; font-weight: bold;">大模型对话知识库v0.0.1</h1>
<h4 align="center">基于Dash+langchain开发的接入ai大模型api的知识库对话</h4>



## 应用简介

学习了datawhale的开源学习教程动手学大模型开发，一直在关注和使用各种ai工具，也在学一些ai开发的知识，看到这个教程比较全面，正好可以结合自己常用的数据平台开发工具dash来开发一个web界面，做了个简单的demo，可以本地调用各种大模型api来和自己的文本知识库来对话。

- web应用开发主要采用Dash(基于flask和react)、feffery-antd-components(feffery老师封装的的ant design组件库)等，可以纯python迅速开发。

- 后端大模型开发用langchain和chroma向量数据库，主要运用https://github.com/datawhalechina/llm-universe这个开源教程的相关知识。

  ​

## 主要功能

1. 选择大模型厂商，填写key和模型等参数，现在支持智谱，openai，讯飞星火，百度文心这几家的模型，默认用的智谱glm4。
2. 可以直接选择和不同的大模型对话或者和自己上传文本建立的知识库对话。
3. 知识库对话需要先上传pdf或者md文件上传并向量入库。
4. 知识库对话可以自己填写提示词来不断迭代提示词，取得更好的效果。
5. 对话界面可选支持单轮和带历史记录多轮对话，默认单轮对话。
6. 可以保存当前界面的全部历史聊天记录为md并下载。
7. 可以一键清除当前界面全部历史聊天记录。
8. 可以选择不同厂商的embedding模型，现在有智谱和openai的。

   ​

## 后续计划

- 支持更多家的对话模型和embedding模型
- 支持上传更多种类的文本文件上传
- 支持更多功能和更多可选参数的设置
- 学习后续进阶教程，优化RAG检索和angent等

## 项目运行

```bash
# 新建虚拟环境
conda create -n zsk python=3.11  -y

# 进入环境
conda activate zsk

# 克隆项目
git clone https://github.com/qibaizhang/chat_kb_app.git

# 进入项目根目录
cd chat_kb_app

# 安装项目依赖环境
pip install -r requirements.txt

# 运行项目
python wsgi.py

#浏览器打开http://127.0.0.1:8055/就可以看到界面,如果要局域网访问，把localhost改成0.0.0.0
```

### 开发

#### 可以自己修改源代码执行python app.py进行调试





