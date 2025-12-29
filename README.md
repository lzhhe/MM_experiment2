# M502083B 多模态机器学习大作业
- 25120399
- 刘子赫
- 项目链接 `https://github.com/lzhhe/MM_experiment2.git` 
## 项目简介与核心功能列表
主要实现多模态的文献加入与查找功能，主要包括以下几个
- 添加单个文献（可提供/不提供topic）
- 扫描文件夹当中文献（可提供/不提供topic）
- 自然语言搜索相关文献
- 自然语言搜索相关文献当中相关片段和页码
- 自然语言搜索相关文献当中的图片

## 环境配置与依赖安装说明
### 硬件配置
本地电脑配置为i7-11800H，显卡为3060ti laptop，显存6G，属于是算力不太足的计算卡，因此只使用了较小的模型完成这次作业
### 软件配置
1. 请使用pip安装全部所需库 
- `pip install -r requirements.txt` 
2. 请手动拉取需要的本地LLM大模型qwen3:4b
- `ollama pull qwen3:4b`
3. 为了保证本地性，所有的权重都进行了预下载放在了项目目录下，但是由于github有大小限制（所有权重大小为884Mb左右，主要是ViT32b和Florence较大），请从谷歌云盘手动下载需要使用的模型，放在models文件夹中
- 链接为`https://drive.google.com/file/d/1sOIJD3k_gvGeWSB3h3xHhf35aAVPpcDO/view?usp=sharing`
- models文件夹结构如下：
```
├─models
│  ├─all-MiniLM-L6-v2
│  │  └─1_Pooling
│  ├─clip
│  └─Florence-2-base
│      ├─.no_exist
│      │  └─5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac
│      ├─blobs
│      ├─refs
│      └─snapshots
│          └─5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac
```
- 需要手动创建一个models文件夹，然后将下载的压缩包解压到models当中，最终结构如上
## 详细的使用说明（包含具体的命令行示例）
有演示视频可以观看：链接如下：
`https://drive.google.com/file/d/1K3qZ-inGlBO65LNwdV8FQTMBMPHsEMos/view?usp=sharing`
- 添加单个文献（可提供/不提供topic）
- `python main.py add_paper <path> --topics "Topic1,Topic2"`
- 例如 `python main.py add_paper "./test_pdfs/paper.pdf" --topics "CV,NLP"`

- 扫描文件夹当中文献（可提供/不提供topic）
- `python main.py bulk_add <path>` 
- 例如 `python main.py bulk_add ./test_pdfs` 

- 自然语言搜索相关文献
- `python main.py search_paper <query>`
- 例如 `python main.py search_paper "What is the core idea of collaborative perception?"`

- 自然语言搜索相关文献当中相关片段和页码
- `python main.py search_content <query>`
- 例如 `python main.py search_content "What is the core idea of collaborative perception?"`

- 自然语言搜索相关文献当中的图片
- `python main.py search_image <query>`
- 例如 `python main.py search_image "traffic scenes with multiple vehicles"`

## 技术选型说明（使用了哪些模型、数据库等）
主要使用了以下的几个模型

- all-MiniLM-L6-v2
  - 轻量化模型，用于将提取到的文献纯文本内容转化为向量输入到chromaDB当中

- clip (ViT-B-32)
  - 用于将提取到的文献内图片转化为向量输入到chromaDB当中

- Florence-2-base
  - 用于图片内容识别和OCR文本识别，主要是为了给图片添加图注，描述和提取图片内文字，便于支持文搜图功能

- qwen3:4b
  - 本地大语言模型，正好会吃满在本地cuda和显存，用于文献的文本的分类功能，负责提供标签

## 文件结构
这里的文件结构为运行之后的，在github原始版本是不会有outputs和papers文件夹，在运行一次相关添加等操作之后会自动生成，这里的结构为实例，为视频操作展示之后的文件结构，详见视频
- models：负责存放模型
- outputs：存放输出
  - chroma_db：向量数据库
  - pdf：存放pdf预处理后的元数据和图片
- papers：按照领域分类存放处理过后的文献
- test_pdfs：示例程序当中存放待处理论文的位置，可以改为其他的路径
示例文件结构（使用过各种功能后）：
```
├─models
│  ├─all-MiniLM-L6-v2
│  │  └─1_Pooling
│  ├─clip
│  └─Florence-2-base
│      ├─.no_exist
│      │  └─5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac
│      ├─blobs
│      ├─refs
│      └─snapshots
│          └─5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac
├─outputs
│  ├─chroma_db
│  │  ├─3cf8d59b-ba6a-4a2d-9745-5df11c230674
│  │  ├─9e29b756-9f5a-49b9-9193-76e71a56d9d7
│  │  └─f282c5ce-ac59-4983-8ab2-399c8d7c1ecd
│  └─pdf
│      ├─DB3_Team_s_Solution_For_Meta_KDD_Cup_25_5e0b67c7
│      │  └─images
│      ├─SIRAG_Towards_Stable_and_Interpretable_RAG_with_8675cc85
│      │  └─images
│      ├─Song_Collaborative_Semantic_Occupancy_Prediction_with_Hybrid_Feature_Fusion_in_C_9622473e
│      │  └─images
│      └─Xu_CoSDH_Communication-Efficient_Collaborative_Perception_via_Supply-Demand_Awar_6619734f
│          └─images
├─papers
│  ├─CV
│  └─NLP
├─test_pdfs

```


