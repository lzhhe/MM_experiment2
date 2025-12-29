# 本地 AI 智能文献与图像管理助手 (Local Multimodal AI Agent)

## 1. 项目简介 (Project Introduction)
本项目是一个基于 Python 的本地多模态 AI 智能助手，旨在解决本地大量文献和图像素材管理困难的问题。不同于传统的文件名搜索，本项目利用多模态神经网络技术，实现对内容的**语义搜索**和**自动分类**。本项目可以帮助各位同学理解多模态大模型的实际应用，并且可以在实际日常学习生活中帮助各位同学管理自己的本地知识库。希望各位同学可以不局限于本次作业规定的内容，通过自己的构建、扩展和维护实现自己的本地AI助手。

项目可使用本地化部署，也可以调用云端大模型 API 以获得更强的性能。

## 2. 核心功能要求 (Core Features)

### 2.1 智能文献管理
*   **语义搜索**: 支持使用自然语言提问（如“Transformer 的核心架构是什么？”）。系统需基于语义理解返回最相关的论文文件，进阶要求可返回具体的论文片段或页码。
*   **自动分类与整理**:
    *   **单文件处理**: 添加新论文时，根据指定的主题（如 "CV, NLP, RL"）自动分析内容，将其归类并移动到对应的子文件夹中。
    *   **批量整理**: 支持对现有的混乱文件夹进行“一键整理”，自动扫描所有 PDF，识别主题并归档到相应目录。
*   **文件索引**: 支持仅返回相关文件列表，方便快速定位所需文献。

### 2.2 智能图像管理
*   **以文搜图**: 利用多模态图文匹配技术，支持通过自然语言描述（如“海边的日落”）来查找本地图片库中最匹配的图像。

## 3. 技术选型与模型建议 (Technical Stack)

本项目采用模块化设计，支持替换不同的后端模型。学生可根据自身硬件条件选择本地部署或调用云端 API（如 Gemini, GPT-4o 等）。

### 3.1 推荐配置 (轻量级/本地化)
*   **文本嵌入**: `SentenceTransformers` (如 `all-MiniLM-L6-v2`) —— 速度快，内存占用小。
*   **图像嵌入**: `CLIP` (如 `ViT-B-32`) —— OpenAI 开源的经典图文匹配模型。
*   **向量数据库**: `ChromaDB` —— 无需服务器，开箱即用的嵌入式数据库。

### 3.2 进阶配置建议 (高性能/多功能)
如果您拥有较好的硬件资源（如 NVIDIA GPU），可以尝试以下方案：

*   **图像描述与问答**:
    *   **Florence-2 (Microsoft)**: 轻量级全能视觉模型，支持 OCR、检测、描述。
    *   **Moondream**: 专为边缘设备设计的小型视觉语言模型。
    *   **LLaVA**: 开源多模态大模型，支持复杂的图文对话。
*   **文本理解**:
    *   **本地 LLM**: 如 `Llama-3` 或 `Qwen-2` (通过 Ollama 部署)，实现更精准的分类。

## 4. 环境要求 (Environment)

*   **操作系统**: Windows / macOS / Linux
*   **Python 版本**: 建议 Python 3.8 及以上
*   **内存**: 建议 8GB 及以上

## 5. 作业提交要求 (Submission Guidelines)

为规范作业提交与评测流程，请严格按照以下要求提交：

### 5.1 代码提交
*   **GitHub 仓库**: 将完整项目代码上传至 GitHub 个人仓库，并提交仓库链接。
*   **README 文档**: 仓库首页必须包含详细的 `README.md`，内容包括：
    *   项目简介与核心功能列表。
    *   环境配置与依赖安装说明。
    *   **详细的使用说明**（包含具体的命令行示例）。
    *   技术选型说明（使用了哪些模型、数据库等）。

### 5.2 评测接口规范
*   **统一入口**: 项目根目录下必须包含 `main.py` 文件。
*   **一键调用**: 必须支持通过命令行参数调用核心功能。参考格式如下（不仅限于此）：
    *   添加/分类论文: `python main.py add_paper <path> --topics "Topic1,Topic2"`
    *   搜索论文: `python main.py search_paper <query>`
    *   以文搜图: `python main.py search_image <query>`

### 5.3 演示文档
请提交一份 PDF 格式的演示报告（或直接包含在 README 中），内容需包括：
*   **运行截图**: 关键功能的运行结果截图（如搜索结果、分类后的文件夹结构）。
*   **演示视频 (可选)**: 录制一段屏幕录像，演示从环境启动到功能使用的全过程。

---

# Local Multimodal AI Agent (English Version)

## 1. Project Introduction
This project is a Python-based local multimodal AI assistant designed to solve the difficulty of managing large collections of local documents and images. Unlike traditional filename-based searches, this project utilizes multimodal neural network technology to achieve **semantic search** and **automatic classification** of content.

The project is designed to be flexible, supporting both fully offline local deployment (for privacy) and cloud-based large model API integration for enhanced performance.

## 2. Core Features

### 2.1 Intelligent Document Management
*   **Semantic Search**: Supports natural language queries (e.g., "What is the core architecture of Transformer?"). The system should return the most relevant paper documents based on semantic understanding. Advanced implementations can return specific snippets or page numbers.
*   **Automatic Classification & Organization**:
    *   **Single File Processing**: When adding a new paper, the system automatically analyzes the content based on specified topics (e.g., "CV, NLP, RL") and moves it to the corresponding subfolder.
    *   **Batch Organization**: Supports "one-click organization" for existing messy folders, automatically scanning all PDFs, identifying topics, and archiving them into appropriate directories.
*   **File Indexing**: Supports returning only a list of relevant files for quick location.

### 2.2 Intelligent Image Management
*   **Text-to-Image Search**: Utilizes multimodal text-image matching technology to allow users to find the best-matching images in the local library using natural language descriptions (e.g., "sunset by the sea").

## 3. Technical Stack & Recommendations

This project adopts a modular design, allowing for the replacement of different backend models. Students can choose between local deployment or calling cloud APIs (such as Gemini, GPT-4o) based on their hardware conditions.

### 3.1 Recommended Configuration (Lightweight/Local)
*   **Text Embedding**: `SentenceTransformers` (e.g., `all-MiniLM-L6-v2`) — Fast speed, low memory usage.
*   **Image Embedding**: `CLIP` (e.g., `ViT-B-32`) — OpenAI's classic open-source text-image matching model.
*   **Vector Database**: `ChromaDB` — Serverless, out-of-the-box embedded database.

### 3.2 Advanced Configuration (High Performance)
If you have better hardware resources (such as NVIDIA GPUs), consider the following options:

*   **Image Captioning & QA**:
    *   **Florence-2 (Microsoft)**: Lightweight yet powerful vision model supporting OCR, detection, and captioning.
    *   **Moondream**: Small vision-language model designed for edge devices.
    *   **LLaVA**: Open-source multimodal large model supporting complex image-text dialogue.
*   **Text Understanding**:
    *   **Local LLM**: Such as `Llama-3` or `Qwen-2` (deployed via Ollama) for more precise classification.

## 4. Environment Requirements

*   **OS**: Windows / macOS / Linux
*   **Python Version**: Recommended Python 3.8+
*   **Memory**: Recommended 8GB+ (for loading basic Embedding models)

## 5. Submission Guidelines

To standardize the submission and evaluation process, please strictly follow these requirements:

### 5.1 Code Submission
*   **GitHub Repository**: Upload the complete project code to a personal GitHub repository and submit the link.
*   **README**: The repository homepage must include a detailed `README.md` containing:
    *   Project introduction and core feature list.
    *   Environment configuration and dependency installation instructions.
    *   **Detailed usage instructions** (including specific command-line examples).
    *   Technical stack explanation (models, databases used, etc.).

### 5.2 Evaluation Interface Specification
*   **Unified Entry Point**: The project root directory must contain a `main.py` file.
*   **One-Click Execution**: Core features must be callable via command-line arguments. Reference format (not limited to):
    *   Add/Classify Paper: `python main.py add_paper <path> --topics "Topic1,Topic2"`
    *   Search Paper: `python main.py search_paper <query>`
    *   Search Image: `python main.py search_image <query>`

### 5.3 Demonstration Documentation
Please submit a PDF demonstration report (or include it in the README), which must include:
*   **Screenshots**: Screenshots of key function results (e.g., search results, folder structure after classification).
*   **Demo Video (Optional)**: Record a screen capture video demonstrating the entire process from environment startup to feature usage.
