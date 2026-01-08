# 🧪 SubtitleLab

<p align="center">
  <b>Refine your Whisper-generated subtitles with the power of Intelligence.</b><br>
  <i>An advanced post-processing suite for perfectly polished SRT/ASS subtitles.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Mac%20%7C%20Linux-orange?style=for-the-badge" alt="Platform">
  <img src="https://img.shields.io/badge/GUI-Flet-red?style=for-the-badge" alt="Flet">
</p>

---

## 📖 Overview

**SubtitleLab** is a premium post-processing tool designed to transform raw, hallucination-prone Whisper transcripts into professional-grade subtitles. By leveraging Large Language Models (LLMs), SubtitleLab handles the linguistic nuances that traditional rule-based tools miss—specifically optimized for **Japanese to Chinese** subtitle correction.

### ❓ Why SubtitleLab?

Whisper is revolutionary, but its output often suffers from:
- **Hallucinations**: Repeating phrases or "ghost" text during silence.
- **Segmentation Issues**: Awkward line breaks or fragmented sentences.
- **Homophone Confusion**: Incorrect character selection (同音字) based on phonetic similarity.

SubtitleLab acts as an intelligent "AI Editor" that understands context, tone, and flow to ensure your subtitles are not just accurate, but readable.

---

## ✨ Key Features

- **🧠 LLM-Powered Correction**: Context-aware fixing of homophone errors and linguistic slips.
- **✂️ Smart Sentence Management**: Automatically merges fragmented segments and splits overly long lines.
- **🧹 Whisper Cleanup**: Deep detection and removal of common Whisper hallucinations (e.g., "Thanks for watching", repetitive loops).
- **🎨 Modern Flet GUI**: A sleek, reactive interface with dark mode support and real-time feedback.
- **⚡ Async Engine**: High-performance asynchronous processing with live progress tracking.
- **📊 Quality Validation**: Automated scoring system that evaluates coherence, timing, and formatting.
- **🌐 Universal API Support**: Works with any OpenAI-compatible provider (OpenAI, DeepSeek, Claude via proxy, etc.).

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/roymecat/subtitle-lab.git
cd subtitlelab

# Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Launching the App

Simply run the entry point from your terminal:

```bash
subtitlelab
```

---

## 🛠️ Configuration

SubtitleLab offers extensive customization through its modern settings panel:

- **API Endpoint**: Connect to your preferred LLM provider.
- **Model Selection**: Toggle between speed and accuracy (e.g., GPT-4o vs GPT-3.5-Turbo).
- **Processing Logic**: Adjust the aggressiveness of sentence merging and hallucination filters.
- **Language Profiles**: Optimized presets for Japanese source to Chinese target.

---

## 💻 Tech Stack

- **UI Framework**: [Flet](https://flet.dev/) (Flutter for Python)
- **Core Engine**: Python 3.10+
- **LLM Client**: OpenAI Python SDK
- **Subtitle Logic**: [pysubs2](https://github.com/tomasgroth/pysubs2)
- **Reliability**: Tenacity (Advanced retry mechanisms)

---

## 📸 Screenshots

<p align="center">
  <i>(Screenshot placeholders - Modern GUI coming soon!)</i>
</p>

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  Made with ❤️ by the SubtitleLab Team
</p>
