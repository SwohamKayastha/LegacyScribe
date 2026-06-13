---
title: LegacyScribe
emoji: 📖
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
license: mit
short_description: Turn grandparents' spoken memories into a beautiful family memory book
tags:
  - nepali
  - memory
  - family
  - fine-tuned
  - llama-cpp
  - offline
  - agents
---

# 📖 LegacyScribe — Family Memory Agent

> *Every family has a story worth keeping.*

Grandparents have stories, recipes, wisdom — and it all risks dying with them. LegacyScribe is a gentle AI companion that draws out those memories through warm conversation and weaves them into a beautiful, printable family memory book.

## How it works

1. **Speak or type freely** — in English, Nepali, or mixed
2. A **5-agent pipeline** extracts, structures, and verifies each memory fragment
3. The **memory book** builds itself in real time on the right side of the screen
4. Download a PDF chapter when you're ready

## Agents under the hood

| Agent | Role |
|---|---|
| Questioner | Asks one gentle follow-up question |
| Arc Detector | Identifies narrative stage (setup → tension → turn → meaning) |
| Extractor | Pulls who, when, where, emotion from loose speech |
| Reconciler | Catches contradictions across sessions |
| Publisher | Synthesizes atomic notes into narrative prose |

## Model

Fine-tuned **Qwen2.5-7B-Instruct** with LoRA rank 16 on 850 culturally-grounded examples spanning Nepali festivals, family relationships, livelihoods, and village life. Runs fully offline via llama.cpp.

## Hackathon

Built for the **Build Small Hackathon 2026** — Thousand Token Wood category.

Badges: **Off the Grid** · **Custom UI** · **Field Notes**