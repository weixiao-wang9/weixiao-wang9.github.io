---
id: 202512170427
type: source
subtype: lecture_note
course: "[[Computer Networks (CS 6250)]]"
module: "Module 10: Multimedia Applications"
title: Multimedia Applications and Streaming
status: Finished
tags:
  - input
  - networks
  - multimedia
  - voip
  - streaming
created: 2025-12-17
---

# Multimedia Applications and Streaming

## 📌 Context & Summary
> [!abstract] Context
> This lecture explores network applications that employ audio or video. It contrasts the requirements of **Stored Streaming** (Netflix) against **Conversational VoIP** (Skype). It details the technical underpinnings of **Video Compression** (Spatial vs. Temporal redundancy) and the modern architecture of **DASH** (Dynamic Adaptive Streaming over HTTP), focusing on algorithms that adapt bitrate to network conditions.

## 📝 Notes & Highlights

### Multimedia Categories
- **Streaming Stored:** Interactive (pause/skip), continuous playout, starts quickly. High jitter tolerance due to buffering.
- **Conversational (VoIP):** Real-time, highly delay-sensitive (needs <150ms ideal, >400ms is unusable), but loss-tolerant.
- **Streaming Live:** Similar to stored but time-sensitive; broadcast-like scaling challenges.

### VoIP Mechanisms
- **Encoding:** Analog audio is digitized via sampling and quantization (e.g., PCM).
- **Signaling:** Protocols like **SIP** handle user location, session establishment, and negotiation.
- **QoS Metrics:** End-to-end delay, Jitter (delay variation), and Packet Loss.

### Video Compression
- **Spatial Redundancy:** Compressing within a single image (JPEG). Uses DCT (Discrete Cosine Transform) and Quantization.
- **Temporal Redundancy:** Compressing across frames using **I-frames** (independent), **P-frames** (predicted), and **B-frames** (bi-directional).

### Streaming Architecture (DASH)
- **Protocol:** Uses **HTTP over TCP**. Stateless servers allow use of standard CDNs and traversal of middleboxes.
- **Bitrate Adaptation:**
    - **Rate-Based:** Estimates future bandwidth based on past throughput. Prone to underestimation due to TCP's "ON-OFF" behavior.
    - **Buffer-Based:** Selects bitrate based on current buffer occupancy. Avoids re-buffering but risks oscillation.

## 🔗 Extracted Concepts
- [[Concept - Multimedia Application Types]]
- [[Concept - VoIP QoS Mechanisms]]
- [[Concept - Video Compression Techniques]]
- [[Concept - DASH Architecture]]
- [[Concept - Bitrate Adaptation Algorithms]]