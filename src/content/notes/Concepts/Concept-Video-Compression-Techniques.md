---
id: 202512170428
type: atom
course: "[[Computer Networks (CS 6250)]]"
tags: [concept, compression, video]
related_topics: []
created: 2025-12-17
---

# Video Compression (Spatial & Temporal)

## 💡 The Core Idea
Raw video has a massive bitrate. Compression reduces size by exploiting **Spatial Redundancy** (pixels within a frame are similar) and **Temporal Redundancy** (adjacent frames are similar).

## 🧠 Mechanisms

### Spatial (Image) Compression (JPEG)
1.  **Transformation:** Convert RGB to YCbCr (Luminance/Chrominance).
2.  **DCT:** Apply Discrete Cosine Transform to 8x8 blocks to separate high/low frequencies.
3.  **Quantization:** Divide coefficients by a quantization table ($Q$) to reduce precision of high frequencies (Lossy step):
    $$F_{quantized}(u, v) = \frac{F(u, v)}{Q(u, v)}$$
   .

### Temporal Compression (MPEG)
Instead of sending every full image, send differences.
* **I-Frame (Intra):** A full, self-contained image (JPEG encoded).
* **P-Frame (Predicted):** Encodes only the changes from the previous frame.
* **B-Frame (Bi-directional):** Encodes changes relative to both past and future frames.
* **GoP:** A "Group of Pictures" is the sequence between I-frames.



## 🔗 Connections
- **Source:** [[Source - Multimedia Applications]]