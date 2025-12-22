# 输出样例说明

本文档说明项目运行后的期望输出结果。

---

## 输出目录结构

```
outputs/
├── generation/              # 人脸生成结果
│   ├── face_1_final.png
│   ├── face_2_final.png
│   ├── ...
│   ├── face_1_denoising_process.png
│   └── ...
│
└── interpolation/           # 身份插值结果
    ├── interpolation_1_to_2.png
    ├── interpolation_2_to_3.png
    └── ...
```

## 输出说明

### 1. 人脸生成 (`outputs/generation/`)

- **`face_X_final.png`**: 最终生成的人脸图像 (128×128)
- **`face_X_denoising_process.png`**: 去噪过程可视化，展示从噪声到清晰人脸的转变

### 2. 身份插值 (`outputs/interpolation/`)

- **`interpolation_X_to_Y.png`**: 身份X到Y的完整插值序列
- 展示两个身份之间的平滑过渡过程
