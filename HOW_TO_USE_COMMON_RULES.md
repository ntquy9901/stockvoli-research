# How to Use ML/DS Common Rules

**Cách sử dụng Common Rules cho nhiều dự án khác nhau**

---

## 🎯 Repository Mới Đã Tạo

**GitHub Repository:** https://github.com/ntquy9901/ml-ds-common-rules

**Mục đích:** Chứa common clean code rules có thể tái sử dụng cho mọi dự án ML/DS.

---

## 📋 Cách Sử Dụng

### Cách 1: Clone và Copy (Đơn giản nhất)

**Dành cho:** Dự án mới, tích hợp một lần

```bash
# Clone repository
git clone https://github.com/ntquy9901/ml-ds-common-rules.git temp-rules

# Copy COMMON_RULES.md vào dự án của bạn
cp temp-rules/COMMON_RULES.md your-project/docs/common-rules.md

# Xóa thư mục tạm
rm -rf temp-rules
```

---

### Cách 2: Git Submodule (Tốt nhất để cập nhật)

**Dành cho:** Dự án đang chạy, muốn tự động cập nhật

```bash
# Trong thư mục dự án của bạn
cd your-project

# Thêm submodule
git submodule add https://github.com/ntquy9901/ml-ds-common-rules.git docs/common-rules

# Commit
git add .gitmodules docs/common-rules
git commit -m "Add ML/DS common rules as submodule"

# Sau này muốn cập nhật
git submodule update --remote docs/common-rules
```

---

### Cách 3: Tích Hợp Trực Tiếp (Tùy chỉnh)

**Dành cho:** Muốn nhúng rules trực tiếp vào CLAUDE.md

**Bước 1:** Copy nội dung COMMON_RULES.md
**Bước 2:** Paste vào CLAUDE.md của dự án bạn
**Bước 3:** Thêm project-specific rules sau Section 2

**Ví dụ cấu trúc:**
```markdown
# Your Project CLAUDE.md

## 1. Project Overview
Giới thiệu dự án của bạn...

## 2. Common ML/DS Research Rules
[PASTE NỘI DUNG COMMON_RULES.M vào đây]

## 3. Project-Specific Rules
Rules cụ thể cho dự án của bạn...

## 4. Architecture
Quyết định kiến trúc...
```

---

## 📁 Cấu Trúc Repository

```
ml-ds-common-rules/
├── README.md                          # Tổng quan và hướng dẫn
├── COMMON_RULES.md                    # Full quy tắc chi tiết
├── QUICK_REFERENCE.md                 # Tham khảo nhanh
├── INTEGRATION_GUIDE.md              # Hướng dẫn tích hợp
├── CLAUDE_TEMPLATE.md                 # Template cho CLAUDE.md
└── examples/
    ├── ml-research/                   # Ví dụ ML research
    ├── data-science/                  # Ví dụ data science
    └── mlops/                         # Ví dụ MLOps
```

---

## 🎓 Nội Dung Common Rules

### COMMON_RULES.md (8 Phần Chính)

1. **Core Principles** (Nguyên tắc cốt lõi)
   - Code is read much more than written
   - Leave code better than you found it
   - Keep it simple
   - Match code quality to maturity

2. **Naming Conventions** (Quy tắc đặt tên)
   - Variables: `learning_rate` not `lr`
   - Functions: `evaluate_model` not `eval`
   - Classes: `Classifier` not `Manager`
   - Constants: `MAX_EPOCHS = 100`

3. **Function Design** (Thiết kế hàm)
   - Small functions (< 30 lines)
   - One thing only
   - Few parameters (< 3)
   - No side effects

4. **Code Organization** (Tổ chức code)
   - File structure (one concern per file)
   - Newspaper principle (high-level at top)
   - Related code close together

5. **Comments & Documentation** (Comment và docs)
   - Explain WHY not HOW
   - Use docstrings (Google style)
   - Delete outdated comments

6. **Formatting & Style** (Định dạng)
   - PEP 8 compliance
   - Vertical openness
   - Consistent style

7. **Anti-Patterns** (Cần tránh)
   - Naming: `x`, `data`, `manager`
   - Functions: Large (> 50 lines)
   - Organization: Mixed concerns

8. **Research Best Practices** (Tốt nhất cho research)
   - Version control everything
   - Track hyperparameters
   - Reproducible seeds
   - Save intermediate results

---

## 🚀 Sử Dụng Cho Dự Án Mới

### Bước 1: Clone Template

```bash
# Tạo dự án mới
mkdir my-new-ml-project
cd my-new-ml-project

# Clone common rules
git clone https://github.com/ntquy9901/ml-ds-common-rules.git temp-rules

# Copy template
cp temp-rules/CLAUDE_TEMPLATE.md CLAUDE.md

# Tùy chỉnh cho dự án của bạn
# Edit CLAUDE.md, thêm project-specific rules

# Xóa temp
rm -rf temp-rules
```

### Bước 2: Tùy Chỉnh

Trong `CLAUDE.md`, thêm sau Section 2:

```markdown
## 3. Project-Specific Rules

### 3.1. [Lĩnh vực của bạn] Specific Rules
- Rules cụ thể cho lĩnh vực
- Ví dụ: Financial ML, Computer Vision, NLP

### 3.2. Architecture
- Quyết định kiến trúc của dự án
- Model choices, data pipeline, etc.
```

---

## 🔄 Cập Nhật khi Có Version Mới

### Nếu Dùng Submodule

```bash
# Cập nhật lên version mới nhất
git submodule update --remote docs/common-rules

# Review changes
git diff docs/common-rules

# Commit nếu hài lòng
git add docs/common-rules
git commit -m "Update ML/DS common rules to latest version"
```

### Nếu Dùng Copy

```bash
# Clone version mới nhất
git clone https://github.com/ntquy9901/ml-ds-common-rules.git temp-rules

# So sánh với version hiện tại
diff temp-rules/COMMON_RULES.md docs/common-rules.md

# Merge changes thủ công
# Review và update file của bạn

# Xóa temp
rm -rf temp-rules
```

---

## 📊 Ví Dụ Sử Dụng Thực Tế

### Ví Dụ 1: Dự án TimesFM Này

```bash
# Đã tích hợp common rules vào Section 2 của CLAUDE.md
# Xem file: CLAUDE.md

# Cấu trúc:
## 1. Project Overview (TimesFM VN30)
## 2. Common ML/DS Research Rules (Universal rules)
## 3. TimesFM-Specific (Financial ML rules)
## 4. Vietnamese Market Specifics
## 5. Metrics & Validation
# ... etc
```

### Ví Dụ 2: Dự án Computer Vision Mới

```bash
# Tạo dự án CV mới
mkdir cv-object-detection
cd cv-object-detection

# Clone common rules
git clone https://github.com/ntquy9901/ml-ds-common-rules.git temp-rules
cp temp-rules/CLAUDE_TEMPLATE.md CLAUDE.md
rm -rf temp-rules

# Tùy chỉnh CLAUDE.md cho CV
# Section 3: Computer Vision Specific Rules
# - Image preprocessing pipeline
# - Data augmentation strategies
# - Model architectures for object detection
```

### Ví Dụ 3: Dự án NLP

```bash
# Tương tự nhưng thêm NLP-specific rules
# Section 3: NLP Specific Rules
# - Text preprocessing
# - Tokenization strategies
# - Transformer models
```

---

## 🎓 Training và Onboarding

### Cho Team Member Mới

```bash
# Bước 1: Đọc quick reference (5 phút)
# Đọc file: docs/common-rules/QUICK_REFERENCE.md

# Bước 2: Đọc full rules (20 phút)
# Đọc file: docs/common-rules/COMMON_RULES.md

# Bước 3: Đọc project-specific rules (10 phút)
# Đọc file: CLAUDE.md (Section 3+)

# Bước 4: Áp dụng trong code
# Sử dụng quick reference checklist khi code
```

### Code Review Checklist

```markdown
## Code Review Checklist

### Common Rules (Section 2)
- [ ] Variables/functions named descriptively?
- [ ] Functions small (< 30 lines)?
- [ ] Code organized logically?
- [ ] Comments explain WHY not HOW?
- [ ] No obvious anti-patterns?
- [ ] Code formatted consistently?

### Project-Specific (Section 3+)
- [ ] [Rules cụ thể cho dự án của bạn]
```

---

## 📚 Tài Liệu Tham Khảo

### Trong Repository

- **README.md:** Tổng quan và quick start
- **COMMON_RULES.md:** Full guidelines (chi tiết)
- **QUICK_REFERENCE.md:** Tham khảo nhanh (hàng ngày)
- **INTEGRATION_GUIDE.md:** Hướng dẫn tích hợp chi tiết
- **CLAUDE_TEMPLATE.md:** Template để bắt đầu

### Examples

- **examples/ml-research/:** ML research project example
- **examples/data-science/:** Data science project example
- **examples/mlops/:** MLOps project example

### External Resources

- **Clean Code** by Robert C. Martin
- **The Pragmatic Programmer** by Andrew Hunt and David Thomas
- **PEP 8 Style Guide** for Python Code

---

## 🤝 Đóng Góp

Tìm thấy cải tiến? Đóng góp lại!

```bash
# 1. Fork repository
https://github.com/ntquy9901/ml-ds-common-rules

# 2. Tạo branch
git checkout -b feature/improvement

# 3. Thay đổi
# Edit COMMON_RULES.md hoặc files khác

# 4. Submit PR
# Giải thích lý do changes
# Show examples nếu applicable
```

---

## 📞 Hỗ Trợ

**Câu hỏi?**
- Open issue: https://github.com/ntquy9901/ml-ds-common-rules/issues
- Start discussion: https://github.com/ntquy9901/ml-ds-common-rules/discussions
- Email: ntquy9901@gmail.com

---

## ✅ Summary

**Repository:** https://github.com/ntquy9901/ml-ds-common-rules

**3 Cách Sử Dụng:**
1. Clone và copy (đơn giản nhất)
2. Git submodule (tự động cập nhật)
3. Tích hợp trực tiếp (tùy chỉnh)

**Key Files:**
- COMMON_RULES.md (full guidelines)
- QUICK_REFERENCE.md (daily reference)
- INTEGRATION_GUIDE.md (integration help)
- CLAUDE_TEMPLATE.md (starting template)

**Universal Rules:**
- Naming conventions
- Function design
- Code organization
- Documentation practices
- Research best practices

**Ready to use for all ML/DS projects!** 🚀

---

*Last Updated: 2026-06-13*
*Universal ML/DS Clean Code Rules*
