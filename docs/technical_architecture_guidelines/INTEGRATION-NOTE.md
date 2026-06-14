# Integration Note - Technical Architecture Guidelines

**Date:** 2026-06-11
**Purpose:** Document integration with bmad-create-architecture skill

---

## ✅ **Configuration Updated**

### **What Was Changed:**

Updated `.claude/skills/bmad-create-architecture/customize.toml` to include all technical architecture guidelines as persistent facts.

### **Files Added to Persistent Facts:**

```toml
persistent_facts = [
  "file:{project-root}/**/project-context.md",
  "file:{project-root}/docs/technical_architecture_guidelines/00-README.md",
  "file:{project-root}/docs/technical_architecture_guidelines/01-Time-Series-ML-Fundamentals.md",
  "file:{project-root}/docs/technical_architecture_guidelines/02-Data-Processing-Best-Practices.md",
  "file:{project-root}/docs/technical_architecture_guidelines/03-Model-Training-Guidelines.md",
  "file:{project-root}/docs/technical_architecture_guidelines/04-Testing-Validation-Strategy.md",
  "file:{project-root}/docs/technical_architecture_guidelines/05-Production-Readiness-Checklist.md",
  "file:{project-root}/docs/technical_architecture_guidelines/06-Common-Pitfalls-Solutions.md",
]
```

---

## 🎯 **How It Works**

### **When bmad-create-architecture is Activated:**

1. **System loads `customize.toml`**
2. **Reads all persistent_facts entries**
3. **Loads file contents** for each `file:{project-root}/...` entry
4. **Makes context available** to the architecture workflow

### **Result:**

- ✅ AI automatically reads all 7 guideline documents
- ✅ Context available during architecture creation
- ✅ Best practices incorporated into decisions
- ✅ Pitfalls avoided based on lessons learned

---

## 📋 **Available Context During Architecture Creation**

### **Time Series ML Fundamentals:**
- Data leakage prevention (38.9% impact lesson)
- Temporal split vs random sampling
- Proper validation strategies

### **Data Processing Best Practices:**
- Log transformation (mandatory)
- Vietnamese market specifics
- Financial data pipeline

### **Model Training Guidelines:**
- TimesFM 2.5 architecture
- LoRA adapter configuration
- SGD optimizer (not AdamW)

### **Testing & Validation:**
- True out-of-sample testing
- Statistical validation (Diebold-Mariano)
- Performance metrics

### **Production Readiness:**
- Deployment checklist
- Monitoring strategy
- Maintenance procedures

### **Common Pitfalls:**
- 7 real-world issues encountered
- Solutions for each
- Prevention strategies

---

## 🚀 **Usage Example**

### **When Creating Architecture:**

```bash
# User invokes bmad-create-architecture
/bmad-create-architecture

# AI automatically:
1. Loads all 7 guideline documents
2. Incorporates lessons learned
3. Avoids common pitfalls
4. Follows best practices
5. Creates compliant architecture
```

### **Example Context Awareness:**

```
AI: "I'm reading the Time Series ML Fundamentals guideline...

      CRITICAL RULE: Always use temporal split for time series.
      Data leakage can inflate metrics by 38.9%.

      When designing the data architecture, I will:
      1. Use temporal split (80/20) for train/test
      2. Verify no temporal overlap
      3. Implement proper forward validation

      This ensures the architecture avoids the data leakage
      issue that caused 38.9% metric inflation in our project."
```

---

## ✅ **Verification**

### **How to Verify Integration:**

```bash
# 1. Check customize.toml
cat .claude/skills/bmad-create-architecture/customize.toml

# Should see all 7 guideline files listed in persistent_facts

# 2. Test by invoking skill
/bmad-create-architecture

# AI should reference guidelines in its responses
```

### **Expected Behavior:**

When creating architecture for time series ML projects, AI will:

1. ✅ **Mention temporal split requirement**
2. ✅ **Reference data leakage prevention**
3. ✅ **Specify log transformation**
4. ✅ **Recommend SGD optimizer**
5. ✅ **Include true out-of-sample validation**
6. ✅ **Avoid common pitfalls**

---

## 📊 **Benefits of Integration**

### **For Architecture Creation:**

- ✅ **Consistency:** All architectures follow same standards
- ✅ **Quality:** Best practices automatically incorporated
- ✅ **Speed:** No need to repeat guidelines
- ✅ **Accuracy:** Avoids known pitfalls

### **For Development Teams:**

- ✅ **Standards:** Everyone follows same patterns
- ✅ **Learning:** Guidelines embedded in workflow
- ✅ **Onboarding:** New developers learn from context
- ✅ **Maintenance:** Centralized guidelines repository

---

## 🔧 **Maintenance**

### **Updating Guidelines:**

```bash
# When guidelines are updated:

1. Update files in docs/technical_architecture_guidelines/
2. No need to update customize.toml (glob pattern matches all)
3. Next skill invocation loads updated content
```

### **Adding New Guidelines:**

```bash
# Create new guideline file
touch docs/technical_architecture_guidelines/07-New-Guideline.md

# Add to customize.toml persistent_facts
# (Automatic if following naming convention)

# Test by invoking skill
/bmad-create-architecture
```

---

## 📝 **Summary**

### **What Was Done:**

1. ✅ Updated `.claude/skills/bmad-create-architecture/customize.toml`
2. ✅ Added 7 technical architecture guidelines to persistent_facts
3. ✅ Guidelines now automatically loaded during architecture creation

### **Impact:**

- **Before:** AI might not follow best practices
- **After:** AI automatically incorporates all lessons learned

### **Result:**

> **Every architecture created with bmad-create-architecture will now**
> **follow our hard-earned best practices and avoid known pitfalls.**

---

**Status:** ✅ Integration Complete
**Effect:** Immediate (next skill invocation)
**Scope:** All future architecture creations
**Benefit:** Consistent, high-quality architectures

---

*This integration ensures our hard-learned lessons benefit all future projects.*

**Last Updated:** 2026-06-11
