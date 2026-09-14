# CAG Estimate 🚀

**Context-Augmented Generation for Project Estimation** - An intelligent API that generates detailed project estimations based on meeting transcriptions using Anthropic's Claude AI.

## 📖 Main Documentation

**➡️ [See the full project documentation →](./cag-estimate/README.md)**

The actual API implementation and comprehensive documentation is located in the `cag-estimate/` folder.

## Quick Overview

### What is CAG Estimate?

CAG Estimate uses the **Context-Augmented Generation (CAG)** architecture pattern to:
- Analyze meeting transcriptions
- Generate detailed project estimations
- Break down tasks with hours and costs
- Recommend team size and timeline
- Provide cost tracking for API usage

### Key Features

✅ **Context-Augmented Generation** - Uses reference examples to improve estimation quality  
✅ **Detailed Task Breakdown** - Each task includes hours, costs, and deliverables  
✅ **Team Composition Recommendations** - Suggests optimal team size and roles  
✅ **Timeline Estimates** - Calculates project duration in weeks  
✅ **Cost Analysis** - Provides both LLM API costs and project costs  
✅ **Automated Verification** - 5-stage validation pipeline for manager confidence  
✅ **Production Ready** - FastAPI, error handling, CORS support, comprehensive logging  

### Quick Example

**Input (Meeting Transcription):**
```
"we have to estimate a new feature for the mobile app in iOS only, 
than most include a new chat with an agent than support the grocery 
shopping carts creation base on the grocery shops around where you live..."
```

**Output (Generated Estimation):**
```json
{
  "project_name": "Grocery Price Comparison & Smart Shopping Mobile App",
  "total_hours": 384,
  "total_cost_usd": 15360,
  "team_size": "2 developers (1 iOS, 1 Backend) + 1 QA Engineer",
  "duration_weeks": 10,
  "hourly_rate": 40
}
```

**Verification:** ✅ All validations passed!

## 📂 Project Structure

```
cag-estimate/
├── cag-estimate/                     # Main project folder
│   ├── README.md                     # ⭐ FULL DOCUMENTATION HERE
│   ├── VERIFICATION.md               # Verification pipeline guide
│   ├── pyproject.toml                # Project configuration
│   ├── .env.example                  # Environment template
│   ├── src/cag_estimate/
│   │   ├── config.py                 # Settings management
│   │   ├── main.py                   # FastAPI app
│   │   ├── routers/estimations.py    # API endpoints
│   │   ├── services/llm_service.py   # LLM integration
│   │   └── context/examples.py       # Reference examples
│   └── tests/
│       ├── test_verification.py      # Verification tests (9/9 passing)
│       └── verify_api.py             # API integration tests
└── README.md                          # This file
```

## 🚀 Getting Started

1. **Navigate to the project directory:**
   ```bash
   cd cag-estimate
   ```

2. **Read the full documentation:**
   ```bash
   cat README.md
   ```

3. **Or open in your editor:**
   ```bash
   # VS Code, Sublime, etc.
   code cag-estimate/README.md
   ```

## 🔗 Important Links

📖 **[Full Documentation](./cag-estimate/README.md)** - Complete setup, API guide, examples  
🧪 **[Verification Pipeline](./cag-estimate/VERIFICATION.md)** - Manager's confidence levels guide  
🐍 **[Python Project Config](./cag-estimate/pyproject.toml)** - Dependencies and settings  
📝 **[Environment Template](./cag-estimate/.env.example)** - Configuration template  

## 📊 Technology Stack

- **Language:** Python 3.11+
- **API Framework:** FastAPI 0.141+
- **LLM:** Anthropic Claude (Haiku 4.5)
- **Data Validation:** Pydantic 2.13+
- **Package Management:** UV
- **Testing:** Pytest

## 💰 Pricing

**Per Estimation Request:** ~$0.00006 (less than a penny!)
- Input tokens: $3 per 1M
- Output tokens: $15 per 1M
- Ultra-affordable for production use

## ✅ Testing Status

```
Verification Pipeline: 9/9 tests ✅ PASSING
├── Schema Validation ✅
├── Business Logic ✅
├── Reasonableness Checks ✅
├── Content Quality ✅
├── Consistency Validation ✅
├── Invalid Schema Detection ✅
├── Calculation Error Detection ✅
├── Reasonableness Bounds ✅
└── Complete Validation ✅
```

## 📋 Files in Repository

- **`cag-estimate/`** - Main project implementation (4,232+ lines of code)
- **`README.md`** - This file (project overview)
- **`.gitignore`** - Git ignore rules

## 🎯 Next Steps

1. **[Read the Full Documentation](./cag-estimate/README.md)** for setup instructions
2. **[Review Verification Guide](./cag-estimate/VERIFICATION.md)** for manager workflow
3. **Navigate to `cag-estimate/`** and follow the Getting Started section

---

**⭐ Start here:** [`./cag-estimate/README.md`](./cag-estimate/README.md)

Generated with [Claude Code](https://claude.com/claude-code) | Powered by [Anthropic Claude API](https://docs.anthropic.com)
