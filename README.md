# 🎯 NLP Text Classification with Translation


Streamlit application for analyzing customer service transcripts with **automatic language detection and translation**, sentiment analysis, and category classification.

## Project Structure
- `Latest Streamlit app in root folder (use this one).
- `archive/streamlit_nlp_app.py` → Old version, kept for reference only.

## 🚀 Live Demo

**Streamlit Cloud:** [Launch App](https://venkatbilla2008-nlp-text-pipeline.streamlit.app) *(Update this URL after deployment)*

## ✨ Features

- **15 Category Classification** - Automatic issue type detection
- **5-Level Sentiment Analysis** - Very negative to very positive
- **🌍 Multi-language Support** - Automatic translation to English
- **Language Detection** - Detects source language automatically
- **Parquet Output** - 30-50% smaller file sizes than CSV
- **Multi-threaded Processing** - Fast parallel analysis
- **Interactive UI** - Drag-and-drop file upload with real-time progress

## 🌍 Translation Support

The app automatically:
1. Detects the language of customer text
2. Translates non-English text to English
3. Performs analysis on translated text
4. Preserves both original and translated text in output

**Supported Languages:** All major languages (Spanish, French, German, Chinese, Japanese, etc.)

## 📊 Usage

### Input Requirements

Your CSV/Excel file must contain:
- **Conversation Id** - Unique identifier for each conversation
- **transcripts** - Full conversation transcript

**Example format:**
```
Consumer: I can't login | Agent: Let me help
Consumer: No puedo iniciar sesión | Agent: Te ayudaré
```

### Output Format

The app generates 6 columns:
1. **Conversation Id** - Original ID
2. **Consumer_Text** - Extracted consumer text
3. **Translated_Text** - English translation (if applicable)
4. **Category** - Main issue category
5. **Subcategory** - Specific subcategory
6. **Sentiment** - Emotional analysis

## 🎓 Supported Categories

**Main Categories (14):**
- login issue
- account issue
- playback issue
- device issue
- content restriction
- ad issue
- recommendation issue
- ui issue
- general feedback
- network failure
- app crash
- performance issue
- data sync issue
- subscription issue

**Sentiment Levels (5):**
very negative → negative → neutral → positive → very positive

## 🔧 Local Development

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/venkatbilla2008/nlp-text-pipeline.git
cd nlp-text-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_nlp_app.py
```

The app will open at `http://localhost:8501`

## ⚙️ Configuration

### Settings (Available in Sidebar)

- **Enable Translation** - Toggle automatic translation
- **Number of Threads** - Adjust for processing speed
- **Translation Delay** - Prevent rate limiting (0.1-2.0 seconds)
- **Preview Rows** - Number of rows to display (10-500)

### Performance Tips

- **Small files** (<1,000 rows): Use 4 threads
- **Large files** (10,000+ rows): Use 2 threads for stability
- **Translation enabled**: Add 0.5s delay to avoid rate limits

## 📋 File Limits

- **Maximum file size:** 100 MB
- **Maximum rows:** 50,000
- **Supported formats:** CSV, XLSX

## 🌟 Output Formats

### Parquet (Recommended)
- 30-50% smaller than CSV
- Faster to load
- Preserves data types

### CSV
- Compatible with Excel
- Easy to open and edit
- Universal format

## 📊 Example Output

| Conversation Id | Consumer_Text | Translated_Text | Category | Subcategory | Sentiment |
|----------------|---------------|-----------------|----------|-------------|-----------|
| CONV_001 | I can't login | | login issue | login | negative |
| CONV_002 | No puedo iniciar sesión | I cannot log in | login issue | login | negative |

## 🐛 Troubleshooting

### Translation Errors

If you see translation errors:
1. Increase translation delay to 1.0s
2. Reduce number of threads
3. Try processing smaller batches

### NLTK Data Errors

The app automatically downloads required NLTK data on first run. If errors occur, the data will be re-downloaded automatically.

### Performance Issues

- Reduce number of threads to 2
- Disable translation for English-only datasets
- Process large files in smaller batches

## 📦 Dependencies

Core libraries:
- **Streamlit** - Web framework
- **Pandas** - Data processing
- **TextBlob** - Sentiment analysis
- **AFINN** - Sentiment lexicon
- **googletrans** - Translation (uses Google Translate API)
- **langdetect** - Language detection
- **PyArrow** - Parquet support

## 🔒 Privacy & Security

- All processing happens **locally** or on your Streamlit Cloud instance
- No data is stored permanently
- Translation uses public Google Translate API
- Files are processed in memory only

## 📄 License

MIT License - Feel free to use and modify for your projects

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/venkatbilla2008/nlp-text-pipeline/issues)
- **Documentation:** See this README

## 🎯 Roadmap

Future enhancements:
- [ ] Support for more languages
- [ ] Custom category training
- [ ] API endpoint
- [ ] Batch file processing
- [ ] Export to multiple formats
- [ ] Advanced analytics dashboard

---

**Built with:** Streamlit, Pandas, TextBlob, Google Translate API

**Version:** 2.0.0 (with Translation)  
**Last Updated:** December 2024
