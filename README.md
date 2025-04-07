LLM Rank Tracker
A powerful open-source tool to track and analyze brand visibility in AI-generated search results across major LLM platforms.
Show Image
📊 Overview
LLM Rank Tracker helps you monitor how often and prominently your brand, product, or keyword is mentioned in AI-generated search results. As AI-powered search becomes increasingly integrated into search experiences, traditional SEO tools no longer cover the full picture of brand visibility. This tool leverages free LLM APIs (currently Groq) to analyze brand mentions and provide comprehensive visibility insights with visual analytics.
✨ Key Features

Brand Mention Tracking: Monitor how often your brand appears in AI responses
Ranking Analysis: Determine where your brand ranks in competitive lists
Sentiment Assessment: Analyze how positively or negatively your brand is portrayed
Competitive Insights: See which other brands appear alongside yours
Visual Analytics: View trends with interactive charts and graphs
Exportable Data: Download results in CSV and JSON formats
Historical Tracking: Monitor changes in visibility over time
Brand Comparison: Compare multiple brands across different metrics
LLM Model Selection: Test with different Groq models
Custom Queries: Run specific prompts to test brand visibility

🛠️ Installation
Local Development

Clone this repository:
bashCopygit clone https://github.com/your-username/llm-rank-tracker.git
cd llm-rank-tracker

Create a virtual environment and install dependencies:
bashCopypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Get your Groq API key from https://console.groq.com/keys
Create a .env file with your API key:
CopyGROQ_API_KEY=your_api_key_here

Run the Streamlit app:
bashCopystreamlit run app.py


Streamlit Cloud Deployment

Fork this repository to your GitHub account
Sign up for Streamlit Cloud
Create a new app and connect it to your GitHub repository
In the Streamlit Cloud settings, add your Groq API key as a secret:

Name: GROQ_API_KEY
Value: your_api_key_here



🧑‍💻 Usage Guide
Brand Analysis

Enter your brand name and category
Select the query type that best fits your analysis needs
Choose your LLM model and temperature settings (optional)
Click "Run Analysis"
View comprehensive results including:

Visibility metrics (mentions, rank, sentiment)
Interactive charts showing your brand compared to competitors
Context of how your brand was mentioned
Actionable recommendations based on the results



Historical Data

Run multiple analyses to build historical data
View trends over time with interactive line charts
Export historical data in CSV or JSON format

Brand Comparison

Compare multiple brands across different metrics
Analyze competitive positioning with visualization tools
See average sentiment and mention counts across brands

📈 Data Visualization
LLM Rank Tracker includes several data visualization features:

Bar Charts: Compare brand mentions across competitors
Pie Charts: Visualize the distribution of brand visibility
Line Charts: Track changes in rank position and mention count over time
Sentiment Analysis Charts: Analyze emotional context around brand mentions

📤 Export Options
Download your data in multiple formats:

CSV Exports: For spreadsheet analysis
JSON Exports: For developers or advanced data analysis
Individual Reports: Export specific analysis results
Bulk Exports: Download all historical data

🚀 Roadmap
Phase 1 (Current)

✅ Brand mention tracking and analysis
✅ Groq API integration (multiple models)
✅ Interactive data visualizations
✅ Historical data tracking
✅ Brand comparison features
✅ Data export tools

Phase 2 (Planned)

 Support for additional LLM providers (Claude, Gemini)
 Schedule automated checks on a daily/weekly basis
 Advanced sentiment analysis with entity recognition
 Customizable dashboards
 Alert system for visibility changes
 API integration for programmatic access

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgements

Built with Streamlit
Powered by Groq
Visualizations with Plotly
Sentiment analysis with NLTK

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to suggest improvements or report bugs.
