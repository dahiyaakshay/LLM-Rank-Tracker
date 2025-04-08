import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime
import base64
from backend import query_groq_llm, extract_brand_mentions
from prompts import generate_prompt

# Page configuration
st.set_page_config(
    page_title="LLM Rank Tracker - AI Brand Visibility Analytics",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []
if "comparison_data" not in st.session_state:
    st.session_state.comparison_data = {}

# Header with custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4B61D1;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
        margin-top: 0;
    }
    .stat-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    <h1 class="main-header">LLM Rank Tracker</h1>
    <p class="sub-header">Track brands, products, and topics across AI-powered search engines</p>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        """
        LLM Rank Tracker is a versatile open-source tool to analyze how brands, products, 
        and topics appear in AI-generated search results. Get insights on brand visibility,
        market analysis, consumer trends, and more.
        
        [View on GitHub](https://github.com/yourusername/llm-rank-tracker)
        """
    )
    
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password", 
                            help="Get your free API key from https://console.groq.com/keys")
    
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    
    # Model selection
    model_option = st.selectbox(
        "LLM Model",
        ["llama3-8b-8192", "llama3-70b-8192"],
        help="Select which Groq model to use. Larger models may provide more accurate results."
    )
    
    # Add temperature slider
    temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.2, 
        step=0.1,
        help="Lower values = more consistent responses, Higher values = more creative responses"
    )
    
    # View history button
    if st.button("View Analysis History") and st.session_state.history:
        st.session_state.show_history = True
    
    # Clear history button
    if st.button("Clear History") and st.session_state.history:
        st.session_state.history = []
        st.session_state.comparison_data = {}
        st.success("History cleared!")

# Tabs for different features
tabs = st.tabs(["Brand Analysis", "Historical Data", "Comparison"])

# Brand Analysis Tab
with tabs[0]:
    st.markdown("### Enter Topic and Category")
    
    # Two columns for input fields
    col1, col2 = st.columns(2)
    with col1:
        # Use text area for multiple brands or topics
        brands_input = st.text_area("Brand, Product, or Topic", 
                                 placeholder="Enter brands or topics to analyze, one per line (e.g., iPhone\nSamsung\nAI in healthcare)",
                                 help="Enter one or more brands, products, or topics to track, each on a new line")
        
        # Process the brands input
        brands = [brand.strip() for brand in brands_input.split('\n') if brand.strip()]
        if not brands:
            st.warning("Please enter at least one brand or topic to track")
            brand_name = ""  # Default empty value
        else:
            # Use the first brand as the primary one for queries that need a single brand
            brand_name = brands[0]
            if len(brands) > 1:
                st.info(f"Primary subject for analysis: {brand_name}")
    
    with col2:
        col2a, col2b = st.columns(2)
        with col2a:
            category = st.text_input("Category or Context", 
                               help="Enter the category, industry, or context for your query (e.g., smartphones, healthcare, marketing)")
        with col2b:
            # Country selection dropdown
            countries = [
                "Global (No specific country)",
                # Europe
                "United Kingdom", "Germany", "France", "Italy", "Spain", "Netherlands", 
                "Sweden", "Switzerland", "Poland", "Belgium",
                # North America
                "United States", "Canada", "Mexico",
                # Asia
                "Japan", "China", "South Korea", "India", "Singapore", 
                "Hong Kong", "Taiwan", "Malaysia", "Thailand"
            ]
            selected_country = st.selectbox(
                "Target Country/Region",
                countries,
                help="Select a country or region to focus the analysis on"
            )
    
    # Query type selection
    query_type = st.selectbox(
        "Query Type",
        [
            "Top brands in category",
            "Best products for specific use case",
            "Popular alternatives to a brand",
            "General market analysis",
            "Technology or feature comparison",
            "Consumer insights",
            "Custom query"
        ]
    )
    
    # Custom query input (appears only if custom query is selected)
    custom_query = ""
    if query_type == "Custom query":
        custom_query = st.text_area("Enter your custom query", 
                                   help="Write a custom prompt about the brand and category")
    
    # Run analysis button
    if st.button("Run Analysis", type="primary", disabled=not (brands and category and (api_key or os.environ.get("GROQ_API_KEY")))):
        with st.spinner("Querying AI engines and analyzing responses..."):
            # Generate the prompt based on inputs
            prompt = generate_prompt(brand_name, category, query_type, custom_query, all_brands=brands, country=selected_country)
            
            # Display the prompt being used
            with st.expander("View Prompt"):
                st.code(prompt)
            
            # Query the LLM
            llm_response = query_groq_llm(prompt, model=model_option, temperature=temperature)
            
            # Create a container for multi-brand results
            all_brand_results = {}
            
            # Process each brand
            for brand in brands:
                # Extract and analyze brand mentions for each brand - pass category for context
                brand_results = extract_brand_mentions(llm_response, target_brand=brand, category=category)
                all_brand_results[brand] = brand_results
            
            # Get the detected category from the primary brand analysis
            detected_category = all_brand_results[brand_name].get("category", "smartphone")
            
            # Add timestamp and query info for the primary brand
            timestamp = datetime.now()
            primary_results = all_brand_results[brand_name]
            analysis_result = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "brand": brand_name,
                "all_brands": brands,
                "category": category,
                "detected_category": detected_category,
                "country": selected_country,
                "query_type": query_type,
                "model": model_option,
                "is_mentioned": primary_results["is_mentioned"],
                "rank": primary_results["rank"],
                "mention_count": primary_results["mention_count"],
                "sentiment": primary_results["sentiment"],
                "raw_response": llm_response,
                "mention_contexts": primary_results["mention_contexts"],
                "other_brands": primary_results["other_brands"],
                "multi_brand_results": all_brand_results
            }
            
            # Add to history
            st.session_state.history.append(analysis_result)
            
            # Update comparison data for all analyzed brands
            for brand, results in all_brand_results.items():
                if brand not in st.session_state.comparison_data:
                    st.session_state.comparison_data[brand] = []
                
                st.session_state.comparison_data[brand].append({
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "rank": results["rank"] if results["rank"] > 0 else 10,  # Default to 10 if not ranked
                    "mentions": results["mention_count"],
                    "sentiment": results["sentiment"],
                    "category": category,
                    "detected_category": detected_category,
                    "country": selected_country
                })
            
            # Display results
            st.markdown("## Analysis Results")
            
            # Display summary for all analyzed brands
            st.markdown("### Brand Visibility Summary")
            
            # Create a table to display all brand metrics
            brand_summary_data = []
            for brand, results in all_brand_results.items():
                brand_summary_data.append({
                    "Brand": brand,
                    "Mentioned": "✅" if results["is_mentioned"] else "❌",
                    "Rank": results["rank"] if results["rank"] > 0 else "Not ranked",
                    "Mentions": results["mention_count"],
                    "Sentiment": results["sentiment"].capitalize()
                })
            
            brand_summary_df = pd.DataFrame(brand_summary_data)
            st.dataframe(brand_summary_df, use_container_width=True)
            
            # Create metrics for the primary brand
            st.markdown(f"### Primary Brand: {brand_name}")
            
            # Create three columns for key metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown("""
                <div class="stat-box">
                    <h3>Visibility Status</h3>
                    <h2 style="color: {};">{}</h2>
                </div>
                """.format(
                    "#2ecc71" if primary_results["is_mentioned"] else "#e74c3c",
                    "✅ Mentioned" if primary_results["is_mentioned"] else "❌ Not Mentioned"
                ), unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown("""
                <div class="stat-box">
                    <h3>Rank Position</h3>
                    <h2>{}</h2>
                </div>
                """.format(primary_results["rank"] if primary_results["rank"] > 0 else "Not ranked"), unsafe_allow_html=True)
            
            with metric_col3:
                sentiment_color = {
                    "positive": "#2ecc71",
                    "neutral": "#f39c12",
                    "negative": "#e74c3c"
                }.get(primary_results["sentiment"], "#f39c12")
                
                sentiment_emoji = {
                    "positive": "😀",
                    "neutral": "😐",
                    "negative": "😟"
                }.get(primary_results["sentiment"], "😐")
                
                st.markdown("""
                <div class="stat-box">
                    <h3>Sentiment</h3>
                    <h2 style="color: {};">{} {}</h2>
                </div>
                """.format(
                    sentiment_color,
                    sentiment_emoji,
                    primary_results["sentiment"].capitalize()
                ), unsafe_allow_html=True)
            
            # Display brand mention chart
            st.markdown("### Brand Mentions Visualization")
            
            # Create a dataframe for the chart
            brands_data = {"Brand": [], "Mentions": [], "Type": []}
            
            # Add all analyzed brands data
            for brand, results in all_brand_results.items():
                if results["is_mentioned"]:
                    brands_data["Brand"].append(brand)
                    brands_data["Mentions"].append(results["mention_count"])
                    brands_data["Type"].append("Tracked Brand")
            
            # Add other brands data (only including relevant ones from the same category)
            all_tracked_brands = set(brands)
            relevant_other_brands = {}
            
            # First pass - collect all other brands
            for other_brand, count in primary_results["other_brands"].items():
                if other_brand not in all_tracked_brands:
                    relevant_other_brands[other_brand] = count
            
            # Add top relevant brands to the chart data
            for other_brand, count in list(relevant_other_brands.items())[:5]:
                brands_data["Brand"].append(other_brand)
                brands_data["Mentions"].append(count)
                brands_data["Type"].append("Other Brand")
            
            # Create dataframe
            brands_df = pd.DataFrame(brands_data)
            
            if not brands_df.empty:
                # Create column chart
                fig = px.bar(
                    brands_df, 
                    x="Brand", 
                    y="Mentions", 
                    title=f"Brand Mentions Comparison ({detected_category.capitalize()} Category)",
                    color="Type",
                    color_discrete_map={
                        "Tracked Brand": "#4B61D1",
                        "Other Brand": "#95A5A6"
                    },
                    category_orders={"Type": ["Tracked Brand", "Other Brand"]}
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title="Brand",
                    yaxis_title="Number of Mentions",
                    plot_bgcolor="white"
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
            
            # Display pie chart for brand distribution
            if relevant_other_brands or len(brands) > 1:
                st.markdown("### Brand Visibility Distribution")
                
                # Create pie chart data
                pie_data = {"Brand": [], "Mentions": []}
                
                # Add all tracked brands
                for brand, results in all_brand_results.items():
                    if results["is_mentioned"]:
                        pie_data["Brand"].append(brand)
                        pie_data["Mentions"].append(results["mention_count"])
                
                # Add other relevant brands
                for other_brand, count in list(relevant_other_brands.items())[:7]:  # Limit to top 7 other brands
                    pie_data["Brand"].append(other_brand)
                    pie_data["Mentions"].append(count)
                
                if pie_data["Brand"]:  # Only create chart if we have data
                    # Create dataframe
                    pie_df = pd.DataFrame(pie_data)
                    
                    # Create pie chart
                    fig_pie = px.pie(
                        pie_df,
                        values="Mentions",
                        names="Brand",
                        title=f"Distribution of Brand Mentions ({detected_category.capitalize()} Category)",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    # Update layout
                    fig_pie.update_layout(
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Display the raw response
            with st.expander("Raw AI Response"):
                st.markdown(llm_response)
            
            # Display the brand mentions analysis
            st.markdown("### Brand Mentions Analysis")
            
            # Create tabs for each brand's mention contexts
            brand_tabs = st.tabs(brands)
            
            # Display mention contexts for each brand in its own tab
            for idx, brand in enumerate(brands):
                with brand_tabs[idx]:
                    results = all_brand_results[brand]
                    if results["is_mentioned"]:
                        # Display the context of mentions
                        st.markdown("#### Mention Contexts")
                        for midx, context in enumerate(results["mention_contexts"]):
                            st.markdown(f"**Mention {midx+1}:** {context}")
                    else:
                        st.error(f"❌ Brand '{brand}' was not mentioned in the AI response.")
                        
                        st.markdown("#### Other brands mentioned:")
                        for oidx, (other_brand, count) in enumerate(results["other_brands"].items()):
                            if other_brand not in all_tracked_brands or other_brand == brand:
                                st.markdown(f"{oidx+1}. **{other_brand}** - {count} mentions")
            
            # Show recommendations based on analysis
            st.markdown("### Recommendations")
            
            # Create recommendations based on the query type and results
            if query_type in ["Top brands in category", "Best products for specific use case", "Popular alternatives to a brand"]:
                # Brand-focused recommendations
                if not primary_results["is_mentioned"]:
                    st.markdown("""
                    - Consider creating more content highlighting your brand in this category
                    - Look at the top mentioned brands to understand their visibility advantage
                    - Try different query formulations to see if your brand appears elsewhere
                    - Analyze competitor content to understand what's driving their visibility
                    """)
                elif primary_results["rank"] > 3:
                    st.markdown("""
                    - Your brand is mentioned but not at the top - consider content strategies to improve positioning
                    - Analyze the context of your mention to understand how your brand is perceived
                    - Monitor regularly to track changes in visibility
                    - Focus on differentiating features that competitors may not have
                    """)
                else:
                    st.markdown("""
                    - Your brand has good visibility in this query context
                    - Continue monitoring to maintain this position
                    - Consider expanding to related categories
                    - Leverage this strength in marketing materials
                    """)
            else:
                # General query recommendations
                st.markdown("""
                - Save this analysis for future reference and tracking of trends
                - Run periodic analyses to monitor changes in AI perception of this topic
                - Compare results across different categories/contexts for broader insights
                - Use the insights for content creation, market positioning, or business strategy
                """)
            
            # Add download options
            st.markdown("### Download Results")
            
            # Create a DataFrame from the analysis
            download_data = {
                "Attribute": ["Primary Brand", "Category", "Detected Category", "Country/Region", "Timestamp", "All Brands Analyzed"],
                "Value": [
                    brand_name, 
                    category,
                    detected_category,
                    selected_country,
                    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    ", ".join(brands)
                ]
            }
            download_df = pd.DataFrame(download_data)
            
            # Create detailed results for all brands
            all_brands_data = []
            for brand, results in all_brand_results.items():
                all_brands_data.append({
                    "Brand": brand,
                    "Mentioned": "Yes" if results["is_mentioned"] else "No",
                    "Rank": str(results["rank"]) if results["rank"] > 0 else "Not ranked",
                    "Mentions": results["mention_count"],
                    "Sentiment": results["sentiment"].capitalize()
                })
            all_brands_df = pd.DataFrame(all_brands_data)
            
            # Add other brands section
            other_brands_data = {"Brand": [], "Mentions": []}
            for other_brand, count in relevant_other_brands.items():
                other_brands_data["Brand"].append(other_brand)
                other_brands_data["Mentions"].append(count)
            
            if other_brands_data["Brand"]:
                other_brands_df = pd.DataFrame(other_brands_data)
            else:
                other_brands_df = pd.DataFrame({"Brand": [], "Mentions": []})
            
            # CSV download buttons
            summary_csv = download_df.to_csv(index=False)
            brands_csv = all_brands_df.to_csv(index=False)
            other_brands_csv = other_brands_df.to_csv(index=False)
            
            # Function to create a download link
            def get_download_link(data, filename, text):
                b64 = base64.b64encode(data.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
                return href
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(get_download_link(brands_csv, f"llm_rank_tracker_brands_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv", 
                                           "📥 Download Brand Analysis as CSV"), unsafe_allow_html=True)
            
            with col2:
                st.markdown(get_download_link(other_brands_csv, f"llm_rank_tracker_competitors_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv", 
                                           "📥 Download Competitor Data as CSV"), unsafe_allow_html=True)
            
            # JSON export option
            json_data = json.dumps({
                "analysis": {
                    "primary_brand": brand_name,
                    "all_brands": brands,
                    "category": category,
                    "detected_category": detected_category,
                    "country": selected_country,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "query_type": query_type,
                },
                "brand_results": {brand: {
                    "is_mentioned": results["is_mentioned"],
                    "rank": results["rank"],
                    "mention_count": results["mention_count"],
                    "sentiment": results["sentiment"],
                    "mention_contexts": results["mention_contexts"]
                } for brand, results in all_brand_results.items()},
                "other_brands": relevant_other_brands
            }, indent=2)
            
            st.markdown(get_download_link(json_data, f"llm_rank_tracker_complete_{timestamp.strftime('%Y%m%d_%H%M%S')}.json", 
                                       "📥 Download Complete Data as JSON"), unsafe_allow_html=True)

# Historical Data Tab
with tabs[1]:
    st.markdown("### Historical Analysis Data")
    
    if not st.session_state.history:
        st.info("No historical data available yet. Run some analyses to see data here.")
    else:
        # Create a dataframe from history
        history_data = []
        for item in st.session_state.history:
            if "all_brands" in item:
                # For multi-brand analyses
                brands_analyzed = len(item.get("all_brands", []))
                country = item.get("country", "Global (No specific country)")
                detected_cat = item.get("detected_category", "")
                history_data.append({
                    "Timestamp": item["timestamp"],
                    "Primary Brand": item["brand"],
                    "Brands Analyzed": brands_analyzed,
                    "Category": item["category"],
                    "Detected Category": detected_cat,
                    "Country/Region": country,
                    "Mentioned": "Yes" if item["is_mentioned"] else "No",
                    "Rank": item["rank"] if item["rank"] > 0 else "Not ranked",
                    "Mentions": item["mention_count"],
                    "Sentiment": item["sentiment"].capitalize()
                })
            else:
                # For single brand analyses (backward compatibility)
                history_data.append({
                    "Timestamp": item["timestamp"],
                    "Primary Brand": item["brand"],
                    "Brands Analyzed": 1,
                    "Category": item["category"],
                    "Detected Category": "",
                    "Country/Region": "Global (No specific country)",
                    "Mentioned": "Yes" if item["is_mentioned"] else "No",
                    "Rank": item["rank"] if item["rank"] > 0 else "Not ranked",
                    "Mentions": item["mention_count"],
                    "Sentiment": item["sentiment"].capitalize()
                })
        
        history_df = pd.DataFrame(history_data)
        
        # Display the history table
        st.dataframe(history_df, use_container_width=True)
        
        # Create line chart for historical data
        if len(st.session_state.history) > 1:
            st.markdown("### Visibility Trends")
            
            # Prepare data for line chart
            line_data = []
            for item in st.session_state.history:
                if item["is_mentioned"]:
                    line_data.append({
                        "Timestamp": item["timestamp"],
                        "Brand": item["brand"],
                        "Rank": item["rank"] if item["rank"] > 0 else 10,  # Default to 10 if not ranked
                        "Mentions": item["mention_count"]
                    })
            
            if line_data:
                line_df = pd.DataFrame(line_data)
                
                # Create tabs for different metrics
                metric_tabs = st.tabs(["Rank Position", "Mention Count"])
                
                # Rank Position tab
                with metric_tabs[0]:
                    fig_rank = px.line(
                        line_df,
                        x="Timestamp",
                        y="Rank",
                        color="Brand",
                        markers=True,
                        title="Rank Position Over Time (Lower is Better)",
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    # Invert y-axis since lower rank is better
                    fig_rank.update_layout(
                        yaxis=dict(
                            autorange="reversed",
                            title="Rank Position"
                        ),
                        xaxis_title="Date & Time",
                        legend_title="Brand"
                    )
                    
                    st.plotly_chart(fig_rank, use_container_width=True)
                
                # Mention Count tab
                with metric_tabs[1]:
                    fig_mentions = px.line(
                        line_df,
                        x="Timestamp",
                        y="Mentions",
                        color="Brand",
                        markers=True,
                        title="Mention Count Over Time",
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    fig_mentions.update_layout(
                        yaxis_title="Number of Mentions",
                        xaxis_title="Date & Time",
                        legend_title="Brand"
                    )
                    
                    st.plotly_chart(fig_mentions, use_container_width=True)
            else:
                st.info("No trend data available yet. Run more analyses to see trends.")
        
        # Download all history data
        st.markdown("### Download All Historical Data")
        
        csv_history = history_df.to_csv(index=False)
        b64 = base64.b64encode(csv_history.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="llm_rank_tracker_history.csv">📥 Download All History as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Create a more detailed JSON export with all data
        json_history = json.dumps(st.session_state.history, indent=2, default=str)
        b64_json = base64.b64encode(json_history.encode()).decode()
        href_json = f'<a href="data:file/json;base64,{b64_json}" download="llm_rank_tracker_history.json">📥 Download Complete History as JSON</a>'
        st.markdown(href_json, unsafe_allow_html=True)

# Comparison Tab
with tabs[2]:
    st.markdown("### Brand Comparison")
    
    if not st.session_state.comparison_data:
        st.info("No comparison data available yet. Run some analyses to compare brands.")
    else:
        # Create selector for brands to compare
        available_brands = list(st.session_state.comparison_data.keys())
        
        col1, col2 = st.columns(2)
        with col1:
            selected_brands = st.multiselect(
                "Select brands to compare",
                options=available_brands,
                default=available_brands[:min(3, len(available_brands))],
                help="Choose brands to include in the comparison"
            )
        
        with col2:
            # Get all countries in the comparison data
            all_countries = set()
            for brand, entries in st.session_state.comparison_data.items():
                for entry in entries:
                    all_countries.add(entry.get("country", "Global (No specific country)"))
            
            # Allow filtering by country
            selected_countries = st.multiselect(
                "Filter by country/region",
                options=sorted(list(all_countries)),
                default=[],
                help="Select countries to include in the comparison (leave empty for all)"
            )
        
        if selected_brands:
            # Create comparison data
            comparison_data = []
            for brand in selected_brands:
                for entry in st.session_state.comparison_data[brand]:
                    # Apply country filter if selected
                    entry_country = entry.get("country", "Global (No specific country)")
                    if selected_countries and entry_country not in selected_countries:
                        continue
                        
                    comparison_data.append({
                        "Brand": brand,
                        "Timestamp": entry["timestamp"],
                        "Rank": entry["rank"],
                        "Mentions": entry["mentions"],
                        "Sentiment": entry["sentiment"].capitalize(),
                        "Category": entry["category"],
                        "Detected Category": entry.get("detected_category", ""),
                        "Country": entry_country
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create tabs for different comparison views
            comp_tabs = st.tabs(["Rank Comparison", "Mentions Comparison", "Sentiment Analysis"])
            
            # Rank Comparison tab
            with comp_tabs[0]:
                if len(comparison_df) > 0:
                    fig_rank_comp = px.line(
                        comparison_df,
                        x="Timestamp",
                        y="Rank",
                        color="Brand",
                        markers=True,
                        title="Rank Position Comparison (Lower is Better)",
                        color_discrete_sequence=px.colors.qualitative.Vivid
                    )
                    
                    # Invert y-axis since lower rank is better
                    fig_rank_comp.update_layout(
                        yaxis=dict(
                            autorange="reversed",
                            title="Rank Position"
                        ),
                        xaxis_title="Date & Time",
                        legend_title="Brand"
                    )
                    
                    st.plotly_chart(fig_rank_comp, use_container_width=True)
                else:
                    st.info("No rank data available for the selected brands.")
            
            # Mentions Comparison tab
            with comp_tabs[1]:
                if len(comparison_df) > 0:
                    fig_mentions_comp = px.line(
                        comparison_df,
                        x="Timestamp",
                        y="Mentions",
                        color="Brand",
                        markers=True,
                        title="Mention Count Comparison",
                        color_discrete_sequence=px.colors.qualitative.Vivid
                    )
                    
                    fig_mentions_comp.update_layout(
                        yaxis_title="Number of Mentions",
                        xaxis_title="Date & Time",
                        legend_title="Brand"
                    )
                    
                    st.plotly_chart(fig_mentions_comp, use_container_width=True)
                    
                    # Add a bar chart for average mentions
                    avg_mentions = comparison_df.groupby("Brand")["Mentions"].mean().reset_index()
                    avg_mentions["Mentions"] = avg_mentions["Mentions"].round(1)
                    
                    fig_avg_mentions = px.bar(
                        avg_mentions,
                        x="Brand",
                        y="Mentions",
                        color="Brand",
                        title="Average Mentions by Brand",
                        color_discrete_sequence=px.colors.qualitative.Vivid
                    )
                    
                    st.plotly_chart(fig_avg_mentions, use_container_width=True)
                else:
                    st.info("No mention data available for the selected brands.")
            
            # Sentiment Analysis tab
            with comp_tabs[2]:
                if len(comparison_df) > 0:
                    # Convert sentiment to numeric
                    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
                    comparison_df["Sentiment_Score"] = comparison_df["Sentiment"].map(lambda x: sentiment_map.get(x, 0))
                    
                    # Create a grouped bar chart for sentiment distribution
                    sentiment_counts = comparison_df.groupby(["Brand", "Sentiment"]).size().reset_index(name="Count")
                    
                    fig_sentiment = px.bar(
                        sentiment_counts,
                        x="Brand",
                        y="Count",
                        color="Sentiment",
                        title="Sentiment Distribution by Brand",
                        color_discrete_map={
                            "Positive": "#2ecc71",
                            "Neutral": "#f39c12",
                            "Negative": "#e74c3c"
                        },
                        barmode="group"
                    )
                    
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                    
                    # Show average sentiment score
                    avg_sentiment = comparison_df.groupby("Brand")["Sentiment_Score"].mean().reset_index()
                    avg_sentiment["Sentiment_Score"] = avg_sentiment["Sentiment_Score"].round(2)
                    
                    # Rename for display
                    avg_sentiment.columns = ["Brand", "Average Sentiment (-1 to 1)"]
                    
                    st.markdown("### Average Sentiment Score by Brand")
                    st.markdown("*Scale: -1 (Negative) to 1 (Positive)*")
                    st.dataframe(avg_sentiment, use_container_width=True)
                else:
                    st.info("No sentiment data available for the selected brands.")
            
            # Download comparison data
            st.markdown("### Download Comparison Data")
            
            csv_comparison = comparison_df.to_csv(index=False)
            b64_comp = base64.b64encode(csv_comparison.encode()).decode()
            href_comp = f'<a href="data:file/csv;base64,{b64_comp}" download="llm_rank_tracker_comparison.csv">📥 Download Comparison Data as CSV</a>'
            st.markdown(href_comp, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
    LLM Rank Tracker is an open-source tool. This tool is for educational purposes.
    <br>Created with ❤️ using Streamlit and Groq.
    </div>
    """, 
    unsafe_allow_html=True
)
