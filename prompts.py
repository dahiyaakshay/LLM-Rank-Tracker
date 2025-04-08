def generate_prompt(brand_name: str, category: str, query_type: str, custom_query: str = "", all_brands: list = None) -> str:
    """
    Generate a prompt for the LLM based on the input parameters.
    
    Args:
        brand_name: The primary brand or product name to track
        category: The category or context for the query
        query_type: The type of query to generate
        custom_query: A custom query (used if query_type is "Custom query")
        all_brands: List of all brands to analyze (optional)
        
    Returns:
        A formatted prompt string
    """
    # Use the provided list of brands or default to just the primary brand
    if all_brands is None:
        all_brands = [brand_name]
        
    # If custom query is provided and selected, use it
    if query_type == "Custom query" and custom_query:
        return custom_query
    
    # Standard system instructions to ensure consistent formatting
    system_instruction = (
        "You are a knowledgeable AI assistant specialized in providing accurate, "
        "up-to-date information about brands, products, and market trends. "
        "Please respond with a clear, structured answer that includes:"
        "\n\n1. A numbered list of relevant brands/products (when appropriate)"
        "\n2. Brief descriptions highlighting key features, strengths, and weaknesses"
        "\n3. A clear indication of ranking or comparison when listing multiple options"
        "\n\nBe specific about brand names and ensure your response is comprehensive but "
        "focuses on the most relevant and popular options. If you mention a brand, do so "
        "explicitly by name rather than with pronouns or vague references."
    )
    
    # General query system instruction
    general_system_instruction = (
        "You are a knowledgeable AI assistant specialized in providing accurate, "
        "up-to-date information about various topics including technology, market trends, "
        "consumer behavior, and industry analysis. "
        "Please respond with a clear, structured answer that is comprehensive yet focused "
        "on the most relevant information. When appropriate, include examples, data points, "
        "or comparisons to illustrate your points."
    )
    
    # Generate prompt based on query type
    if query_type == "Top brands in category":
        prompt = f"{system_instruction}\n\n"
        prompt += f"What are the top 10 brands or companies offering {category} in 2024? "
        prompt += f"Please provide a ranked, numbered list with a brief description of each, "
        prompt += f"mentioning their key products, features, and what makes them stand out. "
        
        # Add specific instructions for the brands we're tracking
        if len(all_brands) > 1:
            brands_str = ", ".join(all_brands)
            prompt += f"Include these brands in your analysis if they're relevant to this category: {brands_str}. "
        else:
            prompt += f"Include {brand_name} in your analysis if it's relevant to this category. "
            
        prompt += f"Be sure to indicate the relative ranking or market position of each brand clearly."
        
    elif query_type == "Best products for specific use case":
        prompt = f"{system_instruction}\n\n"
        prompt += f"What are the best {category} products available today? "
        prompt += f"Please provide a comprehensive, ranked list of the top 10 options with "
        prompt += f"descriptions of their key features, strengths, and ideal use cases. "
        
        # Add specific instructions for the brands we're tracking
        if len(all_brands) > 1:
            brands_str = ", ".join(all_brands)
            prompt += f"Include products from these brands in your analysis if they're relevant: {brands_str}. "
        else:
            prompt += f"Include products from {brand_name} in your analysis if they're relevant. "
            
        prompt += f"For each product, clearly indicate why it ranks in its position and what "
        prompt += f"specific advantages it offers over lower-ranked alternatives."
        
    elif query_type == "Popular alternatives to a brand":
        prompt = f"{system_instruction}\n\n"
        prompt += f"What are the top alternatives to {brand_name} in the {category} category? "
        prompt += f"Please provide a ranked, numbered list of 8-10 competing products or services with "
        prompt += f"detailed explanations of how they compare to {brand_name}. For each alternative, include:"
        prompt += f"\n1. Key differentiating features"
        prompt += f"\n2. Pricing comparison (if available)"
        prompt += f"\n3. Target audience or use case"
        prompt += f"\n4. Advantages and disadvantages compared to {brand_name}"
        
        # Include request to mention other brands if they're in our tracking list
        if len(all_brands) > 1:
            other_brands = [b for b in all_brands if b != brand_name]
            brands_str = ", ".join(other_brands)
            prompt += f"\n\nPlease be sure to include {brands_str} in your analysis if they are relevant alternatives."
    
    elif query_type == "General market analysis":
        prompt = f"{general_system_instruction}\n\n"
        prompt += f"Provide a detailed market analysis of the {category} industry or sector. Include information about:"
        prompt += f"\n1. Current market trends and growth projections"
        prompt += f"\n2. Key players and their market share"
        prompt += f"\n3. Consumer behavior and preferences"
        prompt += f"\n4. Technological innovations affecting the market"
        prompt += f"\n5. Future outlook and predictions"
        
        if all_brands and len(all_brands) > 0:
            brands_str = ", ".join(all_brands)
            prompt += f"\n\nPlease include information about these specific brands if relevant: {brands_str}."
    
    elif query_type == "Technology or feature comparison":
        prompt = f"{general_system_instruction}\n\n"
        prompt += f"Compare and contrast the key technologies, features, or approaches in the {category} field. Include:"
        prompt += f"\n1. Major technological approaches or methodologies"
        prompt += f"\n2. Pros and cons of each approach"
        prompt += f"\n3. Use cases where each technology excels"
        prompt += f"\n4. Future developments or improvements expected"
        
        if all_brands and len(all_brands) > 0:
            brands_str = ", ".join(all_brands)
            prompt += f"\n\nIf appropriate, reference how these brands implement the technologies: {brands_str}."
    
    elif query_type == "Consumer insights":
        prompt = f"{general_system_instruction}\n\n"
        prompt += f"Analyze consumer behavior, preferences, and trends related to {category}. Include information about:"
        prompt += f"\n1. Key consumer segments and their preferences"
        prompt += f"\n2. Purchasing patterns and decision factors"
        prompt += f"\n3. Emerging consumer trends"
        prompt += f"\n4. Pain points and unmet needs"
        prompt += f"\n5. How consumer behavior has evolved recently"
        
        if all_brands and len(all_brands) > 0:
            brands_str = ", ".join(all_brands)
            prompt += f"\n\nWhen relevant, mention how consumers perceive these brands: {brands_str}."
        
    else:
        # Default fallback prompt
        prompt = f"{system_instruction}\n\n"
        
        if len(all_brands) > 1:
            # Multiple brands case
            brands_str = ", ".join(all_brands)
            prompt += f"Please provide detailed information about these brands in the {category} category: {brands_str}. "
            prompt += f"For each brand, include its market positioning, key strengths and weaknesses, "
            prompt += f"and how it compares to other major products in this category. "
            prompt += f"If applicable, include where each brand ranks among competitors and why."
        else:
            # Single brand case
            prompt += f"Please provide detailed information about {brand_name} in the context of {category}. "
            prompt += f"Include how it compares to other major brands or products in this category, "
            prompt += f"its market positioning, key strengths and weaknesses, and any notable features "
            prompt += f"that differentiate it from competitors. If applicable, include where it ranks "
            prompt += f"among competitors and why."
    
    # Add a request for clear formatting to aid in analysis
    if query_type in ["Top brands in category", "Best products for specific use case", "Popular alternatives to a brand"]:
        prompt += "\n\nWhen listing brands or products, please use a clear numbered format (1., 2., etc.) and ensure each brand name is explicitly mentioned at the beginning of its description."
    else:
        prompt += "\n\nPlease organize your response with clear headings and numbered lists where appropriate to make the information easy to analyze."
    
    return prompt
