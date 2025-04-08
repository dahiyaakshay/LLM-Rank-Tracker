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
    prompt += "\n\nWhen listing brands or products, please use a clear numbered format (1., 2., etc.) and ensure each brand name is explicitly mentioned at the beginning of its description."
    
    return prompt
