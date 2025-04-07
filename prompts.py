def generate_prompt(brand_name: str, category: str, query_type: str, custom_query: str = "") -> str:
    """
    Generate a prompt for the LLM based on the input parameters.
    
    Args:
        brand_name: The brand or product name to track
        category: The category or context for the query
        query_type: The type of query to generate
        custom_query: A custom query (used if query_type is "Custom query")
        
    Returns:
        A formatted prompt string
    """
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
        prompt += f"Include {brand_name} in your analysis if it's relevant to this category. "
        prompt += f"Be sure to indicate the relative ranking or market position of each brand clearly."
        
    elif query_type == "Best products for specific use case":
        prompt = f"{system_instruction}\n\n"
        prompt += f"What are the best {category} products available today? "
        prompt += f"Please provide a comprehensive, ranked list of the top 10 options with "
        prompt += f"descriptions of their key features, strengths, and ideal use cases. "
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
        
    else:
        # Default fallback prompt
        prompt = f"{system_instruction}\n\n"
        prompt += f"Please provide detailed information about {brand_name} in the context of {category}. "
        prompt += f"Include how it compares to other major brands or products in this category, "
        prompt += f"its market positioning, key strengths and weaknesses, and any notable features "
        prompt += f"that differentiate it from competitors. If applicable, include where it ranks "
        prompt += f"among competitors and why."
    
    # Add a request for clear formatting to aid in analysis
    prompt += "\n\nWhen listing brands or products, please use a clear numbered format (1., 2., etc.) and ensure each brand name is explicitly mentioned at the beginning of its description."
    
    return prompt
