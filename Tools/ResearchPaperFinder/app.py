from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_exa import ExaSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

# Load API keys
load_dotenv()
exa_api_key = os.getenv("EXA_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=google_api_key,
    temperature=0.1
)

# Tools
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)
exa_tool = ExaSearchResults(exa_api_key=exa_api_key)

# College affiliation strings
COLLEGE_PATTERNS = [
    "Sri Krishna Arts and Science College",
    "Sri Krishna Arts & Science College", 
    "Sri Krishna Arts and Science College, Coimbatore",
    "SKASC",
]

def affiliation_matches(text: str):
    """Check if text contains SKASC affiliation"""
    text = (text or "").lower()
    return any(p.lower() in text for p in COLLEGE_PATTERNS)

def analyze_with_gemini(query, results):
    """Use Gemini to analyze and filter results intelligently"""
    try:
        system_prompt = """You are a research assistant specialized in identifying academic papers from 
        Sri Krishna Arts and Science College (SKASC), Coimbatore. Analyze search results and determine:
        1. Which results are actually research papers from SKASC faculty/students
        2. Extract key information: title, authors, publication details
        3. Provide a concise summary of each relevant paper
        4. Ignore results that are not research papers or don't have SKASC affiliation"""
        
        human_prompt = f"""
        Query: {query}
        
        Search Results: {str(results)[:3000]}  # Truncate to avoid token limits
        
        Please analyze these results and return only the papers that are:
        - Research papers, theses, or technical reports
        - Authored by SKASC faculty or students (look for affiliation patterns)
        - Include: title, authors, publication info, summary, and relevance to query
        """
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        return response.content
        
    except Exception as e:
        print(f"Gemini analysis error: {e}")
        return None

def generate_search_query_with_gemini(user_query):
    """Use Gemini to generate optimized search queries"""
    try:
        prompt = f"""
        Convert this research query into an optimized search query for academic databases:
        Original: "{user_query}"
        
        Focus on finding research papers from "Sri Krishna Arts and Science College" (SKASC).
        Include keywords that would appear in academic papers. Make it specific to research publications.
        
        Return only the optimized search query, nothing else.
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
        
    except Exception as e:
        print(f"Gemini query generation error: {e}")
        return user_query  # Fallback to original query

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.form["query"]
    
    # Generate optimized search query with Gemini
    optimized_query = generate_search_query_with_gemini(user_query)
    exa_prompt = f"""Find research papers from Sri Krishna Arts and Science College (SKASC) about: {optimized_query}
    Focus on papers where authors are affiliated with SKASC. Look for research papers, theses, conference proceedings."""
    
    # 1️⃣ Try EXA first with Gemini-enhanced query
    try:
        exa_result = exa_tool._run(
            query=exa_prompt,
            num_results=15,  # Get more results for better filtering
            text_contents_options=True,
            highlights=True,
        )

        filtered = []
        
        if hasattr(exa_result, 'results'):
            # Use Gemini to analyze and filter results
            gemini_analysis = analyze_with_gemini(user_query, exa_result.results)
            
            for r in exa_result.results:
                # Safely extract attributes
                title = getattr(r, "title", "") or ""
                text = getattr(r, "text", "") or ""
                
                authors_attr = getattr(r, "authors", [])
                if isinstance(authors_attr, list):
                    authors = " ".join([str(a) for a in authors_attr if a]) if authors_attr else ""
                else:
                    authors = str(authors_attr) if authors_attr else ""
                
                summary = getattr(r, "summary", "") or ""
                highlights = getattr(r, "highlights", "") or ""
                url = getattr(r, "url", "#") or "#"
                
                combined = " ".join([str(field) for field in [title, text, authors, summary, highlights] if field])
                
                card = {
                    "title": title or "(no title)",
                    "authors": authors or "Unknown",
                    "excerpt": summary or highlights or (text[:500] if text else "No excerpt available"),
                    "link": url,
                    "affiliation_match": affiliation_matches(combined),
                    "confidence": "high" if affiliation_matches(combined) else "low"
                }

                if card["affiliation_match"]:
                    filtered.append(card)

        if filtered:
            return jsonify({
                "Exa": filtered,
                "gemini_analysis": gemini_analysis[:1000] if gemini_analysis else "No additional analysis available"
            })
        else:
            # If no filtered results, try fallbacks
            pass

    except Exception as e:
        print("EXA error:", e)

    # 2️⃣ Fallback to Arxiv
    try:
        arxiv_result = arxiv.run(tool_input=user_query)
        if arxiv_result:
            return jsonify({
                "Arxiv": [{
                    "title": "Arxiv Research Paper",
                    "authors": "See publication details",
                    "excerpt": arxiv_result[:800],
                    "link": f"https://arxiv.org/search/?query={user_query.replace(' ', '+')}",
                    "affiliation_match": affiliation_matches(arxiv_result),
                    "confidence": "medium"
                }],
                "note": "Found on Arxiv - verify SKASC affiliation manually"
            })
    except Exception as e:
        print("Arxiv error:", e)

    # 3️⃣ Final fallback: Wikipedia with Gemini enhancement
    try:
        wiki_result = wiki.run(tool_input=user_query)
        if wiki_result:
            # Use Gemini to extract SKASC-relevant info from Wikipedia
            wiki_prompt = f"""Extract any information about Sri Krishna Arts and Science College (SKASC) 
            from this Wikipedia content related to {user_query}: {wiki_result[:2000]}"""
            
            wiki_analysis = llm.invoke([HumanMessage(content=wiki_prompt)]).content
            
            return jsonify({
                "Wikipedia": [{
                    "title": "Wikipedia Information",
                    "authors": "",
                    "excerpt": wiki_analysis[:800] or wiki_result[:800],
                    "link": f"https://en.wikipedia.org/wiki/{user_query.replace(' ','_')}",
                    "affiliation_match": affiliation_matches(wiki_result),
                    "confidence": "low"
                }]
            })
    except Exception as e:
        return jsonify({"Error": str(e)})
    
    # If all fail
    return jsonify({
        "message": "No specific SKASC research papers found. Try broadening your search or check institutional repositories.",
        "suggestions": [
            "Check SKASC institutional repository directly",
            "Search Google Scholar with 'site:skasc.ac.in'",
            "Contact SKASC research department"
        ]
    })

if __name__ == "__main__":
    app.run(debug=True)