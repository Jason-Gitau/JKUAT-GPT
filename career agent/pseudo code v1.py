from fastapi import FastAPI, Request
import openai
from tools import linkedin_api, github_api, web_search

app = FastAPI()

# === Router ===
@app.post("/query")
async def handle_query(req: Request):
    data = await req.json()
    query = data["query"]
    user_id = data["user_id"]

    if is_study_query(query):
        return await handle_study_query(query, user_id)
    else:
        return await handle_career_query(query, user_id)

# === Career Agent ===
async def handle_career_query(query, user_id):
    # Step 1: Ask LLM what tool (if any) it needs
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a career advisor agent."},
                  {"role": "user", "content": query}],
        functions=[
            {
                "name": "search_jobs",
                "description": "Search jobs on LinkedIn",
                "parameters": {"type": "object","properties": {"keyword": {"type": "string"}}}
            },
            {
                "name": "fetch_github_repos",
                "description": "Find trending GitHub repos for projects",
                "parameters": {"type": "object","properties": {"topic": {"type": "string"}}}
            },
            {
                "name": "web_search",
                "description": "Do a general web search",
                "parameters": {"type": "object","properties": {"query": {"type": "string"}}}
            }
        ],
        function_call="auto"
    )

    # Step 2: If tool is requested, call it
    if response.get("choices")[0]["finish_reason"] == "function_call":
        fn_call = response["choices"][0]["message"]["function_call"]
        tool_name = fn_call["name"]
        args = fn_call["arguments"]

        if tool_name == "search_jobs":
            tool_result = linkedin_api.search_jobs(args["keyword"])
        elif tool_name == "fetch_github_repos":
            tool_result = github_api.fetch_repos(args["topic"])
        elif tool_name == "web_search":
            tool_result = web_search(args["query"])
        else:
            tool_result = "Unknown tool"

        # Step 3: Give result back to LLM for final answer
        final = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a career advisor agent."},
                {"role": "user", "content": query},
                {"role": "function", "name": tool_name, "content": str(tool_result)}
            ]
        )
        return {"answer": final["choices"][0]["message"]["content"]}

    # Step 4: No tool needed → just answer directly
    else:
        return {"answer": response["choices"][0]["message"]["content"]}
