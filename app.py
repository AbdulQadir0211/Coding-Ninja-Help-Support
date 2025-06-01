import streamlit as st
import requests
import asyncio
import aiohttp

API_URL = "http://localhost:8000/query"  # Change if deployed remotely

st.set_page_config(page_title="Coding Ninja AI Assistant", layout="centered")

st.title("🤖 Coding Ninja Multi-Agent System")
st.markdown("Ask me anything related to Coding Ninjas: customer support, code help, project ideas, course recommendations, interview prep & more!")

user_input = st.text_input("💬 Enter your query:", placeholder="e.g., I need help with Python code")

if st.button("Submit") and user_input.strip():
    st.info("Processing...")

    async def fetch_response():
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, json={"text": user_input}) as resp:
                if resp.status != 200:
                    return {"error": f"API Error: {resp.status}"}
                return await resp.json()

    response_data = asyncio.run(fetch_response())

    if "error" in response_data:
        st.error(response_data["error"])
    else:
        response = response_data["responses"][0]
        st.subheader(f"🎯 Agent: {response['agent_type'].replace('_', ' ').title()}")
        st.markdown(f"**📝 Response:**\n\n{response['content']}")

        if response["sources"]:
            st.markdown("🔗 **Sources:**")
            for src in response["sources"]:
                st.markdown(f"- [{src}]({src})")

        if response["suggested_actions"]:
            st.markdown("✅ **Suggested Actions:**")
            for action in response["suggested_actions"]:
                st.markdown(f"- {action}")
        
        st.success("✅ Response generated!")
