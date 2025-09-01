import streamlit as st
import asyncio
import os
import tiktoken
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

load_dotenv()

st.set_page_config(
    page_title="Product Development Agent",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Product Development Agent")
st.markdown("Generate comprehensive MVP briefs using AI agents")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
task_input = st.sidebar.text_area(
    "Product Brief Request:",
    value="Create a one-page MVP brief for a burnout monitoring software...",
    height=150
)

if st.sidebar.button("🎯 Generate Brief", type="primary"):
    with st.spinner("🤖 Agents are working..."):
        
        # Initialize models
        model_client_openai_1 = OpenAIChatCompletionClient(
            model="gpt-4.1-mini",
            api_key=os.environ["OPENAI_API_KEY"],
        )
        model_client_openai_2 = OpenAIChatCompletionClient(
            model="gpt-5-mini",
            api_key=os.environ["OPENAI_API_KEY"],
        )
        model_client_anthropic = AnthropicChatCompletionClient(
            model="claude-sonnet-4-20250514",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

        # Create agents
        ux_researcher = AssistantAgent(
            name="ux_researcher",
            model_client=model_client_openai_1,
            system_message=(
                "Role: UX Researcher. Break down user needs, JTBD, pain points, "
                "and success metrics. Produce concise bullets and cite assumptions. "
                "Say 'APPROVED' when your research notes are ready."
            ),
        )

        tech_lead = AssistantAgent(
            name="tech_lead",
            model_client=model_client_openai_2,
            system_message=(
                "Role: Tech Lead / Engineer. Propose a simple, feasible architecture, "
                "interfaces, and a minimal backlog. Flag risks and unknowns. "
                "Say 'APPROVED' when your plan is ready."
            ),
        )

        qa_compliance = AssistantAgent(
            name="qa_compliance",
            model_client=model_client_anthropic,
            system_message=(
                "Role: QA & Compliance. Review for safety, privacy, and regulatory fit "
                "(ISO/OSHA/IEC basics). List testable acceptance criteria. "
                "Say 'APPROVED' when review is ready."
            ),
        )

        workers_termination = TextMentionTermination(text="APPROVED") | MaxMessageTermination(max_messages=8)
        workers_team = RoundRobinGroupChat(
            participants=[ux_researcher, tech_lead, qa_compliance],
            termination_condition=workers_termination,
        )

        orchestrator_pm = SocietyOfMindAgent(
            name="orchestrator_pm",
            team=workers_team,
            model_client=model_client_openai_2,
        )

        synthesizer = AssistantAgent(
            name="synthesizer",
            model_client=model_client_openai_1,
            system_message=(
                "Role: Tech Writer (Synthesizer). Combine the orchestrator's summary and "
                "worker outputs into a crisp deliverable (one page). Include: "
                "Problem, Target Users, Proposed Solution, System Sketch, MVP Scope, "
                "Acceptance Criteria, Risks & Next Steps. End with 'FINALIZED'."
            ),
        )

        outer_termination = TextMentionTermination(text="FINALIZED") | MaxMessageTermination(max_messages=6)
        final_team = RoundRobinGroupChat(
            participants=[orchestrator_pm, synthesizer],
            termination_condition=outer_termination,
        )

        # Run the agent system
        async def run_agents():
            result = await final_team.run(task=task_input)
            return result

        # Execute and display results
        result = asyncio.run(run_agents())
        
        # Token usage tracking
        if result and result.messages:
            encoding = tiktoken.get_encoding("cl100k_base")
            
            input_tokens = 0
            output_tokens = 0
            
            for message in result.messages:
                if hasattr(message, 'content') and message.content:
                    tokens = len(encoding.encode(str(message.content)))
                    if hasattr(message, 'source') and message.source:
                        if 'user' in str(message.source).lower() or 'human' in str(message.source).lower():
                            input_tokens += tokens
                        else:
                            output_tokens += tokens
                    else:
                        output_tokens += tokens
            
            total_tokens = input_tokens + output_tokens
            
            # Display token usage
            st.markdown("### 📊 Token Usage")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📥 Input Tokens", f"{input_tokens:,}")
            with col2:
                st.metric("📤 Output Tokens", f"{output_tokens:,}")
            with col3:
                st.metric("🔢 Total Tokens", f"{total_tokens:,}")
        
        # Display results with icons
        st.success("✅ Brief Generated Successfully!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔍 UX Research")
            st.info("User needs and pain points analysis")
        
        with col2:
            st.markdown("### 🏗️ Technical Architecture")
            st.info("System design and implementation plan")
        
        with col3:
            st.markdown("### ✅ QA & Compliance")
            st.info("Quality assurance and regulatory review")
        
        st.markdown("### 📋 Final MVP Brief")
        
        # Extract final result
        if result and result.messages:
            final_message = result.messages[-1].content
            st.markdown(final_message)
        else:
            st.warning("No final result generated")

# Display agent status
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Agent Status")
st.sidebar.markdown("🔍 **UX Researcher**: Ready")
st.sidebar.markdown("🏗️ **Tech Lead**: Ready") 
st.sidebar.markdown("✅ **QA Compliance**: Ready")
st.sidebar.markdown("📋 **Product Manager**: Ready")
st.sidebar.markdown("✍️ **Tech Writer**: Ready")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Token Tracking")
st.sidebar.markdown("📥 Input tokens counted")
st.sidebar.markdown("📤 Output tokens counted")
st.sidebar.markdown("🔢 Total usage displayed")