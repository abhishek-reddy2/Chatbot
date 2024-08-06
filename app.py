import time
import numpy
import datetime  # Added import for datetime
from google.cloud import bigquery
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vertexai.generative_models import FunctionDeclaration, GenerativeModel, Part, Tool

BIGQUERY_DATASET_ID = "Sample_Fin_Dataset"

list_datasets_func = FunctionDeclaration(
    name="list_datasets",
    description="Get a list of datasets that will help answer the user's question",
    parameters={ "type": "object", "properties": {} },
)

list_tables_func = FunctionDeclaration(
    name="list_tables",
    description="List tables in a dataset that will help answer the user's question",
    parameters={
        "type": "object",
        "properties": { "dataset_id": { "type": "string", "description": "Dataset ID to fetch tables from." }},
        "required": ["dataset_id"],
    },
)

get_table_func = FunctionDeclaration(
    name="get_table",
    description="Get information about a table, including the description, schema, and number of rows that will help answer the user's question. Always use the fully qualified dataset and table names.",
    parameters={
        "type": "object",
        "properties": { "table_id": { "type": "string", "description": "Fully qualified ID of the table to get information about" }},
        "required": ["table_id"],
    },
)

sql_query_func = FunctionDeclaration(
    name="sql_query",
    description="Get information from data in BigQuery using SQL queries",
    parameters={
        "type": "object",
        "properties": { "query": { "type": "string", "description": "SQL query on a single line that will help give quantitative answers to the user's question when run on a BigQuery dataset and table. In the SQL query, always use the fully qualified dataset and table names." }},
        "required": ["query"],
    },
)

sql_query_tool = Tool(
    function_declarations=[ list_datasets_func, list_tables_func, get_table_func, sql_query_func ],
)

model = GenerativeModel(
    "gemini-1.5-pro-001",
    generation_config={"temperature": 1},
    tools=[sql_query_tool],
)

st.set_page_config(
    page_title="Brio",
    page_icon="b.png",
    layout="wide",
)

st.image("brio.png", width=100)

st.title("Finance Q&A Using Enterprise Search")

# Divide the page into two columns
col1, col2 = st.columns([0.4, 0.6])

# Left column for prompts
with col1:
    st.header("Prompt")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if prompt := st.chat_input("Ask me about information in the database..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Save prompt to prompts.txt
        with open("prompts.txt", "a") as f:
            f.write(prompt + "\n")

    st.subheader("Session Prompts")
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(message["content"].replace("$", "\$"))

# Right column for responses
with col2:
    st.header("Response")
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                chat = model.start_chat()
                client = bigquery.Client()

                prompt = last_message["content"] + """
                    Please give a concise, high-level summary followed by detail in
                    plain language about where the information in your response is
                    coming from in the database. Only use information that you learn
                    from BigQuery, do not make up information.
                    """
                try:
                    response = chat.send_message(prompt)
                    response = response.candidates[0].content.parts[0]

                    print(response)

                    api_requests_and_responses = []
                    backend_details = ""
                    api_response = None  # Initialize api_response to avoid 'not defined' error

                    function_calling_in_process = True
                    while function_calling_in_process:
                        try:
                            params = {}
                            for key, value in response.function_call.args.items():
                                params[key] = value

                            print(response.function_call.name)
                            print(params)

                            if response.function_call.name == "list_datasets":
                                api_response = client.list_datasets()
                                api_response = BIGQUERY_DATASET_ID
                                api_requests_and_responses.append(
                                    [response.function_call.name, params, api_response]
                                )

                            if response.function_call.name == "list_tables":
                                api_response = client.list_tables(params["dataset_id"])
                                api_response = str([table.table_id for table in api_response])
                                api_requests_and_responses.append(
                                    [response.function_call.name, params, api_response]
                                )

                            if response.function_call.name == "get_table":
                                api_response = client.get_table(params["table_id"])
                                api_response = api_response.to_api_repr()
                                api_requests_and_responses.append(
                                    [
                                        response.function_call.name,
                                        params,
                                        [
                                            str(api_response.get("description", "")),
                                            str(
                                                [
                                                    column["name"]
                                                    for column in api_response["schema"]["fields"]
                                                ]
                                            ),
                                        ],
                                    ]
                                )
                                api_response = str(api_response)

                            if response.function_call.name == "sql_query":
                                job_config = bigquery.QueryJobConfig(
                                    maximum_bytes_billed=100000000
                                )  # Data limit per query job
                                try:
                                    cleaned_query = (
                                        params["query"]
                                        .replace("\\n", " ")
                                        .replace("\n", "")
                                        .replace("\\", "")
                                    )
                                    query_job = client.query(cleaned_query, job_config=job_config)
                                    api_response = query_job.result()
                                    api_response = str([dict(row) for row in api_response])
                                    api_response = api_response.replace("\\", "").replace("\n", "")
                                    api_requests_and_responses.append(
                                        [response.function_call.name, params, api_response]
                                    )
                                except Exception as e:
                                    api_response = f"{str(e)}"
                                    api_requests_and_responses.append(
                                        [response.function_call.name, params, api_response]
                                    )

                            print(api_response)

                            response = chat.send_message(
                                Part.from_function_response(
                                    name=response.function_call.name,
                                    response={
                                        "content": api_response,
                                    },
                                ),
                            )
                            response = response.candidates[0].content.parts[0]

                            backend_details += "- Function call:\n"
                            backend_details += (
                                "   - Function name: ```"
                                + str(api_requests_and_responses[-1][0])
                                + "```"
                            )
                            backend_details += "\n\n"
                            backend_details += (
                                "   - Function parameters: ```"
                                + str(api_requests_and_responses[-1][1])
                                + "```"
                            )
                            backend_details += "\n\n"
                            backend_details += (
                                "   - API response: ```"
                                + str(api_requests_and_responses[-1][2])
                                + "```"
                            )
                            backend_details += "\n\n"
                            with message_placeholder.container():
                                st.markdown(backend_details)

                        except AttributeError:
                            function_calling_in_process = False

                    time.sleep(3)

                    full_response = response.text
                    with message_placeholder.container():
                        st.markdown(full_response.replace("$", "\$"))
                        with st.expander("Function calls, parameters, and responses:"):
                            st.markdown(backend_details)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "backend_details": backend_details,
                        }
                    )

                    # Convert API response to DataFrame
                    if api_response:
                        try:
                            # Check if `api_response` is a JSON-like string
                            df = pd.DataFrame(eval(api_response))
                            # Display response as text if there's only one column and one row
                            if len(df.columns) == 1:
                                st.write("### Text Response")
                                st.write(df.iloc[0, 0])

                            # Display response as pie chart if there are two columns
                            elif len(df.columns) == 2:
                                st.write("### Pie Chart")
                                fig, ax = plt.subplots()
                                df.set_index(df.columns[0]).plot.pie(y=df.columns[1], figsize=(5, 5), autopct='%1.1f%%', ax=ax)
                                ax.legend(
                                    loc='upper right',
                                    bbox_to_anchor=(1.3, 1.15),
                                    fontsize='xx-small',
                                    title='Legend',
                                    ncol=2,
                                    markerscale=0.5
                                )
                                st.pyplot(fig)

                            # Display response as bar chart if there are three or more columns
                            else:
                                st.write("### Bar Chart")
                                st.bar_chart(df.set_index(df.columns[0]))

                        except Exception as e:
                            st.error(f"Error converting response to DataFrame or plotting: {e}")

                except Exception as e:
                    st.error(f"Error: {e}")

st.markdown("</div>", unsafe_allow_html=True)
