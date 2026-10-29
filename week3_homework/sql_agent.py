import asyncio

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart

from sql_tools import SQLTools


class SQLResult(BaseModel):
    sql_query: str = Field(description="The SQL query executed to answer the question.")
    result_text: str = Field(description="Relevant query result, including headers and exact values.")
    row_count: int = Field(description="Number of result rows returned by SQL, excluding headers.")


sql_tools = SQLTools()

agent = Agent(
    "openai:gpt-4o-mini",
    output_type=SQLResult,
    tools=[
        sql_tools.get_schema,
        sql_tools.run_sql,
    ],
    instructions="""
    You are a SQL assistant for the NYC Yellow Taxi trips table in DuckDB.
    Always call get_schema() first, before writing or running any SQL.
    Wait for its result and use the schema to determine column names and types.
    Then call run_sql(query) to execute the SQL needed to answer the question.
    Base your final answer only on the returned SQL result.
    Return a structured SQLResult with sql_query containing the query used,
    result_text containing the relevant result with headers and exact numeric
    values (do not round), and row_count containing the number of returned
    result rows, excluding the header. An aggregate returning one row has
    row_count = 1; this is not the total number of taxi records or the value
    inside a COUNT(*) result. run_sql returns at most 50 rows.
    """,
)


async def main():
    result = await agent.run("What's the average trip distance for rides with 2 passengers?")
    print("Structured SQLResult:")
    print(result.output.model_dump_json(indent=2))
    print("\nSQL query:")
    print(result.output.sql_query)
    print("\nSQL result / average trip distance:")
    print(result.output.result_text)
    print("\nTool-call order:")
    print(" -> ".join(
        part.tool_name
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart)
        and part.tool_name in {"get_schema", "run_sql"}
    ))


if __name__ == "__main__":
    asyncio.run(main())
