import pytest

from judge import assert_criteria
from sql_agent import agent
from utils import collect_tools


# Verified directly with DuckDB:
# SELECT COUNT(*) FROM trips WHERE passenger_count > 5;
EXPECTED_COUNT = 22413


@pytest.mark.asyncio
async def test_trips_with_more_than_five_passengers():
    result = await agent.run("How many trips had more than 5 passengers?")
    output = result.output

    print("SQL query:", output.sql_query)
    print("Result text:", output.result_text)

    assert isinstance(output.sql_query, str)
    assert output.sql_query.strip()
    assert str(EXPECTED_COUNT) in output.result_text


@pytest.mark.asyncio
async def test_payment_type_tool_calls():
    user_prompt = "What is the most common payment type?"
    result = await agent.run(user_prompt)
    messages = result.new_messages()
    tool_calls = collect_tools(messages)

    for index, call in enumerate(tool_calls, start=1):
        print(f"{index}. {call.name}({call.args})")

    assert len(tool_calls) >= 2
    assert tool_calls[0].name == "get_schema"
    assert any(call.name == "run_sql" for call in tool_calls)
    print("Second tool:", tool_calls[1].name)


@pytest.mark.asyncio
async def test_highest_average_fare_hour():
    user_prompt = "Which hour of the day has the highest average fare amount?"
    result = await agent.run(user_prompt)
    print("SQLResult:", result.output.model_dump_json(indent=2))
    print("Tool calls:", collect_tools(result.new_messages()))
    await assert_criteria(result, [
        "the SQL query correctly calculates average fare by hour of day",
        "the result identifies a specific hour as having the highest average fare",
        "the result includes the actual average fare amount",
    ])
