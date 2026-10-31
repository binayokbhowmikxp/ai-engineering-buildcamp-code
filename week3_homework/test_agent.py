import math
import re

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


def _assert_scenario(result, columns, expected_values):
    """Check the shared SQL, trajectory, and numeric facts for Question 6."""
    tool_calls = collect_tools(result.new_messages())
    print("SQLResult:", result.output.model_dump_json(indent=2))
    print("Tool calls:", tool_calls)
    assert len(tool_calls) >= 2
    assert tool_calls[0].name == "get_schema"
    assert any(call.name == "run_sql" for call in tool_calls)
    sql = result.output.sql_query.lower()
    for column in columns:
        assert column.lower() in sql

    # Accept plain, table, or prose output, including thousands separators.
    numbers = [
        float(token.replace(",", ""))
        for token in re.findall(
            r"(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])",
            result.output.result_text,
        )
    ]
    for expected in expected_values:
        if isinstance(expected, int):
            assert expected in numbers, f"Expected exact integer {expected} in result_text"
        else:
            assert any(math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-6)
                       for value in numbers), f"Expected approximately {expected} in result_text"
    return numbers


# Ground truth below was independently queried from the fixed January 2024 DB.
@pytest.mark.asyncio
async def test_average_tip_credit_card():
    result = await agent.run("What is the average tip amount for credit card payments?")
    _assert_scenario(result, ["tip_amount", "payment_type"], [4.169670627490382])


@pytest.mark.asyncio
async def test_pickup_location_most_trips():
    result = await agent.run("Which pickup location (PULocationID) has the most trips?")
    _assert_scenario(result, ["PULocationID"], [132, 145240])


@pytest.mark.asyncio
async def test_average_fare_long_trips():
    result = await agent.run("What is the average fare for trips longer than 10 miles?")
    _assert_scenario(result, ["fare_amount", "trip_distance"], [62.880812428259745])


@pytest.mark.asyncio
async def test_zero_passengers():
    result = await agent.run("How many trips had zero passengers recorded?")
    _assert_scenario(result, ["passenger_count"], [31465])


@pytest.mark.asyncio
async def test_busiest_day_of_week():
    result = await agent.run("What is the busiest day of the week for taxi trips?")
    numbers = _assert_scenario(result, ["tpep_pickup_datetime"], [495032])
    # DuckDB DOW uses Sunday=0; Wednesday=3. Day names are equally valid.
    assert "wednesday" in result.output.result_text.lower() or 3 in numbers
