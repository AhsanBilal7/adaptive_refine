"""Official NATURAL PLAN scoring functions, vendored verbatim from
https://github.com/google-deepmind/natural-plan (evaluate_*.py; Apache-2.0,
Copyright 2024 Google LLC).  Only the absl CLI entry points and comment dividers were removed."""


# evaluate_calendar_scheduling.py
#

import json
import re


def hour_to_num(hr_str):
  return float(hr_str.split(':')[0]) + (
      0.5 if hr_str.split(':')[1] == '30' else 0.0
  )


def calendar_parse_response(response: str):
  """Parse the response.

  Returns a parsed suggested meeting time in (day, start_hour, end_hour).

  Args:
    response: Raw response from the model.

  Returns:
    A tuple of (day, start_hour, end_hour).
  """
  time_strs = re.findall(r'[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+', response)
  if not time_strs:
    return '', -1, -1
  # If multiple matches are found, return the first one.
  time_str = time_strs[0]
  day, hour_str = (
      time_str.split(',')[0].strip(),
      time_str.split(',')[1].strip(),
  )
  start_hour, end_hour = (
      hour_str.split('-')[0].strip(),
      hour_str.split('-')[1].strip(),
  )
  return day, hour_to_num(start_hour), hour_to_num(end_hour)


def compute_solve_rate(responses: list[str], solutions: list[str]):
  """Computes solve rate by comparing model responses to golden solutions.

  Args:
    responses: A list of model responses.
    solutions: The corresponding list of golden solutions for the same tasks.

  Returns:
    A scalr solve rate.
  """
  solved_count = 0

  for r, s in zip(responses, solutions):
    r_day, r_start_hour, r_end_hour = calendar_parse_response(r)
    s_day, s_start_hour, s_end_hour = calendar_parse_response(s)
    if (
        r_day == s_day
        and r_start_hour == s_start_hour
        and r_end_hour == s_end_hour
    ):
      solved_count += 1
  return float(solved_count) / len(responses)


# evaluate_meeting_planning.py
#

import collections
import datetime
import json
from typing import Any, Sequence


MAX_CONSTRAINTS = 10
NUM_SAMPLES_PER_CONSTRAINTS = 100


def convert_to_time_obj(time_str: str):
  return datetime.datetime.strptime(time_str, "%I:%M%p")


def process_constraints(data: tuple[Any, ...]):
  contraints = collections.defaultdict(dict)
  for name, location, times, meeting_time in data:
    contraints[name]["location"] = location

    start_time = convert_to_time_obj(times.split("to")[0].strip())
    end_time = convert_to_time_obj(times.split("to")[1].strip())
    contraints[name]["start_time"] = start_time
    contraints[name]["end_time"] = end_time
    contraints[name]["meeting_time"] = meeting_time
  return contraints


def validator_from_text(
    plan: list[str],
    processed_constraints: dict[str, Any],
    start_location: str,
    initial_time: str,
    dist_matrix: dict[str, Any],
):
  """Compute the number of valid meetings scheduled in the plan (text format)."""

  # This function is used when the generated plan follows the few-shot format.

  # Example:
  # You start at Russian Hill at 9:00AM.
  # You travel to Marina District in 7 minutes and arrive at 9:07AM.
  # You wait until 3:45PM.
  # You meet James for 75 minutes from 3:45PM to 5:00PM.

  met_with = {}

  score = 0
  cur_location = start_location
  cur_time = convert_to_time_obj(initial_time)
  for step in plan:
    try:
      if step.startswith("You start"):
        continue
      elif step.startswith("You travel"):
        destination = step.split("travel to ")[1].split(" in")[0].strip()
        cur_time = cur_time + datetime.timedelta(
            minutes=dist_matrix[cur_location][destination]
        )

        cur_location = destination
      elif step.startswith("You wait"):

        raw_end_time = step.split("wait until ")[1].split(".")[0].strip()
        end_time = convert_to_time_obj(raw_end_time)

        if end_time <= cur_time:
          raise ValueError("Cannot go backwards in time")

        cur_time = end_time
      elif step.startswith("You meet"):

        person = step.split("meet ")[1].split(" for")[0].strip()
        if person in met_with:
          raise ValueError(
              "Person {person} already met with {met_with}".format(
                  person=person, met_with=met_with[person]
              )
          )

        met_with[person] = 1
        new_time = cur_time + datetime.timedelta(
            minutes=processed_constraints[person]["meeting_time"]
        )

        if (
            cur_location == processed_constraints[person]["location"]
            and cur_time >= processed_constraints[person]["start_time"]
            and new_time <= processed_constraints[person]["end_time"]
        ):

          score += 1
          cur_time = new_time
        else:
          raise ValueError("Invalid meeting time or location")
      else:
        raise ValueError("Unknown plan format")

    except ValueError as e:
      print("Had error: {e} with step: {step}".format(e=e, step=step))
      break

  return score


def validator_from_dict(
    plan: list[Any],
    processed_constraints: dict[str, Any],
    start_location: str,
    initial_time: str,
    dist_matrix: dict[str, Any],
):
  """Compute the number of valid meetings scheduled in the plan (JSON format)."""

  # This function is used when the generated plan does not follow
  # the few-shot format. Before calling this function, the plan should be
  # converted to a list of steps, each of which is a dictionary with the
  # following fields:
  #     {"location": LOCATION,
  #     "person_name": PERSON_TO_MEET,
  #     "start_time": START_TIME}.

  # Example:
  # Plan (text):
  # You start at Russian Hill at 9:00AM.
  # You travel to Marina District in 7 minutes and arrive at 9:07AM.
  # You wait until 3:45PM.
  # You meet James for 75 minutes from 3:45PM to 5:00PM.

  # Corresponding plan format taken by this function (JSON):
  # ```json
  # [
  #     {
  #         "location": "Russian Hill",
  #         "person_name": "N/A",
  #         "start_time": "9:00AM"
  #     },
  #     {
  #         "location": "Marina District",
  #         "person_name": "N/A",
  #         "start_time": "9:07AM"
  #     },
  #     {
  #         "location": "Marina District",
  #         "person_name": "James",
  #         "start_time": "3:45PM"
  #     }
  # ]
  # ```

  met_with = {}

  score = 0
  cur_location = start_location
  cur_time = convert_to_time_obj(initial_time)
  for step in plan[1:]:
    try:
      raw_start_time = step["start_time"]
      location = step["location"]
      if location and location != cur_location:
        cur_time = cur_time + datetime.timedelta(
            minutes=dist_matrix[cur_location][location]
        )
        cur_location = location

      start_time = convert_to_time_obj(raw_start_time)
      if start_time < cur_time:
        raise ValueError("Start time too early")
      cur_time = start_time
      person = step["person_name"]
      if person not in processed_constraints:
        continue
      if person in met_with:
        raise ValueError(
            "Person {person} already met with {met_with}".format(
                person=person, met_with=met_with[person]
            )
        )

      met_with[person] = 1
      new_time = cur_time + datetime.timedelta(
          minutes=processed_constraints[person]["meeting_time"]
      )

      if (
          cur_location == processed_constraints[person]["location"]
          and cur_time >= processed_constraints[person]["start_time"]
          and new_time <= processed_constraints[person]["end_time"]
      ):

        score += 1
        cur_time = new_time
      else:
        raise ValueError("Invalid meeting time or location")

    except ValueError as e:
      print("Had error: {e} with step: {step}".format(e=e, step=step))
      break

  return score


def parse_text_plan(plan: str):
  """Parse the text plan into a list of steps."""

  prefix = "SOLUTION:"
  if prefix in plan:
    plan = plan[plan.find(prefix) + len(prefix):].strip()
  plan = plan.split(".")
  plan = [step.strip() for step in plan]
  final_plan = []
  for step in plan:
    if step:
      final_plan.append(step)
  return final_plan


# evaluate_trip_planning.py
#

import json
import re
from typing import Any


def parse_response(response: str):
  """Parse the response.

  Returns a parsed plan in a list of (city, stay_days) tuples.

  Args:
    response: Raw response from the model.

  Returns:
    Structured plan after parsing.
  """
  pattern_visit = r'\d+-\d+'
  pattern_flight = r'.*Day (\d+).*from (\w+) to (\w+)'
  pattern_days = r'European cities for (\d+) days'

  days, flights, flight_days = [], [], []
  total_days = None
  for piece in response.split('\n'):
    days_match = re.findall(pattern_days, piece)
    if days_match:
      total_days = int(days_match[0])

    visit_match = re.findall(pattern_visit, piece)
    if visit_match:
      days.append(visit_match[0])
      end_day = int(visit_match[0].split('-')[1])
      # Reach the end of the plan, stop to avoid parsing alternative plans.
      if end_day == total_days:
        break
    flight_match = re.findall(pattern_flight, piece)
    if flight_match:
      flights.append(flight_match[0])

  visit_cities, parsed_plan = [], []
  for flight_day, begin_city, end_city in flights:
    flight_days.append(int(flight_day))
    if not visit_cities:
      visit_cities.append(begin_city)
      visit_cities.append(end_city)
    else:
      visit_cities.append(end_city)

  if not days or not flights or not visit_cities:
    return []
  last_day = int(days[-1].split('-')[1])
  flight_days = [1] + flight_days + [last_day]
  for i, visit_city in enumerate(visit_cities):
    city_stay = flight_days[i + 1] - flight_days[i] + 1
    parsed_plan.append((visit_city, city_stay))

  return parsed_plan


def compute_example_score(cities: str, durations: str, parsed_plan: list[Any]):
  """Compute the exact-match accuracy.

  Compute the example-level exact_match score (0/1) given the parsed plan
  and the ground truth in the format of durations and cities.

  Args:
    cities: The cities in the plan in the format of "city1**city2**city3".
    durations: The durations of the stay in each city in the format of
      "1**2**3".
    parsed_plan: The parsed plan from the response.

  Returns:
    Exact-match accuracy of 0 (mismatched) or 1 (matched).
  """

  stays = [x for x in cities.split('**') if x]
  days = [int(x) for x in durations.split('**') if x]
  num_stays = min(len(stays), len(parsed_plan))
  num_match = 0
  for i in range(num_stays):
    if stays[i] == parsed_plan[i][0] and days[i] == parsed_plan[i][1]:
      num_match += 1
    else:
      break
  hard_score = 0.0 if num_match / len(stays) < 1.0 else 1.0
  return hard_score


def compute_score(
    cities: list[str], durations: list[str], responses: list[str]
):
  """Compute the sample-level exact-match accuracy.

  Args:
    cities: List of cities in the plan in the format of "city1**city2**city3".
    durations: List of durations of the stay in each city in the format of
      "1**2**3".
    responses: The raw responses from the model.

  Returns:
    Exact-match score at the sample level.
  """
  parsed_plans = [parse_response(response) for response in responses]
  hard_scores = [
      compute_example_score(city, duration, parsed_plan)
      for city, duration, parsed_plan in zip(cities, durations, parsed_plans)
  ]
  hard_acc = sum(hard_scores) / len(hard_scores)
  return hard_acc

